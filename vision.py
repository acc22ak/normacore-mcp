"""Pure-Python image detection for camera-guided pick and place (no numpy/OpenCV needed).

Frames from Station are small JPEGs (shortest side 224 px), so a downsampled pixel scan in
plain Python is fast enough (a few ms). Two detectors:

- dark_blobs(): the motor in the gripper camera. It is near-black (HSV value ~0.08) on a
  light wooden table (value ~0.8). The gripper jaws are just as dark, so the two jaw
  regions at the bottom corners of the gripper image are masked out.
- marker_blobs(): a strongly coloured marker (green/blue sticker) on the box floor.
  Shaded cardboard and the wooden table have nearly the same orange colour, so the box
  itself can't be found by colour; a marker in a hue the scene doesn't contain can.

All coordinates returned are in full-resolution pixels of the input image, plus the
image size, so callers can normalise.
"""

from __future__ import annotations

import colorsys
import io
from dataclasses import dataclass, asdict

from PIL import Image, ImageDraw


@dataclass
class Blob:
    cx: float        # centroid x (px)
    cy: float        # centroid y (px)
    left: int
    top: int
    right: int
    bottom: int
    area: int        # full-res pixel count (approx)
    fill: float      # area / bbox area

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def as_dict(self) -> dict:
        d = asdict(self)
        d["cx"], d["cy"], d["fill"] = round(self.cx, 1), round(self.cy, 1), round(self.fill, 2)
        return d


def decode(jpeg: bytes) -> Image.Image:
    return Image.open(io.BytesIO(jpeg)).convert("RGB")


def _hsv_grid(img: Image.Image, step: int):
    w, h = img.size
    small = img.resize((w // step, h // step), Image.BILINEAR)
    sw, sh = small.size
    px = small.load()
    grid = [[colorsys.rgb_to_hsv(*(c / 255 for c in px[x, y])) for x in range(sw)] for y in range(sh)]
    return grid, sw, sh


def _components(mask: list[list[bool]], sw: int, sh: int, step: int) -> list[Blob]:
    seen = [[False] * sw for _ in range(sh)]
    blobs = []
    for y0 in range(sh):
        for x0 in range(sw):
            if not mask[y0][x0] or seen[y0][x0]:
                continue
            stack = [(x0, y0)]
            seen[y0][x0] = True
            n = sx = sy = 0
            l, t, r, b = x0, y0, x0, y0
            while stack:
                x, y = stack.pop()
                n += 1
                sx += x
                sy += y
                l, t, r, b = min(l, x), min(t, y), max(r, x), max(b, y)
                for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if 0 <= nx < sw and 0 <= ny < sh and mask[ny][nx] and not seen[ny][nx]:
                        seen[ny][nx] = True
                        stack.append((nx, ny))
            bbox = (r - l + 1) * (b - t + 1)
            blobs.append(Blob(cx=(sx / n + 0.5) * step, cy=(sy / n + 0.5) * step,
                              left=l * step, top=t * step, right=(r + 1) * step, bottom=(b + 1) * step,
                              area=n * step * step, fill=n / bbox))
    return blobs


def jaw_mask_fn(w: int, h: int, jaw_y: float, jaw_left_x: float, jaw_right_x: float):
    """True where a pixel belongs to the (open) gripper jaws in the gripper image."""
    def is_jaw(x: float, y: float) -> bool:
        return y >= jaw_y * h and (x <= jaw_left_x * w or x >= jaw_right_x * w)
    return is_jaw


def dark_blobs(img: Image.Image, *, max_v: float = 0.25, step: int = 2, mask_jaws: bool = True,
               jaw_y: float = 0.66, jaw_left_x: float = 0.27, jaw_right_x: float = 0.64,
               min_area_frac: float = 0.004, min_width_frac: float = 0.06, min_fill: float = 0.3) -> list[Blob]:
    """Dark, compact blobs (largest first). Thin dark things like cables fail min_width/min_fill."""
    w, h = img.size
    grid, sw, sh = _hsv_grid(img, step)
    is_jaw = jaw_mask_fn(w, h, jaw_y, jaw_left_x, jaw_right_x) if mask_jaws else (lambda x, y: False)
    mask = [[grid[y][x][2] <= max_v and not is_jaw((x + 0.5) * step, (y + 0.5) * step) for x in range(sw)]
            for y in range(sh)]
    blobs = [b for b in _components(mask, sw, sh, step)
             if b.area >= min_area_frac * w * h and b.width >= min_width_frac * w and b.fill >= min_fill]
    return sorted(blobs, key=lambda b: -b.area)


def marker_blobs(img: Image.Image, *, hue: tuple[float, float] = (0.20, 0.75), min_s: float = 0.35,
                 min_v: float = 0.25, step: int = 2, min_area_frac: float = 0.002) -> list[Blob]:
    """Saturated blobs whose hue lies in `hue` (default green..blue; the table and cardboard are
    orange, hue ~0.05-0.12). Largest first."""
    w, h = img.size
    grid, sw, sh = _hsv_grid(img, step)
    lo, hi = hue
    mask = [[(lo <= grid[y][x][0] <= hi) and grid[y][x][1] >= min_s and grid[y][x][2] >= min_v
             for x in range(sw)] for y in range(sh)]
    blobs = [b for b in _components(mask, sw, sh, step) if b.area >= min_area_frac * w * h]
    return sorted(blobs, key=lambda b: -b.area)


def _dilate(mask: list[list[bool]], sw: int, sh: int) -> list[list[bool]]:
    return [[mask[y][x] or any(mask[ny][nx] for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))
                               if 0 <= nx < sw and 0 <= ny < sh) for x in range(sw)] for y in range(sh)]


def white_blobs(img: Image.Image, *, max_s: float = 0.38, min_v: float = 0.85, step: int = 2,
                min_area_frac: float = 0.004, max_area_frac: float = 0.15, min_fill: float = 0.48,
                border: float = 0.02, min_ink: float = 0.02) -> list[Blob]:
    """Bright, low-saturation blobs (a white sticker/label). Cardboard and the wooden table are
    saturated orange (s > 0.5), so white stands out by saturation, not by hue. One dilation
    joins the fragments that printed text splits a sticker into. Blobs touching the image
    border are dropped (partly visible background objects). Largest first."""
    w, h = img.size
    grid, sw, sh = _hsv_grid(img, step)
    mask = _dilate([[grid[y][x][1] <= max_s and grid[y][x][2] >= min_v for x in range(sw)] for y in range(sh)], sw, sh)
    bx, by = border * w, border * h
    blobs = [b for b in _components(mask, sw, sh, step)
             if min_area_frac * w * h <= b.area <= max_area_frac * w * h and b.fill >= min_fill
             and b.left > bx and b.top > by and b.right < w - bx and b.bottom < h - by]
    # a label has ink on it (text/logo); glare on cardboard or wood is blank
    blobs = [b for b in blobs if _ink_frac(grid, b, step) >= min_ink]
    return sorted(blobs, key=lambda b: -b.area)


def _ink_frac(grid, b: Blob, step: int, inset: float = 0.15, max_v: float = 0.55) -> float:
    """Fraction of dark pixels inside the blob's box (inset away from its edges)."""
    dx, dy = (b.right - b.left) * inset, (b.bottom - b.top) * inset
    x0, x1 = int((b.left + dx) / step), int((b.right - dx) / step)
    y0, y1 = int((b.top + dy) / step), int((b.bottom - dy) / step)
    cells = [grid[y][x][2] for y in range(y0, y1) for x in range(x0, x1)]
    return sum(v <= max_v for v in cells) / len(cells) if cells else 0.0


def annotate(img: Image.Image, boxes: list[tuple[Blob, str]], points: list[tuple[float, float, str]] = ()) -> bytes:
    """JPEG with labelled blob boxes and target crosses drawn on it (for look-style feedback)."""
    im = img.copy()
    d = ImageDraw.Draw(im)
    for b, label in boxes:
        d.rectangle([b.left, b.top, b.right, b.bottom], outline=(0, 255, 0), width=2)
        d.text((b.left + 2, max(0, b.top - 11)), label, fill=(0, 255, 0))
    for x, y, label in points:
        d.line([x - 6, y, x + 6, y], fill=(255, 0, 255), width=2)
        d.line([x, y - 6, x, y + 6], fill=(255, 0, 255), width=2)
        if label:
            d.text((x + 7, y - 6), label, fill=(255, 0, 255))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return buf.getvalue()
