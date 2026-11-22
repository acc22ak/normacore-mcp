"""Prepare the camera frame Claude sees: crop to the table, downscale, draw slot labels, re-encode.
Image tokens are roughly w*h/750, so a 768x432 frame is about 440 tokens."""
from __future__ import annotations

import io

from PIL import Image, ImageDraw, ImageFont


def prepare(jpeg: bytes, scene: dict) -> tuple[bytes, tuple[int, int]]:
    img = Image.open(io.BytesIO(jpeg)).convert("RGB")
    crop = scene.get("crop")
    slots = dict(scene.get("slots") or {})
    if crop:
        x0, y0 = crop[0], crop[1]
        img = img.crop(tuple(crop))
        slots = {k: [b[0] - x0, b[1] - y0, b[2] - x0, b[3] - y0] for k, b in slots.items()}
    scale = min(1.0, float(scene.get("max_long_side", 768)) / max(img.size))
    if scale < 1.0:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
        slots = {k: [round(v * scale) for v in b] for k, b in slots.items()}
    if slots:
        d = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
        except OSError:
            font = ImageFont.load_default()
        for label, (x0, y0, x1, y1) in sorted(slots.items()):
            d.rectangle([x0, y0, x1, y1], outline=(255, 210, 0), width=3)
            d.rectangle([x0, y0, x0 + 30, y0 + 32], fill=(255, 210, 0))
            d.text((x0 + 6, y0 + 1), label, fill=(0, 0, 0), font=font)
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=int(scene.get("jpeg_quality", 80)))
    return out.getvalue(), img.size
