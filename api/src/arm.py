"""Arm model: joint names, limits and degree <-> normalized conversion. Pure logic, no I/O.

Positions are normalized 0..1 inside each joint's calibrated range (Station's convention).
ST3215: 4096 steps per turn. Degrees are measured from step 2048, which this arm's calibrated
ranges are all centred on (assumed to be the URDF zero; unverified on the arm).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
STEPS = 4096
CENTER = 2048


def load_yaml(name: str) -> dict:
    return yaml.safe_load((ROOT / "config" / name).read_text(encoding="utf-8")) or {}


@dataclass(frozen=True)
class Joint:
    name: str
    id: int
    plus: str                   # what + does, in words, for Claude
    urdf_deg: tuple[float, float]
    soft: tuple[float, float] | None


class Arm:
    def __init__(self, cfg: dict | None = None):
        cfg = cfg or load_yaml("arm.yaml")
        soft = cfg.get("soft_limits") or {}
        self.extra_limits = bool(cfg.get("use_extra_limits", True))
        self.joints: dict[str, Joint] = {
            name: Joint(name, int(j["id"]), j["plus"], tuple(j["urdf_deg"]),
                        tuple(soft[name]) if name in soft else None)
            for name, j in cfg["joints"].items()
        }
        self.jog_joints: list[str] = list(cfg["jog_joints"])
        self.jog_max_deg: float = float(cfg["jog_max_deg"])
        self.follower_bus: str = cfg["follower_bus"]
        # live calibration: name -> (range_min, range_max), filled by update_ranges()
        self.ranges: dict[str, tuple[int, int]] = {}

    # -- calibration -------------------------------------------------------------
    def update_ranges(self, motors: dict[str, dict]) -> None:
        """motors: joint name -> {"range_min", "range_max", ...} from station_io.read_joints()."""
        self.ranges = {n: (int(m["range_min"]), int(m["range_max"])) for n, m in motors.items()}

    def span(self, name: str) -> int:
        lo, hi = self.ranges[name]
        return hi - lo if hi >= lo else (STEPS - lo) + hi

    # -- conversions ---------------------------------------------------------------
    def deg_to_norm(self, name: str, deg: float) -> float:
        lo, _ = self.ranges[name]
        raw = CENTER + deg * STEPS / 360.0
        rel = (raw - lo) % STEPS if self.ranges[name][1] < lo else raw - lo
        return rel / self.span(name)

    def norm_to_deg(self, name: str, norm: float) -> float:
        lo, _ = self.ranges[name]
        raw = lo + norm * self.span(name)
        return (raw - CENTER) * 360.0 / STEPS

    def deg_per_norm(self, name: str) -> float:
        return self.span(name) * 360.0 / STEPS

    def limits(self, name: str) -> tuple[float, float]:
        """Normalized hard limits: the narrowest of the calibrated range, URDF limits and soft limits."""
        j = self.joints[name]
        if not self.extra_limits:   # calibrated range only
            return 0.0, 1.0
        lo = max(0.0, self.deg_to_norm(name, j.urdf_deg[0]))
        hi = min(1.0, self.deg_to_norm(name, j.urdf_deg[1]))
        if j.soft:
            lo, hi = max(lo, j.soft[0]), min(hi, j.soft[1])
        return (lo, hi) if lo <= hi else (hi, hi)

    def clamp(self, name: str, norm: float) -> tuple[float, bool]:
        lo, hi = self.limits(name)
        v = min(hi, max(lo, norm))
        return v, abs(v - norm) > 1e-9
