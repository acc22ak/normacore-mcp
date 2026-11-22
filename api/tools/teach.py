"""Save the follower's current pose into config/poses.yaml (comments are kept).

  .venv/bin/python tools/teach.py home       # or ready, A, B, C

Put the arm in place first: with Station teleop (leader -> follower mirroring) or by hand with
torque off. Teaching only reads joints; it never moves the arm.
"""
import asyncio
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from station_io import RealIO, StationError  # noqa: E402

NAMES = {"home": "", "ready": "", "A": "  ", "B": "  ", "C": "  "}   # name -> indent in poses.yaml


async def main(name: str) -> None:
    io = RealIO()
    try:
        await io.connect()
    except StationError as e:
        if "mirroring" not in str(e):      # teaching through teleop is fine: we only read
            sys.exit(f"FAIL: {e}")
    joints = await io.read_joints()
    pose = ", ".join(f"{n}: {m['position_norm']:.3f}" for n, m in joints.items())
    path = ROOT / "config" / "poses.yaml"
    text = path.read_text(encoding="utf-8")
    pat = re.compile(rf"^{NAMES[name]}{re.escape(name)}:.*$", re.M)
    if not pat.search(text):
        sys.exit(f"no '{name}:' line in {path}")
    path.write_text(pat.sub(f"{NAMES[name]}{name}: {{{pose}}}", text, count=1), encoding="utf-8")
    print(f"saved {name}: {{{pose}}}")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in NAMES:
        sys.exit(f"usage: teach.py {{{'|'.join(NAMES)}}}")
    asyncio.run(main(sys.argv[1]))
