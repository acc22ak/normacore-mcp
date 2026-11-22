"""M0 hardware check. Read-only unless --home is given.

  .venv/bin/python tools/smoke.py           # joints, limits, camera frames (saved to logs/)
  .venv/bin/python tools/smoke.py --home    # ...then glide to the home pose at half speed, after you confirm
"""
import argparse
import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from arm import Arm, load_yaml  # noqa: E402
from skills import Skills  # noqa: E402
from station_io import RealIO, StationError  # noqa: E402


async def main(go_home: bool) -> None:
    io = RealIO()
    t = time.perf_counter()
    try:
        await io.connect()
    except StationError as e:
        sys.exit(f"FAIL: {e}")
    print(f"follower live ({(time.perf_counter() - t) * 1000:.0f} ms)")

    arm = Arm()
    joints = await io.read_joints()
    arm.update_ranges(joints)
    print(f"\n{'joint':15} {'norm':>6} {'deg':>7}  {'limits (norm)':>14}  current  error")
    for n, m in joints.items():
        lo, hi = arm.limits(n) if n in arm.joints else (None, None)
        flag = "" if lo is None or lo <= m["position_norm"] <= hi else "  <- outside limits"
        print(f"{n:15} {m['position_norm']:6.3f} {arm.norm_to_deg(n, m['position_norm']):7.1f}  "
              f"{lo:6.2f}..{hi:<6.2f}  {m['current']:7}  {m['error_status']:5}{flag}")

    for role in ("desk", "gripper"):
        try:
            t = time.perf_counter()
            jpeg, age = await io.read_frame(role)
            out = ROOT / "logs" / f"smoke_{role}.jpg"
            out.parent.mkdir(exist_ok=True)
            out.write_bytes(jpeg)
            print(f"{role} camera: {len(jpeg)} bytes, frame {age:.2f} s old, read in "
                  f"{(time.perf_counter() - t) * 1000:.0f} ms -> {out}")
        except StationError as e:
            print(f"{role} camera: FAIL {e}")

    if not go_home:
        return
    home = load_yaml("poses.yaml")["home"]
    print("\nPlanned move to HOME (current -> target):")
    for n, v in home.items():
        print(f"  {n:15} {joints[n]['position_norm']:.3f} -> {v:.3f}")
    if input("Area around the arm clear? Type yes to move: ").strip().lower() != "yes":
        return print("not moving")
    r = await Skills(io, arm).run({"do": "gesture", "name": "home"})
    print(f"reached={r['reached']} abort={r['abort_reason']} errors={r['errors']} "
          f"elapsed={r['elapsed_s']}s clamps={r['clamps']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", action="store_true")
    asyncio.run(main(ap.parse_args().home))
