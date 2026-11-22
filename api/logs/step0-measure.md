# Step 0 — MCP measure (2026-11-22, office network, no model calls)

Scripts: `bench/measure_io.py` (raw I/O, read-only), `bench/mcp_transcripts.py` (past Claude Code runs),
`bench/probe_read_latency.py`. Raw data: `io_bench.json`, `mcp_transcripts.json`.

## Raw I/O, 30 trials, ms (median / p90)

| arm | median | p90 |
|---|---|---|
| direct Station read, joints (1 entry) | 47.6 | 51.8 |
| direct Station read, camera (1 entry) | 47.5 | 49.9 |
| MCP handler in-process, get_robot_state | 106.5 | 119.6 |
| MCP handler in-process, look(gripper) | 87.4 | 125.7 |
| MCP stdio, get_robot_state | 126.5 | 146.1 |
| MCP stdio, look(gripper) | 100.5 | 140.3 |
| MCP stdio, spawn + initialize (once) | 3768 | — |
| API: TCP+TLS connect | 51 | 65 |
| API: warm HTTPS round trip (401, no key) | 124 | 134 |

- MCP stdio transport adds only ~15–20 ms over calling the handlers in-process.
- Every Station queue read costs ~45 ms regardless of size; the delay happens inside Station before the
  first entry is returned (client has TCP_NODELAY; single-write request makes no difference).
  get_robot_state does 2 reads (~107 ms), look does 2+ (inference-states index + camera queue).
- Only the gripper cam (FHD C3, 32e6:9251) streams. The standalone Web Camera (32e6:9228) fails every
  format (libuvc -99/-51); it shares a USB 2.0 hub with the gripper cam — likely isochronous bandwidth.
  Fixed: moved to a laptop port, streams MJPG 640x360@30 (stored at ~398x224), frame age ~0.15 s.

## Past MCP runs in Claude Code (31 instructions, 3 sessions, 2026-11-21, auto permission mode)

- Model turns per instruction: median 10 (range 2–58). Model time per turn: median 3.1 s.
- Tool round trip as seen by Claude Code: look 1.46 s (p90 4.6), get_robot_state 1.05 s (p90 2.9),
  nudge_joints 0.66 s, move_joints 0.60 s, go_to 3.7 s (includes motion).
- look/get_robot_state are ~1.3 s / ~0.9 s slower inside Claude Code than over raw MCP stdio (~0.1 s).
  They were not allowlisted, so each went through the auto-mode permission check; the allowlisted
  move tools are faster despite moving the arm.
- Shortest instructions: "reset arm to home position" 4 turns / 22.8 s; "pick up the motor ... return" 2 turns / 24.7 s.
