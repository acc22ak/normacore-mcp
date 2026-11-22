"""Where do the ~47 ms per Station read go? time-to-first-entry vs time-to-end-of-stream,
with the client as shipped and with the request sent as a single write."""
import asyncio, os, statistics, struct, sys, time, logging
from pathlib import Path
os.environ.setdefault("NORMA_CORE_REPO", str(Path.home()/"norma-core"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
logging.disable(logging.INFO)
import server as S
import learn as L

async def one(q):
    c = await S.client()
    t = time.perf_counter()
    qr = c.read_from_tail(q, offset=L.encode_shift(1), limit=1, step=1, buf_size=2)
    first = None
    while True:
        e = await qr.data.get()
        if e is None: break
        first = first or time.perf_counter()
    end = time.perf_counter()
    return (first - t) * 1000, (end - t) * 1000

async def run(label, q, n=30):
    for _ in range(3): await one(q)
    r = [await one(q) for _ in range(n)]
    print(f"{label:<34} first-entry {statistics.median(x for x,_ in r):6.1f} ms   end-of-stream {statistics.median(y for _,y in r):6.1f} ms")

async def main():
    q = "st3215/inference"
    await run("as shipped", q)
    c = await S.client()
    import socket
    sock = c.writer.get_extra_info("socket")
    print("TCP_NODELAY on client socket:", sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY))
    orig = c.send_message
    async def single_write(msg):
        p = msg.encode(); c.writer.write(struct.pack('<Q', len(p)) + p); await c.writer.drain()
        c.last_msg_sent_time = time.time()
    c.send_message = single_write
    await run("single-write request", q)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    await run("single-write + NODELAY", q)
asyncio.run(main())
