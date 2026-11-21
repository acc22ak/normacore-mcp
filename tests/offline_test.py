import asyncio, io, os, sys
# usage: python tests/offline_test.py <normacore-mcp dir> <norma-core repo>
sys.path.insert(0, sys.argv[1])
os.environ["NORMA_CORE_REPO"] = sys.argv[2]
import server as S
from PIL import Image as PI
from target.gen_python.protobuf.drivers.st3215 import st3215
from target.gen_python.protobuf.drivers.usbvideo import usbvideo, frame
from target.gen_python.protobuf.station import inference, drivers, commands

def motor_state(pos, torque=0):
    b = bytearray(0x50)
    b[0x38:0x3A] = pos.to_bytes(2, "little"); b[0x28] = torque
    return bytes(b)

def inf_state():
    motors = [st3215.InferenceState_MotorState(id=i, state=motor_state(2048 + 10*i), range_min=1000, range_max=3000) for i in range(1, 9)]
    bus = st3215.InferenceState_BusState(bus=st3215.ST3215Bus(serial_number="5AB0FOLLOW"), motors=motors)
    return st3215.InferenceState(buses=[bus]).encode()

buf = io.BytesIO(); PI.new("RGB", (64, 48), (200, 30, 30)).save(buf, "JPEG"); JPEG = buf.getvalue()
DATA = {
  "st3215/inference": [inf_state()],
  "inference-states": [inference.InferenceRx(entries=[
      inference.InferenceRx_Entry(queue="usbvideo/abc", type=drivers.QueueDataType.QDT_USB_VIDEO_FRAMES),
      inference.InferenceRx_Entry(queue="st3215/inference", type=drivers.QueueDataType.QDT_ST3215_INFERENCE)]).encode()],
  "usbvideo/abc": [usbvideo.RxEnvelope(type=usbvideo.RxEnvelopeType.ET_FRAMES,
      camera=usbvideo.Camera(product="C270 HD WEBCAM"),
      frames=frame.FramesPack(frames_data=[JPEG])).encode()],
}
async def fake_read_last(q, n=1): return [memoryview(bytes(x)) for x in DATA[q]]
sent = []
async def fake_send(c, cmds): sent.extend(cmds)
async def fake_client(): return None
S.read_last, S.send_commands, S.client = fake_read_last, fake_send, fake_client

async def main():
    print("cams:", await S.list_cameras())
    out = await S.look()
    print("look:", [o if isinstance(o, str) else f"<Image {len(o.data)} bytes>" for o in out])
    st = await S.get_robot_state()
    print("state:", st["follower_bus"], st["buses"]["5AB0FOLLOW"]["motors"][1])
    print("move:", await S.move_joints({"1": 0.9, "elbow": 0.5}))
    print("nudge:", await S.nudge_joints({"gripper": -0.05}))
    print("relax:", await S.relax())
    # decode one sent command back
    c = commands.DriverCommandReader(memoryview(sent[3].encode()))
    cmd = st3215.CommandReader(c.get_body())
    sw = cmd.get_sync_write()
    print("goal write: bus", cmd.get_target_bus_serial(), "addr", hex(sw.get_address()),
          [(m.get_motor_id(), int.from_bytes(bytes(m.get_value()), "little")) for m in sw.get_motors()])
    tools = await S.mcp.list_tools()
    print("tools:", [t.name for t in tools])
asyncio.run(main())
