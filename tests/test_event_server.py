import asyncio
import threading

import pytest
from websockets.sync.client import connect

from labelloop.robots import event_server
from labelloop.wallx_client import build_payload, encode_jpeg_b64, pack, unpack
from wallx_stub import Stub, rgb

STATE = {"left_ee": {"position": [.1, .2, .3], "rotation_rpy": [0, 0, 0], "gripper": .1},
         "right_ee": {"position": [.1, -.2, .3], "rotation_rpy": [0, 0, 0], "gripper": .2}}


class FakeHarness:
    def __init__(self, instruction):
        self.instruction, self.steps, self.replies = instruction, [], []
        self.cfg, self.state_name, self.home = {}, "IDLE", None

    def reset(self):
        pass

    def step(self, images, state):
        self.steps.append((images, state))
        return self.instruction

    def log_reply(self, latency_s, error=None):
        self.replies.append(error)


def req(instruction="sort the boxes", emb="quanta_x1_pro/gripper-g"):
    r = build_payload(emb, instruction, {k: encode_jpeg_b64(rgb()) for k in ("front", "left_wrist", "right_wrist")}, STATE)
    r["control_context"] = {"episode": 7}
    return r


def start_relay(relay):
    box, ready = {}, threading.Event()

    async def run():
        async with event_server.start(relay, "127.0.0.1", 0) as s:
            box["port"], box["stop"], box["loop"] = s.sockets[0].getsockname()[1], asyncio.Event(), asyncio.get_running_loop()
            ready.set()
            await box["stop"].wait()

    threading.Thread(target=lambda: asyncio.run(run()), daemon=True).start()
    ready.wait(5)
    return box


@pytest.fixture
def env():
    stub, boxes = Stub(), []

    def make(harness):
        box = start_relay(event_server.Relay(stub.url, {}, (lambda: harness) if harness else None))
        boxes.append(box)
        return connect("ws://127.0.0.1:%d" % box["port"], compression=None, max_size=None).__enter__()

    yield stub, make
    for b in boxes:
        b["loop"].call_soon_threadsafe(b["stop"].set)
    stub.stop()


def ask(ws, r):
    ws.send(pack(r))
    return unpack(ws.recv(timeout=10))


def test_replace_instruction_forward_all(env):
    stub, make = env
    h = FakeHarness("move the right gripper above the box")
    ws = make(h)
    assert unpack(ws.recv(timeout=10))["model"] == "wall-x"  # metadata forwarded
    reply = ask(ws, req())
    assert stub.requests[0]["instruction"] == "move the right gripper above the box"
    assert stub.requests[0]["control_context"] == {"episode": 7}
    assert reply["actions"]["right_ee"]["gripper"][31] == pytest.approx(.2)
    assert h.steps[0][0]["front"].shape == (48, 64, 3) and h.steps[0][1] == STATE and h.replies == [None]


def test_error_reply_unchanged(env):
    stub, make = env
    ws = make(FakeHarness("x"))
    ws.recv(timeout=10)
    assert ask(ws, req(emb="quanta_x1_pro/gripper-h"))["error"]["code"] == "bad_embodiment"


def test_hold_path(env):
    stub, make = env
    ws = make(FakeHarness(None))
    ws.recv(timeout=10)
    reply = ask(ws, req())
    assert stub.requests == [] and len(reply["actions"]["left_ee"]["position"]) == 32


def test_passthrough(env):
    stub, make = env
    ws = make(None)
    ws.recv(timeout=10)
    ask(ws, req("goal text"))
    assert stub.requests[0]["instruction"] == "goal text"


def test_invalid_request_closes_1011(env):
    stub, make = env
    ws = make(None)
    ws.recv(timeout=10)
    r = req()
    r["normalization"] = "bogus"
    ws.send(pack(r))
    with pytest.raises(Exception) as e:
        ws.recv(timeout=10)
    assert e.value.rcvd.code == 1011


def test_reset_history_harness_vs_passthrough(env):
    stub, make = env
    h = FakeHarness("a")
    h.subgoal_changed = False
    ws = make(h)
    ws.recv(timeout=10)
    ask(ws, req()); ask(ws, req())
    h.subgoal_changed = True
    ask(ws, req())
    assert ["reset_history" in r for r in stub.requests] == [True, False, True]
    n = len(stub.requests)
    ws2 = make(None)
    ws2.recv(timeout=10)
    ask(ws2, req())
    assert "reset_history" not in stub.requests[n]


def test_scripted_home_no_wallx_call(env):
    stub, make = env
    h = FakeHarness("go home")
    h.cfg, h.state_name = {"scripted_home": True}, "HOME"
    h.home = {a: {"position": [0, 0, 0], "rotation_rpy": [0, 0, 0], "gripper": 0.0} for a in ("left_ee", "right_ee")}
    ws = make(h)
    ws.recv(timeout=10)
    reply = ask(ws, req())
    assert stub.requests == [] and reply["actions"]["left_ee"]["position"][-1] == [0, 0, 0]
