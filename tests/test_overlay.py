import asyncio
import threading
import base64

import cv2
import numpy as np
import pytest
from websockets.sync.client import connect

from labelloop.perception import overlay
from labelloop.robots import event_server
from labelloop.wallx_client import build_payload, encode_jpeg_b64, pack, unpack
from wallx_stub import Stub, rgb
from test_event_server import STATE

STATE = {"left_ee": {"position": [.1, .2, .3], "rotation_rpy": [0, 0, 0], "gripper": .1},
         "right_ee": {"position": [.1, -.2, .3], "rotation_rpy": [0, 0, 0], "gripper": .2}}


def _rect():
    img = np.zeros((720, 1280, 3), np.uint8)
    img[:] = (10, 20, 30)
    return img


def test_overlay_none_is_input():
    img = _rect()
    out = overlay.draw(img, "none")
    assert out is img
    assert np.array_equal(out, img)


def test_overlay_parcel_mask_grid():
    img = _rect()
    parcel = (200, 180, 320, 260)
    target = (800, 300, 950, 430)
    bins = {
        "blue": (1000, 100, 1180, 260),
        "red": (50, 500, 200, 650),
        "black": target,
    }
    roi = (100, 100, 1180, 650)

    parcel_out = overlay.draw(img, "parcel", parcel=parcel, target=target, bins=bins, roi=roi)
    assert np.array_equal(img, _rect())
    assert tuple(parcel_out[168, 250]) == overlay.MAGENTA  # edge of the 12 px padded box
    assert tuple(parcel_out[300, 800]) == overlay.GREEN
    assert tuple(parcel_out[292, 568]) == overlay.YELLOW  # arrow midpoint (parcel centre (260, 220) -> target centre (875, 365))
    assert np.array_equal(parcel_out, overlay.draw(img, "parcel", parcel=parcel, target=target, bins=bins, roi=roi))

    mask_out = overlay.draw(img, "mask", parcel=parcel, target=target, bins=bins, roi=roi)
    assert tuple(mask_out[168, 250]) == overlay.MAGENTA
    assert tuple(mask_out[300, 800]) == overlay.GREEN
    assert tuple(mask_out[150, 1050]) == overlay.TABLE_GREEN
    assert np.array_equal(img, _rect())

    grid_out = overlay.draw(img, "grid", parcel=parcel, target=target, bins=bins, roi=roi, cells=(8, 5))
    assert tuple(grid_out[400, 100 + (1180 - 100) // 8]) == overlay.WHITE  # first vertical grid line
    assert np.array_equal(img, _rect())


class FakeHarness:
    def __init__(self, instruction):
        self.instruction, self.steps, self.replies = instruction, [], []
        self.cfg, self.state_name, self.home = {}, "IDLE", None

    def scripted_reply(self, embodiment, state):
        if self.cfg.get("scripted_home") and self.state_name == "HOME":
            return event_server.hold_response(embodiment, state)

    def reset(self):
        pass

    def step(self, images, state):
        self.steps.append((images, state))
        return self.instruction

    def log_reply(self, latency_s, error):
        self.replies.append(error)

    def prompt_boxes(self, shape):
        h, w = shape[:2]
        parcel = (10, 10, 20, 20)
        target = (30, 30, 40, 40)
        bins = {"black": target, "red": (50, 50, 60, 60)}
        roi = (0, 0, w - 1, h - 1)
        return parcel, target, bins, roi


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

    def make(harness, overlay_mode="none"):
        box = start_relay(event_server.Relay(stub.url, {}, (lambda: harness) if harness else None, overlay_mode=overlay_mode))
        boxes.append(box)
        return connect("ws://127.0.0.1:%d" % box["port"], compression=None, max_size=None).__enter__()

    yield stub, make
    for b in boxes:
        b["loop"].call_soon_threadsafe(b["stop"].set)
    stub.stop()


def ask(ws, r):
    ws.send(pack(r))
    return unpack(ws.recv(timeout=10))


def test_overlay_changes_forwarded_front_image(env):
    stub, make = env
    h = FakeHarness("move the right gripper above the box")
    ws = make(h, overlay_mode="parcel")
    ws.recv(timeout=10)
    ask(ws, req())
    assert len(stub.requests) == 1
    raw = base64.b64decode(req()["observation"]["images"]["front"])
    forwarded = base64.b64decode(stub.requests[0]["observation"]["images"]["front"])
    assert forwarded != raw
    assert stub.requests[0]["instruction"] == "move the right gripper above the box"


def test_overlay_none_keeps_forwarded_front_image_identical(env):
    stub, make = env
    h = FakeHarness("move the right gripper above the box")
    ws = make(h, overlay_mode="none")
    ws.recv(timeout=10)
    ask(ws, req())
    raw = base64.b64decode(req()["observation"]["images"]["front"])
    forwarded = base64.b64decode(stub.requests[0]["observation"]["images"]["front"])
    assert forwarded == raw
