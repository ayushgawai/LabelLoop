"""Full fake 5-parcel stream through both real robot links: real Harness, real WallxClient, stub WALLX."""
import itertools
from pathlib import Path

import numpy as np
from websockets.sync.client import connect

from labelloop.config import load_yaml
from labelloop.harness import Harness
from labelloop.robots import event_server
from labelloop.robots.mock import A, B, UNKNOWN, FakeScene, garbage, nolabel, parcel
from labelloop.robots.x2real_server import X2RealLink
from labelloop.wallx_client import WallxClient, build_payload, encode_jpeg_b64, pack, unpack
from test_event_server import start_relay
from wallx_stub import Stub

ROOT = Path(__file__).parent.parent
TASK = load_yaml(ROOT / "configs/task_qr_sort.yaml")
ROBOT = load_yaml(ROOT / "configs/ex001_6r.yaml")


def fake_clock():
    c = itertools.count()
    return lambda: float(next(c))


def stream():
    objs = [parcel(A), parcel(B, fail_places=2), parcel(UNKNOWN), nolabel(), garbage()]
    return objs, FakeScene(objs)


def check_episode(objs, scene, harness):
    assert [scene.at[id(o)] for o in objs] == ["blue", "red", "reject", "reject", "reject"]
    assert harness.sm.parcels == 5 and not harness.done  # stream mode never latches DONE


def test_x2real_link_full_episode():
    stub = Stub()
    objs, scene = stream()
    harness = Harness(TASK, clock=fake_clock())
    wallx = WallxClient(stub.url, {})
    link = X2RealLink(harness, wallx, ROBOT)
    k = ROBOT["gripper"]["sim_max"] / ROBOT["gripper"]["wallx_max"]
    try:
        for _ in range(500):
            imgs, st = scene.observe()
            grip, grip2 = st["left_ee"]["gripper"] * k, st["right_ee"]["gripper"] * k
            obs = {"images": {"camera_front": imgs["front"], "camera_left": imgs["left_wrist"],
                              "camera_right": imgs["right_wrist"]},
                   "state": {"follow1_pos": np.r_[np.zeros(6), grip], "follow2_pos": np.r_[np.zeros(6), grip2]}}
            chunk = link.step(obs)
            assert chunk.shape == (33, 14)
            # stub echoes state -> arm holds at home: relative pose ~0, gripper unchanged
            assert np.allclose(chunk[1:, :6], 0, atol=1e-5) and np.allclose(chunk[1:, 6], grip, atol=1e-4)
            scene.apply(harness.sm.instruction())
            if harness.sm.parcels == 5:
                break
        check_episode(objs, scene, harness)
    finally:
        wallx.close()
        stub.stop()


def test_event_server_full_episode():
    stub = Stub()
    objs, scene = stream()
    harness = Harness(TASK, clock=fake_clock())
    box = start_relay(event_server.Relay(stub.url, {}, lambda: harness))
    ws = connect("ws://127.0.0.1:%d" % box["port"], compression=None, max_size=None).__enter__()
    try:
        assert unpack(ws.recv(timeout=5))["model"] == "wall-x"
        for _ in range(500):
            imgs, state = scene.observe()
            req = build_payload("quanta_x1_pro/gripper-g", "sort the parcels by QR label",
                                {k: encode_jpeg_b64(v) for k, v in imgs.items()}, state)
            ws.send(pack(req))
            reply = unpack(ws.recv(timeout=10))
            assert len(reply["actions"]["left_ee"]["gripper"]) == 32
            scene.apply(harness.sm.instruction())
            if harness.sm.parcels == 5:
                break
        check_episode(objs, scene, harness)
        sent = {r["instruction"] for r in stub.requests}
        assert "sort the parcels by QR label" not in sent and any("bin" in s for s in sent)
    finally:
        ws.__exit__(None, None, None)
        box["loop"].call_soon_threadsafe(box["stop"].set)
        stub.stop()
