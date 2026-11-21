"""LocalWallxClient against a stub of the Wall-X native server (no network beyond loopback)."""
import threading

import numpy as np
import pytest
from websockets.sync.server import serve

from labelloop.wallx_client import (LocalWallxClient, WallxClient, WallxError, build_payload, encode_jpeg_b64,
                                    from_config, pack, unpack)
from wallx_stub import rgb

STATE = {"left_ee": {"position": [.1, .2, .3], "rotation_rpy": [0, 0, .5], "gripper": .763},
         "right_ee": {"position": [.1, -.2, .3], "rotation_rpy": [0, 0, 0], "gripper": 1.526}}


class NativeStub:
    """Speaks Wall-X's server: metadata frame, then {views,state,instruction} -> {follow1_pos,follow2_pos} (33x7, row 0 = state)."""

    def __init__(self):
        self.requests = []
        self.server = serve(self._handle, "127.0.0.1", 0, compression=None, max_size=None)
        self.url = "ws://127.0.0.1:%d" % self.server.socket.getsockname()[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def _handle(self, ws):
        ws.send(pack({"batch_enabled": True, "model": "wall-x"}))
        for frame in ws:
            req = unpack(frame)
            self.requests.append(req)
            if not req["instruction"]:
                ws.send("Traceback: boom")
                ws.close(1011, "Internal server error")
                return
            ws.send(pack({k: np.vstack([req["state"][k]] + [np.add(req["state"][k], .01 * i) for i in range(1, 33)])
                          for k in ("follow1_pos", "follow2_pos")} | {"server_timing": {"infer_ms": 5.0}}))


def payload(instr="pick"):
    return build_payload("quanta_x1_pro/gripper-g", instr,
                         {k: encode_jpeg_b64(rgb()) for k in ("front", "left_wrist", "right_wrist")}, STATE)


def test_local_translates_both_ways_and_logs_latency(caplog):
    stub = NativeStub()
    c = LocalWallxClient(stub.url)
    caplog.set_level("INFO", logger="labelloop.wallx")
    r = c.infer(payload())
    req = stub.requests[0]
    assert set(req["views"]) == {"camera_front", "camera_left", "camera_right"} and req["instruction"] == "pick"
    cfg = c.robot_cfg
    f1 = req["state"]["follow1_pos"]
    assert np.allclose(f1[:3], np.array(STATE["left_ee"]["position"]) - cfg["home_pose"]["left_ee"]["position"])  # home-relative
    assert np.isclose(f1[6], .763 * cfg["gripper"]["sim_max"] / cfg["gripper"]["wallx_max"])  # sim gripper units
    a = r["actions"]
    assert r["embodiment"] == "quanta_x1_pro/gripper-g" and len(a["left_ee"]["position"]) == 32  # echo row dropped
    assert a["left_ee"]["gripper"][0] > 0 and len(a["right_ee"]["gripper"]) == 32
    assert c.last_latency_s > 0 and "backend=local" in caplog.text and "latency_ms=" in caplog.text
    c.close()


def test_local_error_string_raises_wallxerror():
    stub = NativeStub()
    c = LocalWallxClient(stub.url)
    with pytest.raises(WallxError):
        c.infer(payload(""))
    c.close()


def test_config_switch(tmp_path, monkeypatch):
    cfg = tmp_path / "w.yaml"
    cfg.write_text("backend: local\ncloud: {timeout_s: 7}\nlocal: {url: 'ws://127.0.0.1:1', timeout_s: 3}\n")
    monkeypatch.setenv("WALLX_ENDPOINT", "ws://cloud.invalid")
    monkeypatch.delenv("WALLX_BACKEND", raising=False)
    monkeypatch.delenv("WALLX_API_KEY", raising=False)
    monkeypatch.delenv("WALLX_HEADERS_FILE", raising=False)
    c = from_config(cfg)
    assert type(c) is LocalWallxClient and c.endpoint == "ws://127.0.0.1:1" and c.timeout == 3
    monkeypatch.setenv("WALLX_BACKEND", "cloud")
    c = WallxClient.from_env(cfg)
    assert type(c) is WallxClient and c.endpoint == "ws://cloud.invalid" and c.timeout == 7
    assert type(from_config("configs/wallx.yaml", "local")) is LocalWallxClient


def test_roundtrip_state_through_native_echo():
    """payload state -> native sim-convention state -> (echo reply) -> WALLX actions == original state, all 32 rows."""
    class Echo(NativeStub):
        def _handle(self, ws):
            ws.send(pack({"model": "wall-x"}))
            for frame in ws:
                st = unpack(frame)["state"]
                ws.send(pack({k: np.vstack([st[k]] * 33) for k in ("follow1_pos", "follow2_pos")}))
    stub = Echo()
    c = LocalWallxClient(stub.url)
    a = c.infer(payload())["actions"]
    for arm in STATE:
        assert len(a[arm]["position"]) == 32
        assert np.allclose(a[arm]["position"], [STATE[arm]["position"]] * 32, atol=1e-6)
        assert np.allclose(a[arm]["rotation_rpy"], [STATE[arm]["rotation_rpy"]] * 32, atol=1e-6)
        assert np.allclose(a[arm]["gripper"], STATE[arm]["gripper"], atol=1e-6)
    c.close()
