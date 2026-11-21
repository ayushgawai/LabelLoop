"""Sync WALLX public_v2 client (msgpack + msgpack_numpy over websockets)."""
import base64
import json
import logging
import os
import time
from pathlib import Path

import cv2
import msgpack
import msgpack_numpy
import numpy as np
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

from labelloop.pose import interp_rpy, sim_to_wallx, wallx_to_sim

HORIZON = 32
CONFIG = Path(__file__).resolve().parents[1] / "configs" / "wallx.yaml"
log = logging.getLogger("labelloop.wallx")


class WallxError(Exception):
    def __init__(self, code, message, reply=None):
        super().__init__(f"{code}: {message}")
        self.code, self.message, self.reply = code, message, reply


def pack(value):
    return msgpack.packb(value, default=msgpack_numpy.encode, use_bin_type=True)


def unpack(frame):
    if not isinstance(frame, bytes):
        raise ValueError("WALLX protocol requires a binary MessagePack frame")
    return msgpack.unpackb(frame, raw=False, object_hook=msgpack_numpy.decode)


def endpoint_headers_from_env():
    headers = {}
    if os.environ.get("WALLX_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["WALLX_API_KEY"]
    if os.environ.get("WALLX_HEADERS_FILE"):
        headers.update(json.loads(Path(os.environ["WALLX_HEADERS_FILE"]).read_text()))
    return os.environ["WALLX_ENDPOINT"], headers


class WallxClient:
    """Cloud public_v2 client. Subclass LocalWallxClient talks to a self-hosted Wall-X server with the same infer() contract."""
    backend = "cloud"

    def __init__(self, endpoint, headers=None, timeout=60, open_timeout=30):
        self.endpoint, self.headers, self.timeout, self.open_timeout = endpoint, headers or {}, timeout, open_timeout
        self.ws, self.metadata, self.last_latency_s = None, None, None

    @classmethod
    def from_env(cls, config=CONFIG):
        """Backend (cloud | local) chosen by configs/wallx.yaml `backend`, overridable with env WALLX_BACKEND."""
        return from_config(config)

    def connect(self):
        self.close()
        self._cm = connect(self.endpoint, additional_headers=self.headers, proxy=None, ping_interval=None,
                           compression=None, open_timeout=self.open_timeout, close_timeout=2, max_size=None)
        self.ws = self._cm.__enter__()  # v17 wants the context-manager protocol
        self.metadata = unpack(self.ws.recv(timeout=self.timeout))

    def close(self):
        if self.ws is not None:
            self._cm.__exit__(None, None, None)
            self.ws = None

    def _ask(self, payload, timeout):
        if self.ws is None:
            self.connect()
        self.ws.send(pack(payload))
        return self._decode(self.ws.recv(timeout=timeout))

    _decode = staticmethod(unpack)

    def infer(self, payload, timeout=None):
        t0 = time.monotonic()
        try:
            return self._infer(payload, timeout or self.timeout)
        finally:
            self.last_latency_s = time.monotonic() - t0
            log.info("wallx_infer backend=%s latency_ms=%.0f", self.backend, self.last_latency_s * 1000)

    def _infer(self, payload, timeout):
        try:
            try:
                reply = self._ask(payload, timeout)
            except ConnectionClosed as e:
                reason = e.rcvd.reason if e.rcvd else ""
                self.close()
                if reason == "wallx_session_closed":  # server rejected this request: resending repeats it
                    self.connect()
                    raise WallxError("session_closed", reason) from e
                reply = self._ask(payload, timeout)  # e.g. network drop: reconnect once, re-read handshake
        except TimeoutError:  # a late reply would desync the stream
            self.close()
            raise
        err = reply.get("error") if isinstance(reply, dict) else reply if isinstance(reply, str) else None
        if err is not None:
            code, msg = (err.get("code"), err.get("message")) if isinstance(err, dict) else ("error", err)
            raise WallxError(code, msg, reply)
        return reply


# Wall-X native server (wall_x/_vendor/harrix/serving): same msgpack-over-WS transport, different schema.
LOCAL_CAMERAS = {"front": "camera_front", "left_wrist": "camera_left", "right_wrist": "camera_right"}
LOCAL_ARMS = {"left_ee": "follow1_pos", "right_ee": "follow2_pos"}


class LocalWallxClient(WallxClient):
    """Self-hosted Wall-X (`run_serving` with --serialize-actions, robot_type desktop, no auth).

    The native server speaks the X2Real SIM wire convention (X2Real's Policy Space adapter forwards the sim state unchanged):
    per arm [rel xyz, rel rpy, sim gripper], relative to the home pose in configs/ex001_6r.yaml; follow1 = left, follow2 = right.
    We convert our WALLX-absolute payload with pose.wallx_to_sim, and the (33, 7) reply (row 0 = state echo, dropped) back with
    pose.sim_to_wallx. Ignored: embodiment (fixed by the server's robot_type) and reset_history (the server keeps no history)."""
    backend = "local"

    def __init__(self, endpoint="ws://127.0.0.1:8002", robot_cfg=None, **kw):
        super().__init__(endpoint, None, **kw)
        self._embodiment = None
        if robot_cfg is None:
            import yaml
            robot_cfg = yaml.safe_load(open(CONFIG.parent / "ex001_6r.yaml"))
        self.robot_cfg = robot_cfg

    _decode = staticmethod(lambda frame: frame if isinstance(frame, str) else unpack(frame))  # server sends its traceback as a text frame

    def _ask(self, payload, timeout):
        obs, self._embodiment = payload["observation"], payload.get("embodiment")
        one = {arm: {"position": [obs["state"][arm]["position"]], "rotation_rpy": [obs["state"][arm]["rotation_rpy"]],
                     "gripper": [obs["state"][arm]["gripper"]]} for arm in LOCAL_ARMS}
        row = wallx_to_sim(one, self.robot_cfg)[0].astype(float)  # (14,) = follow1 7 + follow2 7
        req = {"views": {LOCAL_CAMERAS[k]: v for k, v in obs["images"].items()}, "instruction": payload["instruction"],
               "state": {"follow1_pos": row[:7].tolist(), "follow2_pos": row[7:].tolist()}}
        return self._to_wallx(super()._ask(req, timeout))

    def _to_wallx(self, reply):
        if not (isinstance(reply, dict) and "follow1_pos" in reply):
            return reply  # error string / dict passes to infer()'s error check
        f1, f2 = np.asarray(reply["follow1_pos"], float), np.asarray(reply["follow2_pos"], float)
        if len(f1) == HORIZON + 1:  # row 0 echoes the state we sent
            f1, f2 = f1[1:], f2[1:]
        rows = [sim_to_wallx(a, b, self.robot_cfg) for a, b in zip(f1, f2)]
        acts = {arm: {"position": [r[arm]["position"] for r in rows], "rotation_rpy": [r[arm]["rotation_rpy"] for r in rows],
                      "gripper": [r[arm]["gripper"] for r in rows]} for arm in LOCAL_ARMS}
        return {"embodiment": self._embodiment, "actions": acts, "server_timing": reply.get("server_timing")}


def from_config(path=CONFIG, backend=None):
    """WallxClient / LocalWallxClient from configs/wallx.yaml. Cloud endpoint + key come from env (WALLX_ENDPOINT, WALLX_API_KEY)."""
    import yaml
    cfg = yaml.safe_load(open(path))
    backend = backend or os.environ.get("WALLX_BACKEND") or cfg["backend"]
    c = cfg[backend]
    kw = {"timeout": c.get("timeout_s", 60), "open_timeout": c.get("open_timeout_s", 30)}
    if backend == "local":
        return LocalWallxClient(c.get("url", "ws://127.0.0.1:8002"), **kw)
    if backend != "cloud":
        raise ValueError(f"unknown wallx backend {backend!r}")
    return WallxClient(*endpoint_headers_from_env(), **kw)


def build_payload(embodiment, instruction, images_b64, state, reset_history=False):
    p = {"embodiment": embodiment, "normalization": "provided", "instruction": instruction,
            "observation": {"images": images_b64, "state": state}}
    if reset_history:  # WALLX keeps per-session motion history; clear it on a new sub-goal / episode
        p["reset_history"] = True
    return p


def hold_response(embodiment, state):
    """WALLX-shaped reply that repeats the current state for 32 frames (arms stay still)."""
    acts = {arm: {"position": [list(state[arm]["position"])] * HORIZON,
                  "rotation_rpy": [list(state[arm]["rotation_rpy"])] * HORIZON,
                  "gripper": [float(state[arm]["gripper"])] * HORIZON} for arm in ("left_ee", "right_ee")}
    return {"embodiment": embodiment, "actions": acts}


def move_response(embodiment, state, targets, horizon=HORIZON):
    """WALLX-shaped reply easing (smoothstep) each arm in `targets` ({arm: {position, rotation_rpy[, gripper]}}) from `state`
    over `horizon` frames; other arms hold, a missing gripper keeps its value."""
    acts = {}
    for arm in ("left_ee", "right_ee"):
        s = state[arm]
        h = {**s, **targets.get(arm, {})}
        p0, p1 = np.asarray(s["position"], float), np.asarray(h["position"], float)
        pos, rpy, grip = [], [], []
        for i in range(1, horizon + 1):
            u = i / horizon
            e = u * u * (3 - 2 * u)
            pos.append((p0 + e * (p1 - p0)).tolist())
            rpy.append(interp_rpy(s["rotation_rpy"], h["rotation_rpy"], e).tolist())
            grip.append(float(s["gripper"] + e * (h["gripper"] - s["gripper"])))
        acts[arm] = {"position": pos, "rotation_rpy": rpy, "gripper": grip}
    return {"embodiment": embodiment, "actions": acts}


def home_response(embodiment, state, home, horizon=HORIZON):
    """WALLX-shaped reply easing (smoothstep) from `state` to `home` over `horizon` frames."""
    return move_response(embodiment, state, home, horizon)


def encode_jpeg_b64(rgb, quality=80):  # 80 vs 95: ~3x smaller upload to the WALLX cloud (Shenzhen, ~340 ms RTT); the event client sends its own JPEGs
    ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(rgb[..., ::-1]), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def decode_jpeg_b64(s):
    if s.startswith("data:image/jpeg;base64,"):
        s = s[len("data:image/jpeg;base64,"):]
    bgr = cv2.imdecode(np.frombuffer(base64.b64decode(s), np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Not a decodable JPEG")
    return np.ascontiguousarray(bgr[..., ::-1])
