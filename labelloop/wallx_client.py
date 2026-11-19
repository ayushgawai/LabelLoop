"""Sync WALLX public_v2 client (msgpack + msgpack_numpy over websockets)."""
import base64
import json
import os
from pathlib import Path

import cv2
import msgpack
import msgpack_numpy
import numpy as np
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

HORIZON = 32


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
    def __init__(self, endpoint, headers=None):
        self.endpoint, self.headers = endpoint, headers or {}
        self.ws, self.metadata = None, None

    @classmethod
    def from_env(cls):
        return cls(*endpoint_headers_from_env())

    def connect(self):
        self.close()
        self._cm = connect(self.endpoint, additional_headers=self.headers, proxy=None, ping_interval=None,
                           compression=None, open_timeout=30, close_timeout=2, max_size=None)
        self.ws = self._cm.__enter__()  # v17 wants the context-manager protocol
        self.metadata = unpack(self.ws.recv(timeout=60))

    def close(self):
        if self.ws is not None:
            self._cm.__exit__(None, None, None)
            self.ws = None

    def _ask(self, payload, timeout):
        if self.ws is None:
            self.connect()
        self.ws.send(pack(payload))
        return unpack(self.ws.recv(timeout=timeout))

    def infer(self, payload, timeout=60):
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


def encode_jpeg_b64(rgb, quality=95):
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
