"""Local stub WALLX server (sync websockets, background thread)."""
import threading

import numpy as np
from websockets.sync.server import serve

from labelloop.wallx_client import decode_jpeg_b64, hold_response, pack, unpack


class Stub:
    def __init__(self):
        self.requests, self.close_once, self.connections, self.close_on_bad = [], False, 0, False
        self.server = serve(self._handle, "127.0.0.1", 0, compression=None, max_size=None)
        self.url = "ws://127.0.0.1:%d" % self.server.socket.getsockname()[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def stop(self):
        self.server.shutdown()

    @staticmethod
    def check(req):
        if req.get("embodiment") != "quanta_x1_pro/gripper-g":
            return "bad_embodiment"
        imgs = req["observation"]["images"]
        for k in ("front", "left_wrist", "right_wrist"):
            if k not in imgs:
                return "missing_camera"
            try:
                decode_jpeg_b64(imgs[k])
            except Exception:
                return "bad_image"

    def _handle(self, ws):
        self.connections += 1
        ws.send(pack({"model": "wall-x", "model_version": "stub"}))
        for frame in ws:
            req = unpack(frame)
            self.requests.append(req)
            if self.close_once:
                self.close_once = False
                ws.close(1011)
                return
            bad = self.check(req)
            if bad and self.close_on_bad:  # real WALLX: bad request -> 1011 wallx_session_closed
                ws.close(1011, "wallx_session_closed")
                return
            reply = ({"error": {"code": bad, "message": bad}} if bad
                     else hold_response(req["embodiment"], req["observation"]["state"]))
            ws.send(pack(reply))


def rgb(seed=0):
    return np.random.default_rng(seed).integers(0, 255, (48, 64, 3), dtype=np.uint8)
