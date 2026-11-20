"""Sim robot link: X2Real obs (home-relative, sim gripper units) -> harness -> WALLX -> 33x14 sim chunk."""
import time

import cv2
import numpy as np
from websockets.exceptions import WebSocketException

from labelloop.pose import sim_to_wallx, wallx_to_sim
from labelloop.wallx_client import WallxError, build_payload, encode_jpeg_b64, hold_response

CAMERAS = {"camera_front": "front", "camera_left": "left_wrist", "camera_right": "right_wrist"}
WALLX_HW = (480, 640)  # what WALLX was trained on and what X2Real's stock client sends


def _for_wallx(rgb):
    """X2Real may send a larger head image (for QR reading); WALLX gets X2Real's own resize (cv2 INTER_AREA)."""
    if rgb.shape[:2] == WALLX_HW:
        return rgb
    return np.ascontiguousarray(cv2.resize(rgb, WALLX_HW[::-1], interpolation=cv2.INTER_AREA))


def _rgb(a, name):
    a = np.asarray(a)
    if a.ndim == 4:
        a = a[0]
    if a.ndim != 3 or a.shape[2] != 3 or a.dtype != np.uint8:
        raise ValueError(f"image {name}: expected HxWx3 uint8, got {a.shape} {a.dtype}")
    return a


def _arm(state, key):
    a = np.asarray(state[key], float).reshape(-1)
    if a.shape != (7,) or not np.isfinite(a).all():
        raise ValueError(f"state {key}: expected 7 finite values, got {a.shape}")
    return a


class X2RealLink:
    def __init__(self, harness, wallx, robot_cfg, embodiment=None):
        self.harness, self.wallx, self.cfg = harness, wallx, robot_cfg
        self.embodiment = embodiment or robot_cfg["embodiment"]
        self._fresh = True

    def reset(self, episode_info=None):
        self.harness.reset()
        self._fresh = True

    def step(self, obs):
        images = {dst: _rgb(obs["images"][src], src) for src, dst in CAMERAS.items()}
        f1, f2 = _arm(obs["state"], "follow1_pos"), _arm(obs["state"], "follow2_pos")
        state = sim_to_wallx(f1, f2, self.cfg)
        instruction = self.harness.step(images, state)
        if instruction is None:
            reply = hold_response(self.embodiment, state)
            chunk = wallx_to_sim(reply["actions"], self.cfg)
        else:
            reset, self._fresh = self._fresh or self.harness.subgoal_changed, False
            t0 = time.monotonic()
            try:
                reply = self.wallx.infer(build_payload(self.embodiment, instruction,
                                                       {k: encode_jpeg_b64(_for_wallx(v)) for k, v in images.items()}, state, reset))
                chunk = wallx_to_sim(reply["actions"], self.cfg)
                self.harness.log_reply(time.monotonic() - t0)
                if getattr(self.harness, "runlog", None):  # sent state vs WALLX first/last frame: a frame mismatch shows as a jump
                    a = reply["actions"]
                    row = lambda arm, i: [round(v, 4) for v in [*a[arm]["position"][i], *a[arm]["rotation_rpy"][i], a[arm]["gripper"][i]]]
                    self.harness.runlog.write(wallx={arm: {"sent": [round(v, 4) for v in [*state[arm]["position"], *state[arm]["rotation_rpy"], state[arm]["gripper"]]],
                                                           "first": row(arm, 0), "last": row(arm, -1)} for arm in ("left_ee", "right_ee")})
            except (WallxError, TimeoutError, ConnectionError, WebSocketException, KeyError, ValueError) as e:
                self.harness.log_reply(time.monotonic() - t0, error=f"{type(e).__name__}: {e}")
                self._fresh = self._fresh or reset  # failed call: carry the pending reset to the next one
                chunk = wallx_to_sim(hold_response(self.embodiment, state)["actions"], self.cfg)
        # row 0 = current state echo: X2Real skips it (first_frame: prediction_only)
        return np.vstack([np.concatenate([f1, f2])[None], chunk]).astype(np.float32)
