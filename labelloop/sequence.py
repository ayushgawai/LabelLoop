"""Atomic sub-goal sequence for the relay (--sequence FILE): one short WALLX instruction per step, a fixed number of calls each,
the idle arm locked, label reads between steps. Harness-compatible interface (step / scripted_reply / reset / log_reply / lock_arms).

FILE: {"code_hint": "420/90024/001", "lanes": {"blue": [...ZIP3], "red": [...]}, "steps": [
   {"read": true},                                                 # decode the label (head QR, then printed ZIP); else the demo code
   {"say": "pick up the IQBAR box", "calls": 15, "arm": "right"},  # WALLX instruction; the other arm is locked
   {"say": "put the box you are holding into the {mat} bin", "calls": 20, "arm": "right"},
   {"say": "put both arms back to the home position", "calls": 3}]}
{mat} = the bin chosen by the latest read; "arm": "lane" = the lane's arm (blue -> left, red / black -> right).
"""
import re

import numpy as np

from labelloop.perception import label_ocr, scene
from labelloop.perception.qr import locate
from labelloop.wallx_client import move_response

ZIP3 = {"blue": ("940", "941", "950", "951"), "red": ("900", "901", "902")}


def bin_for(code):
    m = re.match(r"^420/(\d{5})/\d{3}$", code or "")
    z = m.group(1)[:3] if m else None
    return next((b for b, zs in ZIP3.items() if z in zs), "black")


class Sequence:
    def __init__(self, spec, runlog=None):
        self.spec, self.runlog = spec, runlog
        self.reset()

    def reset(self):
        self.i, self.calls, self.code, self.mat, self.source = 0, 0, None, "black", None
        self.subgoal_changed, self.done, self.state_name, self.last_log = True, False, "SEQ0", None
        self.spot, self.spot0 = None, None  # the parcel's start box and its non-mat fraction (for "until": "lifted")
        self.bin_mat0, self.binhit, self.held = None, 0, False
        self.active = None  # the step executed this call (scripted_reply needs it)  # bin watchdog: mat uncovered where a bin stood -> hold for good

    @property
    def step_spec(self):
        steps = self.spec["steps"]
        return steps[self.i] if self.i < len(steps) else None

    def _arm(self, s):
        a = s.get("arm")
        return {"blue": "left"}.get(self.mat, "right") if a == "lane" else a

    def lock_arms(self):
        s = self.step_spec
        arm = self._arm(s) if s and "say" in s else None
        return tuple(x for x in ("left", "right") if arm and x != arm)

    def _read(self, images):
        front = images["front"]
        codes = [t for t, _ in locate(front) if t.startswith("420/")]
        source = "qr" if codes else None
        if not codes:
            zips = [z for z, _ in label_ocr.zips(front, self.spec.get("roi", [0.02, 0.48, 0.78, 1.0]))]
            codes, source = (["420/%s/000" % zips[0]], "zip") if zips else ([], None)
        if not codes and self.spec.get("code_hint"):
            codes, source = [self.spec["code_hint"]], "demo"  # DEMO: the label counts as read
        if codes:
            self.code, self.source, self.mat = codes[0], source, bin_for(codes[0])

    def _nonmat(self, front, box):
        x0, y0, x1, y1 = [int(v) for v in box]
        m = scene._mask(front, [self.spec["mat"]], max(2, front.shape[1] // 200))[y0:y1, x0:x1]
        return float(1 - (m > 0).mean()) if m.size else 0.0

    def step(self, images, state):
        before, self.subgoal_changed = self.i, False
        sc = self.spec.get("scene")
        if sc and sc.get("bin_rois") and self.spec.get("watchdog", True):  # watchdog (s4 run 2: with no box in the gripper WALLX dragged the red bin)
            mats = {c: scene.mat_fraction(images["front"], sc, b) for c, b in sc["bin_rois"].items()}
            if self.bin_mat0 is None:
                self.bin_mat0 = mats
            moved = [c for c, v in mats.items() if v - self.bin_mat0[c] >= 0.15]
            self.binhit = self.binhit + 1 if moved else 0
            if self.binhit >= 3 and not self.held:
                self.held = True
                if self.runlog:
                    self.runlog.write(flag="bin_moved", bins=moved)
        if self.held:
            self.state_name, self.done = "HOLD", True
            return None
        if self.spot is None and self.spec.get("mat"):  # the parcel's start spot, once
            self.spot = scene.object_box(images["front"], None, self.spec["scene"]) if self.spec.get("scene") else None
            self.spot0 = self._nonmat(images["front"], self.spot) if self.spot else None
        while self.step_spec is not None and "read" in self.step_spec:  # reads cost no robot call
            self._read(images)
            self.i += 1
            self.calls = 0
        s = self.step_spec
        instr, self.active = None, s
        if s is not None and "move" in s:  # scripted move (organisers allowed simple scripted steps): scripted_reply drives it
            instr = "(scripted move %d)" % self.i
            self.calls += 1
            if self.calls >= s.get("calls", 3):
                self.i, self.calls = self.i + 1, 0
        elif s is not None:
            instr = s["say"].format(mat=self.mat)
            self.calls += 1
            lifted = (s.get("until") == "lifted" and self.spot is not None and self.spot0
                      and self._nonmat(images["front"], self.spot) <= 0.3 * self.spot0)  # the box left its table spot: next step now
            if self.calls >= s.get("calls", 10) or lifted:
                self.i, self.calls = self.i + 1, 0
        self.subgoal_changed = self.i != before or self.calls == 1
        self.done = s is None
        self.state_name = "SEQ%d" % before
        self.last_log = {"step": before, "instruction": instr, "code": self.code, "source": self.source, "bin": self.mat,
                         "lock": list(self.lock_arms()), "left_gripper": state["left_ee"]["gripper"], "right_gripper": state["right_ee"]["gripper"]}
        if self.runlog:
            self.runlog.write(**self.last_log)
        return instr

    def scripted_reply(self, embodiment, state):
        s = self.active
        if not s or "move" not in s:
            return None
        cap, targets = s.get("max_step", 0.10), {}
        for arm, t in s["move"].items():  # {"right": {"position": [...], "rotation_rpy": [...], "gripper": g}} home-relative
            key = arm + "_ee"
            cur = np.asarray(state[key]["position"], float)
            goal = np.asarray(t.get("position", cur), float)
            d = np.linalg.norm(goal - cur)
            nxt = goal if d <= cap else cur + (goal - cur) * cap / d  # at most `cap` metres per call
            targets[key] = {"position": nxt.tolist(), "rotation_rpy": t.get("rotation_rpy", list(state[key]["rotation_rpy"]))}
            if "gripper" in t:
                targets[key]["gripper"] = t["gripper"]
        return move_response(embodiment, state, targets)

    def log_reply(self, latency_s, error=None):
        if self.runlog:
            self.runlog.write(latency_s=latency_s, error=error)

    def prompt_boxes(self, shape):
        return None, None, {}, None


if __name__ == "__main__":  # self-check
    assert bin_for("420/90024/001") == "red" and bin_for("420/95126/001") == "blue" and bin_for("x") == "black"
    import numpy as np
    sq = Sequence({"code_hint": "420/90024/001", "steps": [{"read": True}, {"say": "pick up the IQBAR box", "calls": 2, "arm": "lane"},
                                                           {"say": "put the box you are holding into the {mat} bin", "calls": 1, "arm": "lane"}]})
    img = {"front": np.zeros((720, 1280, 3), np.uint8)}
    st = {"left_ee": {"gripper": 0}, "right_ee": {"gripper": 0}}
    out = [sq.step(img, st) for _ in range(4)]
    assert out == ["pick up the IQBAR box", "pick up the IQBAR box", "put the box you are holding into the red bin", None], out
    assert sq.source == "demo" and sq.mat == "red"
    print("ok")
