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

    def _bag_in_zone(self, front):
        """Parcel tracked by its colour band (spec "track_hsv": [h0, h1, s_min, v_min]; default: the mailer's blue) inside the image
        zone of the chosen bin (spec "zones_px": {bin: [x0, y0, x1, y1] fractions}). Returns (inside, centroid) ."""
        import cv2
        h0, h1, smin, vmin = self.spec.get("track_hsv", [95, 125, 90, 70])
        hsv = cv2.cvtColor(front, cv2.COLOR_RGB2HSV)
        m = (((hsv[..., 0] > h0) & (hsv[..., 0] < h1) & (hsv[..., 1] > smin) & (hsv[..., 2] > vmin)).astype(np.uint8)) * 255
        H, W = m.shape
        m[:int(self.spec.get("table_top", 0.17) * H)] = 0
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, _, st, cen = cv2.connectedComponentsWithStats(m)
        if n < 2:
            return False, None
        i = 1 + int(np.argmax(st[1:, 4]))
        if st[i][4] < 400:
            return False, None
        x, y = cen[i][0] / W, cen[i][1] / H
        z = self.spec.get("zones_px", {}).get(self.mat)
        return bool(z and z[0] <= x <= z[2] and z[1] <= y <= z[3]), (round(x, 3), round(y, 3))

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
        if s is not None and "replay" in s:  # teach and repeat: the recorded demo, 32 rows per call
            instr = "(replay %d)" % self.i
            self.calls += 1
            self.replay = (s, self._demo_rows(s, self.calls))
            if self.replay[1] is None or self.calls >= s.get("calls", 10**6) or self._demo_rows(s, self.calls + 1) is None:
                self.i, self.calls = self.i + 1, 0
        elif s is not None and "release" in s:
            instr = "(release %d)" % self.i
            self.calls += 1
            if self.calls >= s.get("calls", 4):
                self.i, self.calls = self.i + 1, 0
        elif s is not None and "carry" in s:  # scripted two-arm carry of the held box (both arms keep their offset)
            instr = "(carry %d)" % self.i
            self.calls += 1
            if self.calls >= s.get("calls", 6):
                self.i, self.calls = self.i + 1, 0
        elif s is not None and "move" in s:  # scripted move (organisers allowed simple scripted steps): scripted_reply drives it
            instr = "(scripted move %d)" % self.i
            self.calls += 1
            if self.calls >= s.get("calls", 3):
                self.i, self.calls = self.i + 1, 0
        elif s is not None:
            instr = s["say"].format(mat=self.mat, zone=self.spec.get("zones", {}).get(self.mat, ""))
            self.calls += 1
            if s.get("until") == "in_zone":  # cut WALLX off as soon as the parcel is over its zone (m1: it kept carrying it around)
                inside, c = self._bag_in_zone(images["front"])
                self.zonecount = getattr(self, "zonecount", 0) + 1 if inside else 0
                if self.runlog:
                    self.runlog.write(track=c, inside=inside)
                if self.zonecount >= s.get("zone_calls", 2):
                    self.i, self.calls, self.zonecount = self.i + 1, 0, 0
                    return instr
            lifted = (s.get("until") == "lifted" and self.spot is not None and self.spot0
                      and self._nonmat(images["front"], self.spot) <= 0.3 * self.spot0  # the box left its table spot ...
                      and all(0.08 < float(state[a + "_ee"]["gripper"]) < 0.7 for a in s.get("hold", [])))  # ... and the claw(s) grip something
            self.liftcount = getattr(self, "liftcount", 0) + 1 if lifted else 0
            lifted = self.liftcount >= s.get("lift_calls", 2)  # held for 2 calls (h1 run 1: a one-call false positive)
            if lifted:
                self.i, self.calls = self.i + 1, 0
            elif self.calls >= s.get("calls", 10):  # timed out: an "until lifted" step that never lifted skips to the last step (home)
                self.i = len(self.spec["steps"]) - 1 if s.get("until") == "lifted" else self.i + 1
                self.calls = 0
                if s.get("until") == "lifted" and self.runlog:
                    self.runlog.write(flag="not_lifted: skipped to home")
        self.subgoal_changed = self.i != before or self.calls == 1
        self.done = s is None
        self.state_name = "SEQ%d" % before
        self.last_log = {"step": before, "instruction": instr, "code": self.code, "source": self.source, "bin": self.mat,
                         "lock": list(self.lock_arms()), "left_gripper": state["left_ee"]["gripper"], "right_gripper": state["right_ee"]["gripper"]}
        if self.runlog:
            self.runlog.write(**self.last_log)
        return instr

    def _demo_rows(self, s, k):
        import json as _json
        if "_demo" not in s:
            s["_demo"] = _json.loads(open(s["replay"]).read())
        d = s["_demo"]
        a = 32 * (k - 1)
        return None if a >= len(d["right"]) else (a, min(a + 32, len(d["right"])))

    def _carry(self, embodiment, state, s):
        c = s["carry"]
        if "_p0" not in s:  # pose and grip when the carry phase started (shared by the following carry steps via "from": "prev")
            src = self.carry0 if c.get("from") == "prev" and getattr(self, "carry0", None) else None
            s["_p0"] = src or {a: (np.asarray(state[a]["position"], float), list(state[a]["rotation_rpy"]), float(state[a]["gripper"])) for a in ("left_ee", "right_ee")}
            self.carry0 = s["_p0"]
        p0 = s["_p0"]
        if c.get("anchor") == "mid":  # box centre = midpoint of the two claws, in the right arm's frame (frames parallel, +y left;
            off = np.array([0.0, c.get("base_sep", 0.51), 0.0])  # left base ~0.51 m left of the right base: t_tip_A run 1 squeeze)
            ref = (p0["right_ee"][0] + p0["left_ee"][0] + off) / 2
        else:
            ref = np.asarray(p0[c.get("anchor", "right") + "_ee"][0], float)
        to = ref.copy()
        if "to" in c:
            to[:2] = c["to"][:2]
        to[2] = ref[2] + c.get("dz", 0.0)
        delta = to - ref
        cap, targets = s.get("max_step", 0.04), {}
        arms = [x + "_ee" for x in c.get("arms", ["left", "right"])]  # single-arm carry: name only the holding arm (h2 run 1 bug)
        for a in arms:
            goal = p0[a][0] + delta
            cur = np.asarray(state[a]["position"], float)
            d = np.linalg.norm(goal - cur)
            nxt = goal if d <= cap else cur + (goal - cur) * cap / d
            g = c["grip"] if "grip" in c else max(0.0, p0[a][2] - c.get("squeeze", 0.15))  # hold: a bit more closed than measured
            targets[a] = {"position": nxt.tolist(), "rotation_rpy": p0[a][1], "gripper": g}
        return move_response(embodiment, state, targets)

    def scripted_reply(self, embodiment, state):
        s = self.active
        if s and "release" in s:  # open the acting arm's claw fully where it is, then lift it (release over the zone)
            arm = {"blue": "left"}.get(self.mat, "right") if s["release"] == "lane" else s["release"]
            k = arm + "_ee"
            pos = np.asarray(state[k]["position"], float)
            if self.calls == 0 or self.calls > s.get("open_calls", 2):  # step advanced (calls reset) or lifting phase
                pos = pos + np.array([0, 0, min(0.04, s.get("lift", 0.08))])
            return move_response(embodiment, state, {k: {"position": pos.tolist(), "rotation_rpy": list(state[k]["rotation_rpy"]), "gripper": 1.5}})
        if s and "carry" in s:
            return self._carry(embodiment, state, s)
        if s and "replay" in s:
            span = self.replay[1] if getattr(self, "replay", None) and self.replay[0] is s else None
            if span is None:
                return None
            d, (a, b) = s["_demo"], span
            acts = {}
            for arm in ("left", "right"):
                rows = d[arm][a:b] + [d[arm][b - 1]] * (32 - (b - a))
                acts[arm + "_ee"] = {"position": [r[:3] for r in rows], "rotation_rpy": [r[3:6] for r in rows], "gripper": [r[6] for r in rows]}
            return {"embodiment": embodiment, "actions": acts}
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
