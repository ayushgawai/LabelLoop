import re

from labelloop.brain.instructions import instruction
from labelloop.perception.scene import inside, where

# SURVEY -> PLACE (readable code on the table) | FETCH (kraft block, no readable code) | DONE
# PLACE -> SURVEY; FETCH -> PLACE_HELD (wrist reads a code while holding) | REJECT (holding, no code) | SURVEY
# PLACE_HELD -> SURVEY; REJECT -> SURVEY; DONE stays.


def _arm():
    return {"open": False, "hold": 0, "prev": 0.0, "held": False, "released": False}


class StateMachine:
    """step(ev) -> instruction | None (None = hold still).
    ev = {"w", "h": image size, "front": [(code, (x, y))] located, "confirmed": {code}, "wrist": {code},
          "bins": {color: bbox|None}, "blocks": [(x, y)], "kraft": {color: area fraction in bin bbox},
          "grips": (left, right), "verdict": agent dict | None}."""

    def __init__(self, cfg):
        self.cfg, self.pattern = cfg, re.compile(cfg["code_pattern"])
        self.reset()

    def reset(self):
        self.sorted, self.wrong, self.gave_up, self.retries, self.rejected_count = set(), set(), set(), {}, 0
        self.arms, self.frames, self.transitions, self.flags = [_arm(), _arm()], 0, [], []
        self.code, self.where, self.base, self.gone, self.released = None, "in the middle", {}, 0, False
        self.ignored, self.before, self.learn = [], [], False  # rejected blocks stay on the table: remember where
        self.state, self.calls = "SURVEY", 0

    def _go(self, state, why):
        self.transitions.append((self.state, state, why))
        self.state, self.calls, self.gone, self.released = state, 0, 0, False

    def instruction(self):
        if self.state == "SURVEY":
            return None
        zone = self.pattern.match(self.code or "")
        return instruction(self.state, self.cfg, where=self.where, bin=zone and self.cfg["zones"][zone.group(1)])

    def _grips(self, grips):
        g = self.cfg["gripper"]
        for a, v in zip(self.arms, grips):
            a["released"] = False
            if v >= g["open_min"]:
                a["released"], a["held"], a["open"], a["hold"] = a["held"], False, True, 0
            elif v <= g["empty_max"]:
                a.update(open=False, hold=0, held=False)
            elif g["hold_min"] < v < g["hold_max"] and a["open"]:
                a["hold"] = a["hold"] + 1 if a["hold"] and abs(v - a["prev"]) <= g["hold_tol"] else 1
                a["held"] |= a["hold"] >= g["hold_calls"]
            else:
                a["hold"], a["held"] = 0, a["held"] and v >= g["hold_min"]
            a["prev"] = v

    def _landed(self, ev):
        """Bin colour the cube went into, or None: its code seen inside a bin, else kraft area gained in a bin."""
        for color, b in ev["bins"].items():
            if self.code in ev["located"] and inside(ev["located"][self.code], b):
                return color
        gain = {c: ev["kraft"][c] - self.base.get(c, 0) for c in ev["kraft"]}
        best = max(gain, key=gain.get)
        return best if gain[best] >= self.cfg["scene"]["bin_gain"] else None

    def _finish(self, color):
        want = self.cfg["zones"][self.pattern.match(self.code).group(1)]
        self.sorted.add(self.code)
        if color != want:
            self.wrong.add(self.code)
            self.flags.append("wrong_bin:%s:%s" % (self.code, color))
        self._go("SURVEY", "sorted" if color == want else "wrong_bin")

    @staticmethod
    def _centre_dist(p, ev):
        return (p[0] - ev["w"] / 2) ** 2 + (p[1] - ev["h"] / 2) ** 2

    def step(self, ev):
        cfg = self.cfg
        self.transitions, self.flags = [], []
        self.frames += 1
        self.calls += 1
        self._grips(ev["grips"])
        holding, released = any(a["held"] for a in self.arms), any(a["released"] for a in self.arms)
        good = lambda t: self.pattern.match(t) and t not in self.sorted  # noqa: E731
        located = ev["located"] = {t: p for t, p in ev["front"] if self.pattern.match(t)}
        on_table = {t: p for t, p in located.items() if not any(inside(p, b) for b in ev["bins"].values())}
        s = self.state

        if s == "SURVEY":
            targets = {t: p for t, p in on_table.items() if good(t) and t in ev["confirmed"] and t not in self.gave_up}
            if self.frames < cfg["qr_confirm_frames"]:
                return None
            v = ev.get("verdict")
            near = lambda p, pts: any(abs(p[0] - q[0]) + abs(p[1] - q[1]) < 0.08 * ev["w"] for q in pts)  # noqa: E731
            if self.learn:  # first look after a rejection: the block that appeared is the rejected one
                self.ignored += [b for b in ev["blocks"] if not near(b, self.before)]
                self.learn = False
            blocks = [b for b in ev["blocks"] if not near(b, self.ignored)]
            if targets:
                self.code = min(targets, key=lambda t: self._centre_dist(targets[t], ev))
                self.where, self.base = where(targets[self.code][0], ev["w"]), dict(ev["kraft"])
                self._go("PLACE", "code on table")
            elif blocks and not (v and v.get("unsorted_boxes") == 0):
                self.code, self.where = None, where(min(blocks, key=lambda p: self._centre_dist(p, ev))[0], ev["w"])
                self._go("FETCH", "block without readable code")
            else:
                self._go("DONE", "nothing left")
        elif s == "PLACE":
            if self.code in on_table:
                self.gone, self.where = 0, where(on_table[self.code][0], ev["w"])
            else:
                self.gone += 1
            color = self._landed(ev) if self.gone >= 2 else None
            if color:
                self._finish(color)
            elif self.calls >= cfg["place_max_calls"]:
                n = self.retries[self.code] = self.retries.get(self.code, 0) + 1
                if n >= cfg["place_retries"]:
                    self.gave_up.add(self.code)
                self._go("SURVEY", "timeout")
        elif s == "FETCH":
            if ev["blocks"]:
                self.where = where(min(ev["blocks"], key=lambda p: self._centre_dist(p, ev))[0], ev["w"])
            codes = [t for t in sorted(ev["wrist"]) if good(t) and t in ev["confirmed"]]
            if codes and holding:
                self.code, self.base = codes[0], dict(ev["kraft"])
                self._go("PLACE_HELD", "code read from wrist")
            elif self.calls >= cfg["fetch_max_calls"]:
                self.before = list(ev["blocks"])
                self._go("REJECT" if holding else "SURVEY", "timeout")
        elif s == "PLACE_HELD":
            self.released |= released
            color = self._landed(ev) if self.released else None
            if color:
                self._finish(color)
            elif self.calls >= cfg["place_held_max_calls"]:
                self._go("SURVEY", "timeout")
        elif s == "REJECT":
            if released:
                self.rejected_count, self.learn = self.rejected_count + 1, True
                self._go("SURVEY", "rejected")
            elif self.calls >= cfg["reject_max_calls"]:
                self._go("SURVEY", "timeout")
        return self.instruction()
