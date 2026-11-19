import re

from labelloop.brain.instructions import instruction
from labelloop.brain.memory import Memory

WRIST_VIEWS = ("right_wrist", "left_wrist")  # front sees every box; its codes only feed memory.seen
TIMEOUT_TO = {"LOOK": "GRASP", "GRASP": "SURVEY", "PRESENT": "REJECT", "PLACE": "SURVEY", "REJECT": "SURVEY"}


class StateMachine:
    """step(evidence) -> instruction | None.
    evidence = {"now": float, "grip": left gripper, "codes": [(text, view)] confirmed, "verdict": agent dict | None}.
    Returns the instruction of the state we end in (SURVEY -> None = hold)."""

    def __init__(self, cfg):
        self.cfg, self.pattern = cfg, re.compile(cfg["code_pattern"])
        self.reset(0.0)

    def reset(self, now):
        self.memory, self.transitions = Memory(), []
        self._enter("SURVEY", now)

    def _enter(self, state, now):
        self.state, self.t0, self.calls, self.hold, self.was_open, self.prev_grip = state, now, 0, 0, False, 0.0
        if state == "SURVEY":
            self.memory.current_code, self.retries = None, 0

    def _go(self, state, why, now):
        self.transitions.append((self.state, state, why))
        self._enter(state, now)

    def instruction(self):
        if self.state == "SURVEY":
            return None
        zone = self.memory.seen.get(self.memory.current_code)
        return instruction(self.state, self.cfg, self.cfg["zones"].get(zone))

    def _pick(self, new, views):
        for view in views:
            for text, v in new:
                if v == view:
                    return text

    def step(self, ev):
        now, grip, verdict = ev["now"], ev["grip"], ev.get("verdict")
        cfg, g, m, s = self.cfg, self.cfg["gripper"], self.memory, self.state
        self.transitions = []
        new = []  # confirmed, well-formed, not yet sorted
        for text, view in ev.get("codes", []):
            match = self.pattern.match(text)
            if match and text not in m.sorted:
                m.seen[text] = match.group(1)
                new.append((text, view))
        self.calls += 1

        if s == "SURVEY":
            if verdict:
                m.unsorted_estimate, m.bin_side = verdict.get("unsorted_boxes"), verdict.get("bin_side")
            if verdict and verdict.get("unsorted_boxes") == 0:
                self._go("DONE", "nothing left", now)
            else:
                self._go("LOOK", "survey", now)
            return self.instruction()
        if s == "DONE":
            return self.instruction()

        if s == "LOOK":
            code = self._pick(new, WRIST_VIEWS)
            if code:
                m.current_code = code
                self._go("GRASP", "code", now)
            elif self.calls >= cfg["look_attempts"] * cfg["calls_per_attempt"]:
                self._go("GRASP", "code unread", now)
        elif s == "GRASP":
            self._grasp(grip, verdict, now)
        elif s == "PRESENT":
            code = self._pick(new, WRIST_VIEWS)
            if code:
                m.current_code = code
                self._go("PLACE", "code", now)
            elif self.calls >= cfg["present_faces"] * cfg["calls_per_attempt"]:
                self._go("REJECT", "no code", now)
        elif s == "PLACE":
            if grip >= g["open_min"]:
                m.sorted.add(m.current_code)
                self._go("SURVEY", "released", now)
            elif grip <= g["empty_max"]:
                self._go("SURVEY", "dropped", now)
        elif s == "REJECT":
            if grip >= g["open_min"]:
                m.rejected_count += 1
                self._go("SURVEY", "released", now)
            elif grip <= g["empty_max"]:
                self._go("SURVEY", "dropped", now)

        if self.state == s and s in TIMEOUT_TO and now - self.t0 >= cfg["state_timeout_s"]:
            self._go(TIMEOUT_TO[s], "timeout", now)
        return self.instruction()

    def _grasp(self, grip, verdict, now):
        g = self.cfg["gripper"]
        self.was_open |= grip >= g["open_min"]
        holding = g["empty_max"] < grip < g["hold_max"]
        steady = self.hold and abs(grip - self.prev_grip) <= g["hold_tol"]
        self.hold = (self.hold + 1 if steady else 1) if holding else 0
        self.prev_grip = grip
        if self.hold >= g["hold_calls"]:
            self._go("PLACE" if self.memory.current_code else "PRESENT", "grasped", now)
            return
        if (grip <= g["empty_max"] and self.was_open) or (verdict and verdict.get("done") and not holding):
            self.retries += 1
            self.was_open = False
            if self.retries >= self.cfg["grasp_retries"]:
                self._go("SURVEY", "grasp failed", now)
