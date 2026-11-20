import copy
import time

from labelloop.brain.state_machine import StateMachine
from labelloop.perception import scene
from labelloop.perception.qr import Confirmer, decode, locate


class Harness:
    """Perception + state machine + runlog; one step() per WALLX call. `agent` is accepted and unused (v3 has no vision agent)."""

    def __init__(self, task_cfg, agent=None, runlog=None, clock=time.monotonic):
        self.cfg, self.agent, self.runlog, self.clock = task_cfg, agent, runlog, clock
        self.sm = StateMachine(task_cfg)
        self.reset()

    def reset(self):
        self.sm.reset()
        self.home = None  # episode-start pose, captured on the first step()
        self.confirmer = Confirmer(self.cfg["qr_confirm_frames"])
        self.ref, self.empty = None, 0  # empty-zone reference frame
        self.calls, self.subgoal_changed, self.done, self.state_name = 0, False, False, self.sm.state

    def step(self, images, state):
        if self.home is None:
            self.home = copy.deepcopy(state)
        self.calls += 1
        before, cfg = self.sm.state, self.cfg["scene"]
        front = images["front"]
        box = scene.roi_box(front, cfg)
        codes = [(t, p) for t, p in locate(front) if scene.inside(p, box)]
        wrist = {t for t, _ in decode({k: v for k, v in images.items() if k != "front"})}
        confirmed = self.confirmer.update({t for t, _ in codes} | wrist)
        mats = scene.mats(front, cfg)
        kmats = scene.kraft_in_mats(front, mats, cfg)
        pres = 0.0 if self.ref is None else scene.presence(front, self.ref, cfg)
        if before == "IDLE":  # arms are home: keep the reference current whenever the zone has been empty a while
            self.empty = self.empty + 1 if pres < cfg["presence_min"] else 0
            if self.ref is None or self.sm.rebase or self.empty >= cfg["ref_refresh_calls"]:
                self.ref, self.empty, self.sm.rebase = scene.zone_ref(front, cfg), 0, False
        kroi = scene.kraft_in_roi(front, cfg)
        instr = self.sm.step({"presence": pres, "kroi": kroi, "codes": confirmed, "kmats": kmats})
        self.state_name, self.done = self.sm.state, self.sm.state == "DONE"
        self.subgoal_changed = self.sm.state != before
        if self.runlog:
            self.runlog.write(call=self.calls, state=before, instruction=instr, codes=codes, wrist_codes=wrist,
                              confirmed=confirmed, lane=self.sm.lane, presence=pres, kraft_roi=kroi, mats=mats,
                              kraft_mats=kmats, left_gripper=state["left_ee"]["gripper"],
                              right_gripper=state["right_ee"]["gripper"], transitions=self.sm.transitions,
                              flags=self.sm.flags, parcels=self.sm.parcels)
        return instr

    def log_reply(self, latency_s, error=None):
        if self.runlog:
            self.runlog.write(call=self.calls, latency_s=latency_s, error=error)
