import copy
import time

import cv2

from labelloop import look
from labelloop.brain.state_machine import StateMachine
from labelloop.perception import scene
from labelloop.perception.qr import Confirmer, decode, locate
from labelloop.wallx_client import home_response, move_response


class Harness:
    """Perception + state machine + runlog; one step() per WALLX call. `agent` is accepted and unused (v3 has no vision agent)."""

    def __init__(self, task_cfg, agent=None, runlog=None, clock=time.monotonic):
        self.cfg, self.agent, self.runlog, self.clock = task_cfg, agent, runlog, clock
        self.sm = StateMachine(task_cfg)
        self.reset()

    def reset(self):
        self.sm.reset()
        self.home = None  # episode-start pose, captured on the first step()
        self.views = []  # LOOK viewpoints, built at SCAN -> LOOK
        self.confirmer = Confirmer(self.cfg["qr_confirm_frames"])
        self.ref, self.empty, self.bin_refs = None, 0, {}  # empty-zone reference frame; bin snapshots
        self.calls, self.subgoal_changed, self.done, self.state_name = 0, False, False, self.sm.state

    def step(self, images, state):
        if self.home is None:
            self.home = copy.deepcopy(state)
        self.calls += 1
        before, cfg = self.sm.state, self.cfg["scene"]
        front = images["front"]
        box = scene.roi_box(front, cfg)
        if before in ("LOOK", "INSPECT", "HANDOVER"):  # parcel in hand: a label counts anywhere but in a bin (a labelled parcel lying in one)
            bins = [scene.roi_box(front, cfg, "bin_rois", c) for c in cfg["bin_rois"]]
            codes = [(t, p) for t, p in locate(front) if not any(scene.inside(p, b) for b in bins)]
        else:
            codes = [(t, p) for t, p in locate(front) if scene.inside(p, box)]
        if not codes and before not in ("LOOK", "INSPECT", "HANDOVER"):  # small QR missed on this frame: retry on the ROI crop at 2x
            crop = cv2.resize(front[box[1]:box[3], box[0]:box[2]], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            codes = [(t, (box[0] + x / 2, box[1] + y / 2)) for t, (x, y) in locate(crop)]
        wrist = {t for t, _ in decode({k: v for k, v in images.items() if k != "front"})}
        confirmed = self.confirmer.update({t for t, _ in codes} | wrist)
        pres = 0.0 if self.ref is None else scene.presence(front, self.ref, cfg)
        if before == "IDLE":  # arms are home: keep the reference current whenever the zone has been empty a while
            self.empty = self.empty + 1 if pres < cfg["presence_min"] else 0
            if self.ref is None or self.sm.rebase or self.empty >= cfg["ref_refresh_calls"]:
                self.ref, self.empty, self.sm.rebase = scene.zone_ref(front, cfg), 0, False
        kroi = scene.kraft_in_roi(front, cfg)
        bins = {c: scene.presence(front, r, cfg, "bin_rois", c) for c, r in self.bin_refs.items()}  # {} until the first SCAN
        n = len(self.views) * self.look_span()
        instr = self.sm.step({"presence": pres, "kroi": kroi, "codes": confirmed, "bins": bins, "look_calls": n})
        if before == "SCAN" and self.sm.state == "LOOK":  # parcel point from the changed pixels of the zone, once
            lc = self.cfg["look"]
            self.views = look.views(look.pixel_to_table(*scene.centroid(front, self.ref, cfg), front.shape[1::-1]), lc)
        if before == "IDLE" and self.sm.state == "SCAN":  # object arrived, arms home: snapshot every bin
            self.bin_refs = {c: scene.zone_ref(front, cfg, "bin_rois", c) for c in cfg["bin_rois"]}
        self.state_name, self.done = self.sm.state, self.sm.state == "DONE"
        self.subgoal_changed = self.sm.state != before
        if self.runlog:
            self.runlog.write(call=self.calls, state=before, instruction=instr, codes=codes, wrist_codes=wrist,
                              confirmed=confirmed, lane=self.sm.lane, presence=pres, bin_change=bins, kraft_roi=kroi,
                              left_gripper=state["left_ee"]["gripper"],
                              right_gripper=state["right_ee"]["gripper"], transitions=self.sm.transitions,
                              flags=self.sm.flags, parcels=self.sm.parcels, look_view=self.look_index)
        return instr

    def look_span(self):
        return self.cfg.get("look", {}).get("move_calls", 0) + self.cfg.get("look", {}).get("dwell_calls", 0)

    @property
    def look_index(self):
        """Index of the viewpoint being driven during LOOK, else None."""
        if self.sm.state != "LOOK" or not self.views:
            return None
        return min(self.sm.calls // self.look_span(), len(self.views) - 1)

    def scripted_reply(self, embodiment, state):
        """WALLX-shaped reply for a scripted step (home, look-around), or None when WALLX should answer."""
        if self.sm.state == "LOOKHOME" or (self.cfg.get("scripted_home") and self.sm.state == "HOME"):
            return home_response(embodiment, state, self.home)
        if self.look_index is not None:
            arm, pos, rpy = self.views[self.look_index]
            return move_response(embodiment, state, {arm: {"position": pos.tolist(), "rotation_rpy": rpy.tolist()}})
        if self.sm.state == "LOOK":  # every viewpoint unreachable: just hold
            return move_response(embodiment, state, {})

    def log_reply(self, latency_s, error=None):
        if self.runlog:
            self.runlog.write(call=self.calls, latency_s=latency_s, error=error)
