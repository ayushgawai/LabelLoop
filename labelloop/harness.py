import copy
import time
from pathlib import Path

import cv2
import numpy as np

from labelloop import look
from labelloop.brain.state_machine import StateMachine
from labelloop.perception import scene
from labelloop.perception.qr import Confirmer, decode, locate
from labelloop.perception import label_ocr
from labelloop.wallx_client import home_response, move_response


FOOT = 1.0  # held-parcel disc radius = FOOT x half the box diagonal + 3 cm


def apply_profile(cfg):
    """Bind the camera model of cfg["profile"] (default sim) into look.py and its arm frame into cfg["look"]; real_safe forces every scripted
    arm motion off (hold + scripted HOME only), one parcel, scripted home. Returns a new cfg."""
    cfg = copy.deepcopy(cfg)
    if cfg.get("event_fast"):
        cfg["real_safe"] = True  # event_fast sits on top of real_safe
    prof = cfg["profiles"][cfg.get("profile", "sim")]
    ab = look.set_profile(prof)
    cfg["look"].update(ab)
    if prof.get("bin_rois"):  # measured on the real head image: overrides the sim bin regions (blanking, bin-change success)
        cfg["scene"]["bin_rois"] = prof["bin_rois"]
    if prof.get("induction_roi"):
        cfg["scene"]["induction_roi"] = prof["induction_roi"]
    if prof.get("whole_table") is not None:
        cfg["scene"]["whole_table"] = prof["whole_table"]
    if prof.get("empty_ref"):  # a JPEG of the empty table = the initial zone reference (relative to the repo root, else absolute)
        p = Path(prof["empty_ref"])
        cfg["empty_ref"] = str(p if p.is_absolute() else Path(__file__).resolve().parents[1] / p)
    else:
        cfg["empty_ref"] = None
    if cfg.get("real_safe"):
        for k in ("show", "spin", "slide", "look"):
            cfg[k]["enabled"] = False
        cfg.update(calib_test=False, handover=False, scripted_home=True, max_parcels=1)
        cfg["tries"] = {**cfg["tries"], "SWIPE": 0, "INSPECT": 0}
    if cfg.get("event_fast"):  # WALLX-only and fast: no scripted arm motion, HOME by WALLX once per parcel, flips / rejects / oversize by WALLX, real-time budget
        cfg.update(reject_mode="tray", scripted_home=False, home_calls=1, ok_calls=1, qr_confirm_frames=2, scan_max_calls=4, stable_calls=1)
        cfg["tries"] = {**cfg["tries"], "INSPECT": 2, "PLACE": 2, "REJECT": 2, "LIFT2": 2}  # few tries, confident grips (grasp abort in the state machine)
        cfg["max_calls"] = {**cfg["max_calls"], "INSPECT": 6}
        cfg["budget"] = {**cfg["budget"], "case_s": 480, "deliver_reserve_s": 20, "wallx_call_s": 2.5, "sim_case_s": None}
        cfg["instructions"] = {**cfg["instructions"], "PLACE": ["put the box into the {mat} bin"], "INSPECT": ["flip the box over"], "REJECT": ["put the box into the black bin"],
                               "LIFT2": ["pick up the box with both arms and put it into the {mat} bin"]}
    if cfg.get("wording"):  # trial-agent / operator overrides of the WALLX instruction templates ({mat} {arm} {where} {obj})
        cfg["instructions"] = {**cfg["instructions"], **{k: list(v) if isinstance(v, (list, tuple)) else [v] for k, v in cfg["wording"].items()}}
    return cfg


class Harness:
    """Perception + state machine + runlog; one step() per WALLX call. `agent` is accepted and unused (v3 has no vision agent)."""

    def __init__(self, task_cfg, agent=None, runlog=None, clock=time.monotonic):
        if task_cfg.get("reject_mode", "tray") != "tray":  # no tray: its ROI is not a bin (it would mask the table and a code seen there would be ignored)
            task_cfg = {**task_cfg, "scene": {**task_cfg["scene"], "bin_rois": {k: v for k, v in task_cfg["scene"]["bin_rois"].items() if k != "black"}}}
        task_cfg = apply_profile(task_cfg)
        self.cfg, self.agent, self.runlog, self.clock = task_cfg, agent, runlog, clock
        self.sm = StateMachine(task_cfg, clock)
        self.reset()

    def reset(self):
        self.sm.reset()
        self.last_pose = None
        self.home = None  # episode-start pose, captured on the first step()
        self.views, self.point, self.look_info = [], None, None  # LOOK viewpoints / point_test pose, built at SCAN -> LOOK
        self.confirmer = Confirmer(self.cfg["qr_confirm_frames"])  # items are (camera, text): same text in N consecutive frames of ONE camera
        self.spin, self.spin_info, self.spin_sigma = None, None, None  # the planned push of the current SPIN try; the yaw direction all tries keep
        self.show, self.show_info, self.sset, self.show_done = None, None, None, 0  # SHOW plan; its log; the SHOWSET (set-down) plan; SHOW calls driven
        self.slide, self.slide_info = None, None  # the planned stroke of the current SLIDE entry (None = nothing to push)
        self.swipe, self.swipe_info = None, None  # the same for the reject swipe (SWIPE)
        self.calib, self.cal_n, self.cal_cur = None, 0, None  # calib_test schedule, calls consumed, entry of the current call
        self.ref, self.empty, self.bin_refs = None, 0, {}  # empty-zone reference frame; bin snapshots
        self.ref_cfg = False  # the reference is the configured empty-table image (not the live first frame)
        self._empty = False  # the loaded empty-table JPEG resized to the frame (False = not read yet, None = unreadable)
        self.obj_box, self.near = None, None  # event_fast: object on the table with no readable label; the arm nearest it
        self.code_px = None  # front-camera position of the last accepted head code (kept while the label is occluded)
        self.calls, self.subgoal_changed, self.done, self.state_name = 0, False, False, self.sm.state
        self.last_log = None  # the fields passed to runlog.write at the end of step() (the relay records them)

    def step(self, images, state):
        self.sm.begin()  # event_fast: the case clock starts with the first observation
        if self.home is None:
            if self.cfg.get("real_safe"):  # home = a pose seen twice in a row (both arms + grippers still); until then hold
                pose = np.array([*state["left_ee"]["position"], *state["right_ee"]["position"]])
                if self.last_pose is None or np.abs(pose - self.last_pose).max() >= self.cfg["home_still_m"]:
                    self.last_pose = pose
                    return None
            self.home = copy.deepcopy(state)
        self.calls += 1
        if self.cfg.get("calib_test") and self.calib is None:
            self.calib = self.plan_calib(images["front"].shape[1::-1])
        if self.calib is not None:  # calibration latches: the schedule, then hold still for the rest of the episode (never sorts)
            return self.calib_step(state)
        self.cal_cur = None
        before, cfg = self.sm.state, self.cfg["scene"]
        front = images["front"]
        qf = scene.blank_bins(front, cfg) if self.cfg.get("real_safe") else front  # codes are decoded from this: bins blanked first (real_safe)
        box = scene.roi_box(front, cfg)
        if self.ref is None and self.cfg.get("empty_ref"):  # the configured empty-table JPEG is the initial reference (no parcel baked into it)
            e = self.empty_frame(front)
            self.ref, self.ref_cfg = (scene.zone_ref(e, cfg), True) if e is not None else (None, False)
        self.obj_box, self.near = None, None
        if self.cfg.get("event_fast"):  # a parcel on the table even with an unreadable label: non-mat colour, or change vs the configured empty image
            self.obj_box = scene.object_box(front, self.ref if self.ref_cfg else None, cfg)
            self.near = None if self.obj_box is None else "left" if (self.obj_box[0] + self.obj_box[2]) / 2 < front.shape[1] / 2 else "right"
        pre = self.sm.calls  # index of the scripted call whose result this image shows
        on = self.on_parcel(before, front, cfg, box, pre)  # head code centre -> does it lie on the tracked parcel?
        codes = [(t, p) for t, p in locate(qf) if on(p)]
        if codes:
            self.code_px = codes[0][1]
        cb = scene.changed_box(front, self.ref, cfg) if self.ref is not None and before in ("SCAN", "SPIN") else None
        if not codes and cb:  # small QR missed on this frame: retry on the changed-pixel crop (the parcel, anywhere on the table) at 2x
            cb = (max(cb[0] - 20, 0), max(cb[1] - 20, 0), cb[2] + 20, cb[3] + 20)
            crop = cv2.resize(qf[cb[1]:cb[3], cb[0]:cb[2]], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            codes = [(t, p) for t, p in ((t, (cb[0] + x / 2, cb[1] + y / 2)) for t, (x, y) in locate(crop)) if on(p)]
            if codes:
                self.code_px = codes[0][1]
        if not codes and self.cfg.get("event_fast") and self.cfg.get("zip_ocr", True) and before in ("SCAN", "CENTER"):  # QR too small: OCR the printed ZIP (confirmed over frames)
            codes = [("420/%s/000" % z, p) for z, p in label_ocr.zips(qf, cfg["induction_roi"]) if on(p)]
            if codes:
                self.code_px = codes[0][1]
        wcams = {k: v for k, v in images.items() if k in self.wrist_cams(before, pre)}
        wrist = {(v, t) for t, v in decode(wcams)}
        if wrist and self.cfg.get("event_fast"):  # only a parcel-format code not already sorted and not the one the head sees sitting in a bin
            blind = {t for t, p in locate(front) if scene.in_bin(p, front, cfg)}
            wrist = {(v, t) for v, t in wrist if self.sm.pattern.match(t) and t not in self.sm.sorted and t not in blind}
        seen = {("front", t) for t, _ in codes} | wrist
        confirmed = {t for _, t in self.confirmer.update(seen)}
        pres = 0.0 if self.ref is None else scene.presence(front, self.ref, cfg)
        present = bool(before == "IDLE" and self.calls == 1 and self.cfg.get("event_fast")
                       and (pres >= cfg["presence_min"] or codes or self.obj_box is not None))  # a parcel was already there at the episode start
        if before == "IDLE":  # arms are home: keep the reference current whenever the zone has been empty a while
            empty = pres < cfg["presence_min"] or (self.cfg.get("event_fast") and self.obj_box is None and not codes)  # (event_fast: no object in the image = a stale reference)
            self.empty = self.empty + 1 if empty else 0
            if self.ref is None or self.sm.rebase or self.empty >= cfg["ref_refresh_calls"]:
                self.ref, self.ref_cfg, self.empty, self.sm.rebase = scene.zone_ref(front, cfg), False, 0, False
        kroi = scene.kraft_in_roi(front, cfg)
        bins = {c: scene.presence(front, r, cfg, "bin_rois", c) for c, r in self.bin_refs.items()}  # {} until the first SCAN
        n = 10**9 if self.point else len(self.views) * self.look_span()  # point_test: hold until the episode is stopped
        push = len(self.spin["wps"]) + self.cfg["home_calls"] if self.spin else 0
        sc = self.cfg.get("spin", {}).get("scan_calls", 0)
        ev = {"presence": pres, "kroi": kroi, "codes": confirmed, "bins": bins, "look_calls": n, "raw": {t for _, t in seen},
              "grip": {"left": state["left_ee"]["gripper"], "right": state["right_ee"]["gripper"]},
              "show_calls": len(self.show["calls"]) if self.show else 0,
              "showset_calls": len(self.sset["calls"]) + self.cfg["home_calls"] if self.sset else self.cfg["home_calls"],
              "spin_push": push, "spin_calls": push + sc if self.spin else 1,
              "swipe_calls": len(self.swipe["wps"]) + self.cfg["home_calls"] + self.cfg["swipe"]["settle_calls"] if self.swipe else 0,
              "slide_calls": len(self.slide["wps"]) + self.cfg["home_calls"] + self.cfg["slide"]["settle_calls"] if self.slide else 0}  # no feasible push: the try is skipped
        if self.cfg.get("event_fast"):  # the object on the table (label readable or not) / a parcel already there at the start / the arm nearest it
            ev.update(present=present, obj=self.obj_box is not None, near=self.near, where=self.where_words(front, cfg, codes))
        instr = self.sm.step(ev)
        if before == "SCAN" and self.sm.state == "LOOK":  # parcel point from the changed pixels of the zone, once
            self.plan_look(front, cfg)
        if self.sm.state == "SHOW" and self.sm.calls == 0:  # grasp-and-show: nothing to grasp -> the next search step at once
            self.plan_show(front, cfg)
            if not self.show:
                self.sm.end_show()
                instr = self.sm.instruction()
        if before == "SHOW" and self.sm.state == "SHOW" and self.show:  # grasp checks: the gripper closed past empty = missed; the 2 cm test lift moves the parcel
            bad = self.grasp_check(front, state, pre)
            if bad:
                self.show_info = {**self.show_info, "grasp": bad}
                self.sm.abort_show(bad == "missed" and self.sm.tries.get("SHOW", 0) < 2)  # a miss: one retry from a fresh fit
                instr = self.sm.instruction()
        if before == "SHOW" and self.sm.state == "SHOWSET":  # the parcel is in hand (or not yet gripped): put it back where the delivery arm can take it
            self.show_done = pre + 1
            if self.show:
                self.sm.faces |= {c["face"] for c in self.show["calls"][:self.show_done] if c["tag"] == "hold"}  # a face counts once its hold call was driven
            self.plan_showset()
        if self.sm.state == "SPIN" and self.sm.calls == 0:  # a new SPIN try (arm home, parcel as it lies now)
            self.plan_spin(front, cfg)
        if self.sm.state == "SLIDE" and self.sm.calls == 0:  # a new stroke (arm home, parcel as it lies now): nothing to push -> the delivery arm acts
            self.plan_slide(front, cfg)
            if self.sm.state == "SLIDE" and not self.slide:
                self.sm.end_slide()
            instr = self.sm.instruction()
        if self.sm.state == "SWIPE" and self.sm.calls == 0:  # a new stroke (arm home, parcel as it lies now): nothing to push -> give up this try
            self.plan_swipe(front, cfg)
            if not self.swipe:
                self.sm.end_swipe()
            instr = self.sm.instruction()
        if before == "IDLE":  # a new parcel starts with no QR history and no preferred yaw direction
            self.confirmer.counts.clear()
            self.spin_sigma = None
        if before == "IDLE" and self.sm.state == "SCAN":  # object arrived, arms home: snapshot every bin
            self.bin_refs = {c: scene.zone_ref(front, cfg, "bin_rois", c) for c in cfg["bin_rois"]}
        self.state_name, self.done = self.sm.state, self.sm.state == "DONE"
        self.subgoal_changed = self.sm.state != before or any(t[1] == before for t in self.sm.transitions)  # a retry restarts the same sub-goal
        self.last_log = {"call": self.calls, "state": before, "instruction": instr, "codes": codes, "wrist_codes": wrist,
                         "confirmed": confirmed, "lane": self.sm.lane, "presence": pres, "bin_change": bins, "kraft_roi": kroi,
                         "left_gripper": state["left_ee"]["gripper"], "right_gripper": state["right_ee"]["gripper"],
                         "transitions": self.sm.transitions, "flags": self.sm.flags, "parcels": self.sm.parcels,
                         "look_view": self.look_index, "look": self.look_info, "spent_s": self.sm.spent,
                         "spin": self.spin_info, "slide": self.slide_info, "swipe": self.swipe_info, "show": self.show_info}
        self.log(**self.last_log)
        return instr

    def where_words(self, front, cfg, codes):
        """The parcel's place on the free table in words for WALLX ("left", "front right", "back middle", ...): its head label, else the object box; None if unseen."""
        h, w = front.shape[:2]
        if codes:
            u, v = codes[0][1]
        elif self.obj_box is not None:
            u, v = (self.obj_box[0] + self.obj_box[2]) / 2, (self.obj_box[1] + self.obj_box[3]) / 2
        else:
            return None
        x0, y0, x1, y1 = cfg["induction_roi"]
        fx, fy = (u / w - x0) / (x1 - x0), (v / h - y0) / (y1 - y0)
        return ("back " if fy < 0.2 else "front " if fy > 0.7 else "") + ("left" if fx < 1 / 3 else "right" if fx > 2 / 3 else "middle")  # perspective: the far band is thin

    def empty_frame(self, front):
        """The configured empty-table JPEG (empty_ref) resized to the frame, or None; read once (self._empty False = not tried yet)."""
        if self._empty is False:
            self._empty = None
            if self.cfg.get("empty_ref"):
                img = cv2.imread(str(self.cfg["empty_ref"]))  # BGR on disk, the frames are RGB
                if img is None:
                    self.log(warning="empty_ref: cannot read %s" % self.cfg["empty_ref"])
                else:
                    self._empty = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), front.shape[1::-1], interpolation=cv2.INTER_AREA)
        return self._empty

    def parcel_hw(self, front, cfg, lc):
        """Half-width from the changed-pixel bbox (FULL-image pixels, as look.footprint expects); lc["hw"] when the box is missing,
        covers nearly the whole zone (stale reference) or is tiny."""
        box = scene.changed_box(front, self.ref, cfg)
        if box is not None and not 0.02 <= (box[2] - box[0]) * (box[3] - box[1]) / scene.roi_area(front, cfg) <= 0.9:
            box = None
        return look.footprint(box, front.shape[1::-1], lc)

    def plan_look(self, front, cfg):
        """Parcel point + footprint from the head-camera mask, then the IK-gated views (or the point_test pose)."""
        assert self.ref is not None
        lc, wh = self.cfg["look"], front.shape[1::-1]
        # the changed blob is mostly top + front face, so intersect the head ray at the aim-point height (table + target_h), not the table
        p = look.pixel_to_table(*scene.centroid(front, self.ref, cfg), wh, look.TABLE_Z + lc["target_h"])
        hw = self.parcel_hw(front, cfg, lc)
        if lc.get("point_test"):
            ee, rpy, err = look.point_pose("left_ee", p + [0, 0, 0.10], lc)  # fingertips 0.10 m over the estimated point
            self.point = ("left_ee", ee, rpy)
            self.look_info = {"point": p.tolist(), "ee": ee.tolist(), "rpy": rpy.tolist(), "ik_err": float(err)}
            return
        p[2] = look.TABLE_Z
        self.views = look.views(p, lc, hw)
        if not self.views:
            self.log(warning="LOOK: no feasible view of the parcel at %s (hw %.3f): the look-around will hold still" % (p.round(3).tolist(), hw))
        self.look_info = {"point": p.tolist(), "hw": hw, "views": [(a, ee.tolist(), i["ik_err"], i["clear"]) for a, ee, _, i in self.views]}

    def fit(self, front, cfg, sc):
        """3D box fit (look.fit_parcel) to the head-camera change mask (None: nothing there / a stale reference / implausible); sets the flat flag."""
        assert self.ref is not None
        mask = scene.changed_mask(front, self.ref, cfg)
        frac = (mask > 0).sum() / scene.roi_area(front, cfg)
        k = max(1, round(front.shape[1] / 640))  # the blurred change mask is ~2 px fatter than the parcel (at 1280 px): undo it, the height fit is sensitive to it
        mask = cv2.erode(mask, np.ones((2 * k + 1, 2 * k + 1), np.uint8))
        fit = look.fit_parcel(mask, front.shape[1::-1], sc) if 0.02 <= frac <= 0.9 else None
        if fit:
            self.sm.flat = fit["h"] < self.cfg["show"]["flat_h"]  # a flat parcel carries a label on its top or bottom only
        return fit

    def on_parcel(self, before, front, cfg, box, pre):
        """Predicate: does a head-image code centre lie on the tracked parcel? SHOW: on the held parcel's projection (a disc around the show point,
        only once it is lifted); SPIN / LOOK: on the changed pixels of the zone (the parcel lies there, dilated); INSPECT / HANDOVER (WALLX holds it,
        untracked): anywhere but a bin; IDLE (nothing tracked yet): inside the induction ROI (a parcel may already lie there, off the reference); else
        (SCAN ...): inside the induction ROI."""
        base = lambda p: not scene.in_bin(p, front, cfg)  # a parcel already in a bin / tray is never read, in any state
        if before == "SHOW":
            if not self.show or pre < 4:
                return lambda p: False
            P, hold = self.show["P"] - [0, 0, 0.02], self.show["hold"]
            r = FOOT * np.linalg.norm(hold[:3]) / 2 + 0.03
            c, d = look.table_to_pixel(P, front.shape[1::-1]), np.linalg.norm(P - look.HEAD_POS)
            rpx = look.FX * r / d * front.shape[1] / 1280
            return lambda p: base(p) and np.hypot(p[0] - c[0], p[1] - c[1]) <= rpx
        if before in ("INSPECT", "HANDOVER", "IDLE") or self.ref is None:
            return lambda p: base(p) and (before != "IDLE" or scene.inside(p, box))
        if self.cfg.get("event_fast") and before in ("SCAN", "CENTER"):  # a parcel present at the start is IN the reference (no changed pixels): the free-table ROI
            return lambda p: base(p) and scene.inside(p, box)
        m = cv2.dilate(scene.changed_mask(front, self.ref, cfg), np.ones((21, 21), np.uint8))  # SCAN / SPIN / LOOK: the parcel lies on the table
        return lambda p: base(p) and 0 <= int(p[1]) < m.shape[0] and 0 <= int(p[0]) < m.shape[1] and m[int(p[1]), int(p[0])] > 0

    def wrist_cams(self, before, pre):
        """Wrist cameras whose codes count: event_fast = the arm nearest the detected object while the head searches (SCAN / CENTER); none in
        real_safe (head only): SHOW = the holding arm's, once it grips (from the close call); LOOK = the looking arm's; INSPECT / HANDOVER = both
        (untracked); SCAN / SPIN: none (the arms are home, their view is not the parcel)."""
        if self.cfg.get("event_fast"):
            return {self.near + "_wrist"} if self.obj_box is not None and before in ("SCAN", "CENTER") else set()
        if self.cfg.get("real_safe"):
            return set()
        if before == "SHOW":
            return {self.show["arm"][:-3] + "_wrist"} if self.show and pre >= 4 else set()
        if before == "LOOK" and self.look_index is not None:
            return {self.views[self.look_index][0][:-3] + "_wrist"}
        return {"left_wrist", "right_wrist"} if before in ("INSPECT", "HANDOVER") else set()

    def grasp_check(self, front, state, pre):
        """After the close call (pre 2): the holding gripper reads below grasp.empty_below = closed on nothing -> "missed". After the 2 cm test lift (pre 3): the
        parcel's silhouette must match the box lifted by test_lift better than the box on the table, else "no_comotion". None = fine."""
        sc = {**self.cfg["grasp"], **self.cfg["show"]}
        if pre == 2 and state[self.show["arm"]]["gripper"] < sc["empty_below"]:
            return "missed"
        if pre == 3:
            f, m = self.show["fit"], scene.changed_mask(front, self.ref, self.cfg["scene"]) > 0
            ious = [(look.box_mask(f["c"], f["yaw"], *f["half"], f["h"], front.shape[1::-1], dz=dz) > 0) for dz in (0.0, sc["test_lift"])]
            score = [(s & m).sum() / max((s | m).sum(), 1) for s in ious]
            return None if score[1] > score[0] + 0.02 else "no_comotion"

    def homes(self):
        return {a: (np.asarray(self.home[a]["position"], float), np.asarray(self.home[a]["rotation_rpy"], float)) for a in look.SIGN}

    def prompt_boxes(self, shape):
        h, w = shape[:2]
        parcel = self.obj_box if self.obj_box is not None else (
            (max(int(round(self.code_px[0] - 70)), 0), max(int(round(self.code_px[1] - 50)), 0),
             min(int(round(self.code_px[0] + 70)), w - 1), min(int(round(self.code_px[1] + 50)), h - 1))
            if self.code_px is not None else None
        )
        lanes = self.cfg.get("lanes", {})
        lane = self.sm.lane
        target = scene._frac_box(np.empty(shape, dtype=np.uint8), self.cfg["scene"]["bin_rois"].get(lanes.get(lane, "black"))) if lanes.get(lane, "black") in self.cfg["scene"]["bin_rois"] else None
        bins = {k: scene._frac_box(np.empty(shape, dtype=np.uint8), v) for k, v in self.cfg["scene"]["bin_rois"].items()}
        roi = scene.roi_box(np.empty(shape, dtype=np.uint8), self.cfg["scene"])
        return parcel, target, bins, roi

    def plan_slide(self, front, cfg):
        """Delivery arm = the lane's arm. Fit the parcel; the delivery arm can grasp it -> nothing to do; too wide for the gripper -> reroute
        (LIFT2 / reject); else the OTHER arm's face-normal stroke toward the delivery arm's hand-off point (None if no stroke is feasible:
        the delivery arm then tries anyway)."""
        sc = {**self.cfg["look"], **self.cfg["spin"], **self.cfg["grasp"], **self.cfg["slide"]}
        side = self.sm._arm()
        arm, fit, home, self.slide = side + "_ee", None, self.homes(), None
        fit = self.fit(front, cfg, sc)
        info = {"stroke": self.sm.tries["SLIDE"], "delivery": arm, "target": sc["handoff"][side],
                "fit": fit and {"c": fit["c"].tolist(), "half": fit["half"].tolist(), "yaw": fit["yaw"]}}
        if fit is None:
            info["why"] = "no fit"
        elif look.grasp_plan(arm, fit, sc, home):
            info["why"] = "graspable"
        elif look.too_wide(fit, sc):
            info["why"] = "too_wide"
            self.slide_info = info
            if sc["reroute_wide"]:
                self.sm.reroute_wide()
            return
        else:
            self.slide = look.slide_plan(fit, sc["handoff"][side], arm, sc, home)
            info.update(why="stroke" if self.slide else "no stroke", arm=self.slide and self.slide["arm"],
                        tips=self.slide and self.slide["tips"], c_after=self.slide and self.slide["c_after"])
        self.slide_info = info

    def plan_swipe(self, front, cfg):
        """Fit the parcel and plan one reject stroke along -y toward the table's right edge (look.swipe_plan; None = no feasible stroke)."""
        sc = {**self.cfg["look"], **self.cfg["spin"], "dir_min_dot": self.cfg["slide"]["dir_min_dot"], **self.cfg["swipe"]}
        fit = self.fit(front, cfg, sc)
        self.swipe = fit and look.swipe_plan(fit, sc, self.homes())
        self.swipe_info = {"stroke": self.sm.tries["SWIPE"], "fit": fit and {"c": fit["c"].tolist(), "half": fit["half"].tolist(), "yaw": fit["yaw"], "h": fit["h"]},
                           "arm": self.swipe and self.swipe["arm"], "dist": self.swipe and self.swipe["dist"], "off": self.swipe and self.swipe["off"],
                           "why": "stroke" if self.swipe else "no fit" if fit is None else "no stroke"}

    def plan_show(self, front, cfg):
        """Fit the parcel; the arm(s) that can grasp it plan a SHOW (look.show_plan); the one that shows the most faces wins (ties: the nearer to
        the parcel's side). No plan (no fit, too wide for the gripper, nothing reachable) -> self.show None."""
        sc = {**self.cfg["look"], **self.cfg["spin"], **self.cfg["grasp"], **self.cfg["slide"], **self.cfg["show"]}
        fit, home, self.show, self.sset = self.fit(front, cfg, sc), self.homes(), None, None
        plans = [] if fit is None else [look.show_plan(a, fit, sc, home) for a in sorted(look.SIGN, key=lambda a: abs(fit["c"][1] - look.SIGN[a] * sc["arm_y"]))]
        plans = [p for p in plans if p]
        self.show = max(plans, key=lambda p: len(p["faces"]), default=None)
        self.show_info = {"fit": fit and {"c": fit["c"].tolist(), "half": fit["half"].tolist(), "yaw": fit["yaw"]},
                          "arm": self.show and self.show["arm"], "P": self.show and self.show["P"], "faces": self.show and self.show["faces"],
                          "missing": self.show and self.show["missing"], "n_calls": self.show and len(self.show["calls"]),
                          "why": None if self.show else "no fit" if fit is None else "too_wide" if look.too_wide(fit, sc) else "no plan"}

    def plan_showset(self):
        """The set-down after SHOW (`show_done` calls driven): inside the grasp region of the arm that will deliver (the code just read, else the reject arm)."""
        sc = {**self.cfg["look"], **self.cfg["spin"], **self.cfg["grasp"], **self.cfg["slide"], **self.cfg["show"]}
        deliv = self.sm.delivery_arm(self.sm.pend or ())
        self.sset = look.set_down(self.show, deliv, max(self.show_done, 1), sc, self.homes())
        self.show_info = {**self.show_info, "done": self.show_done, "delivery": deliv, "set_down": self.sset["where"], "spot": self.sset["spot"], "yaw": self.sset["yaw"]}

    def plan_spin(self, front, cfg):
        """Fit the parcel's oriented footprint to the head-camera change mask and plan the face-normal push; no plan (the try is skipped)
        when the fit is implausible or no face has a feasible full path. Later tries keep the first plan's yaw direction."""
        sc = {**self.cfg["look"], **self.cfg["spin"]}
        fit, home = self.fit(front, cfg, sc), self.homes()
        self.spin = fit and look.spin_plan(fit, sc, self.spin_sigma, home)
        self.spin_sigma = self.spin["sigma"] if self.spin else self.spin_sigma
        self.spin_info = {"try": self.sm.tries["SPIN"], "fit": fit and {"c": fit["c"].tolist(), "half": fit["half"].tolist(), "yaw": fit["yaw"]},
                          "arm": self.spin and self.spin["arm"], "pitch": self.spin and self.spin["pitch"], "tips": self.spin and self.spin["tips"],
                          "contact": self.spin and self.spin["contact"], "normal": self.spin and self.spin["normal"],
                          "tip_z": self.spin and self.spin["tip_z"], "sigma": self.spin_sigma}

    def plan_calib(self, wh):
        """Schedule of calls: per arm, per table point (refused if no pitch of calib.pitches_deg, i.e. pointing (nearly) down, is reachable): move_calls + hold_calls, then home.
        Logs each target's expected head pixels now; the measured pose is logged at the last hold call."""
        lc, cc, sched = self.cfg["look"], self.cfg["calib"], []
        for arm in ("left_ee", "right_ee"):
            for x, y in cc["points"]:
                tip = np.array([x, y, look.TABLE_Z + cc["tip_h"]])
                r = next(filter(None, (look.reach(arm, tip, np.radians(d), lc) for d in cc["pitches_deg"])), None)  # steepest feasible, else refuse
                info = {"arm": arm, "tip_root": tip, "tip_px": look.table_to_pixel(tip, wh),
                        "floor_px": look.table_to_pixel([x, y, look.TABLE_Z], wh), "refused": r is None}
                if r is None:
                    self.log(calib=info)
                    continue
                info.update(ee=r[0], rpy=r[1], ik_err=r[3], pitch_deg=float(np.degrees(r[1][1])))
                sched += [{"arm": arm, "ee": r[0], "rpy": r[1]}] * cc["move_calls"]
                sched += [{"arm": arm, "ee": r[0], "rpy": r[1]}] * (cc["hold_calls"] - 1) + [{"arm": arm, "ee": r[0], "rpy": r[1], "info": info}]
            sched += [{"home": True}] * self.cfg["home_calls"]
        return sched

    def calib_step(self, state):
        e = self.cal_cur = self.calib[self.cal_n] if self.cal_n < len(self.calib) else {"hold": True}  # latched: hold still
        self.cal_n += 1
        self.state_name = "CALIB"
        if "info" in e:  # last hold: where the arm actually is vs where the head image should show the tip
            ee, rpy = state[e["arm"]]["position"], state[e["arm"]]["rotation_rpy"]
            self.log(calib={**e["info"], "measured_ee": ee, "measured_rpy": rpy,
                            "measured_tip": look.tip_of(e["arm"], ee, rpy, self.cfg["look"])})
        self.log(call=self.calls, state="CALIB", instruction=None, calib_call=self.cal_n)
        return None

    def log(self, **kw):
        if self.runlog:
            self.runlog.write(**kw)

    def look_span(self):
        lc = self.cfg.get("look", {})
        return lc.get("via_calls", 0) + lc.get("move_calls", 0) + lc.get("dwell_calls", 0)

    @property
    def spin_index(self):
        """Index of the push waypoint being driven during SPIN, else None."""
        return self.sm.calls if self.sm.state == "SPIN" and self.spin and self.sm.calls < len(self.spin["wps"]) else None

    @property
    def look_index(self):
        """Index of the viewpoint being driven during LOOK, else None."""
        if self.sm.state != "LOOK" or not self.views:
            return None
        return min(self.sm.calls // self.look_span(), len(self.views) - 1)

    def scripted_reply(self, embodiment, state):
        """WALLX-shaped reply for a scripted step (home, look-around), or None when WALLX should answer."""
        if self.cal_cur:  # calib_test: one arm over the table point, the other home, never WALLX
            e = self.cal_cur
            if e.get("hold"):
                return move_response(embodiment, state, {})
            if e.get("home"):
                return home_response(embodiment, state, self.home)
            other = "right_ee" if e["arm"] == "left_ee" else "left_ee"
            return move_response(embodiment, state, {e["arm"]: {"position": e["ee"].tolist(), "rotation_rpy": e["rpy"].tolist()}, other: self.home[other]})
        if self.sm.state in ("SHOW", "SHOWSET"):  # grasp-and-show / set-down waypoints (gripper values included), then home; the other arm stays home
            sp, i = (self.show, self.sm.calls) if self.sm.state == "SHOW" else (self.sset, self.sm.calls)
            n = sp and len(sp["calls"])
            if sp and i < n:
                c = sp["calls"][i]
                arm = self.show["arm"]
                other = "right_ee" if arm == "left_ee" else "left_ee"
                return move_response(embodiment, state, {arm: {"position": c["ee"].tolist(), "rotation_rpy": c["rpy"].tolist(), "gripper": c["grip"]},
                                                         other: self.home[other]})
            if self.sm.state == "SHOWSET" and (not sp or i < n + self.cfg["home_calls"]):
                return home_response(embodiment, state, self.home)  # (the gripper keeps its last value until home)
            return move_response(embodiment, state, {}) if self.sm.state == "SHOW" else None
        if self.sm.state in ("SPIN", "SLIDE", "SWIPE"):  # push waypoints (gripper closed), then home; the read calls after that hold still
            sp, i = {"SPIN": self.spin, "SLIDE": self.slide, "SWIPE": self.swipe}[self.sm.state], self.sm.calls
            n = sp and len(sp["wps"])
            if sp and i < n:
                ee, rpy = sp["wps"][i]
                other = "right_ee" if sp["arm"] == "left_ee" else "left_ee"
                return move_response(embodiment, state, {sp["arm"]: {"position": ee.tolist(), "rotation_rpy": rpy.tolist(),
                                                                     "gripper": self.cfg["spin"]["gripper_closed"]}, other: self.home[other]})
            if sp and i < n + self.cfg["home_calls"]:
                return home_response(embodiment, state, self.home)
            return None
        if self.sm.state == "LOOKHOME" or (self.cfg.get("scripted_home") and self.sm.state == "HOME"):
            return home_response(embodiment, state, self.home)
        if self.sm.state == "LOOK" and self.point:  # point_test: left fingertips over the parcel, right arm home, no WALLX
            arm, ee, rpy = self.point
            return move_response(embodiment, state, {arm: {"position": ee.tolist(), "rotation_rpy": rpy.tolist()}, "right_ee": self.home["right_ee"]})
        if self.look_index is not None:
            lc = self.cfg["look"]
            arm, pos, rpy, _ = self.views[self.look_index]
            other = "right_ee" if arm == "left_ee" else "left_ee"
            t = {other: self.home[other]}  # the idle arm goes home while this one looks
            if self.sm.calls % self.look_span() < lc["via_calls"]:  # lift straight up first so the sweep clears the parcel
                p = list(state[arm]["position"])
                t[arm] = {"position": [p[0], p[1], look.to_arm(arm, [0, 0, look.TABLE_Z + 0.30], lc)[2]]}  # state is in the arm frame, so is the target
            else:
                t[arm] = {"position": pos.tolist(), "rotation_rpy": rpy.tolist()}
            return move_response(embodiment, state, t)
        if self.sm.state == "LOOK":  # every viewpoint unreachable: just hold
            return move_response(embodiment, state, {})

    def log_reply(self, latency_s, error=None):
        self.log(call=self.calls, latency_s=latency_s, error=error)
