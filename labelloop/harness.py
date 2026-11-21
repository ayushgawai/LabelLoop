import copy
import time

import cv2
import numpy as np

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
        self.views, self.point, self.look_info = [], None, None  # LOOK viewpoints / point_test pose, built at SCAN -> LOOK
        self.confirmer = Confirmer(self.cfg["qr_confirm_frames"])  # items are (camera, text): same text in N consecutive frames of ONE camera
        self.spin, self.spin_info, self.spin_sigma = None, None, None  # the planned push of the current SPIN try; the yaw direction all tries keep
        self.show, self.show_info, self.sset, self.show_done = None, None, None, 0  # SHOW plan; its log; the SHOWSET (set-down) plan; SHOW calls driven
        self.slide, self.slide_info = None, None  # the planned stroke of the current SLIDE entry (None = nothing to push)
        self.calib, self.cal_n, self.cal_cur = None, 0, None  # calib_test schedule, calls consumed, entry of the current call
        self.ref, self.empty, self.bin_refs = None, 0, {}  # empty-zone reference frame; bin snapshots
        self.calls, self.subgoal_changed, self.done, self.state_name = 0, False, False, self.sm.state

    def step(self, images, state):
        if self.home is None:
            self.home = copy.deepcopy(state)
        self.calls += 1
        if self.cfg.get("calib_test") and self.calib is None:
            self.calib = self.plan_calib(images["front"].shape[1::-1])
        if self.calib is not None:  # calibration latches: the schedule, then hold still for the rest of the episode (never sorts)
            return self.calib_step(state)
        self.cal_cur = None
        before, cfg = self.sm.state, self.cfg["scene"]
        front = images["front"]
        box = scene.roi_box(front, cfg)
        if before in ("LOOK", "INSPECT", "HANDOVER", "SHOW"):  # parcel in hand: a label counts anywhere but in a bin (a labelled parcel lying in one)
            bins = [scene.roi_box(front, cfg, "bin_rois", c) for c in cfg["bin_rois"]]
            codes = [(t, p) for t, p in locate(front) if not any(scene.inside(p, b) for b in bins)]
        else:
            codes = [(t, p) for t, p in locate(front) if scene.inside(p, box)]
        if not codes and before not in ("LOOK", "INSPECT", "HANDOVER", "SHOW"):  # small QR missed on this frame: retry on the ROI crop at 2x
            crop = cv2.resize(front[box[1]:box[3], box[0]:box[2]], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            codes = [(t, (box[0] + x / 2, box[1] + y / 2)) for t, (x, y) in locate(crop)]
        wrist = {(v, t) for t, v in decode({k: v for k, v in images.items() if k != "front"})} \
            if before in ("SHOW", "SPIN", "LOOK", "INSPECT", "HANDOVER") else set()  # SCAN counts head ROI codes only
        seen = {("front", t) for t, _ in codes} | wrist
        confirmed = {t for _, t in self.confirmer.update(seen)}
        pres = 0.0 if self.ref is None else scene.presence(front, self.ref, cfg)
        if before == "IDLE":  # arms are home: keep the reference current whenever the zone has been empty a while
            self.empty = self.empty + 1 if pres < cfg["presence_min"] else 0
            if self.ref is None or self.sm.rebase or self.empty >= cfg["ref_refresh_calls"]:
                self.ref, self.empty, self.sm.rebase = scene.zone_ref(front, cfg), 0, False
        kroi = scene.kraft_in_roi(front, cfg)
        bins = {c: scene.presence(front, r, cfg, "bin_rois", c) for c, r in self.bin_refs.items()}  # {} until the first SCAN
        n = 10**9 if self.point else len(self.views) * self.look_span()  # point_test: hold until the episode is stopped
        push = len(self.spin["wps"]) + self.cfg["home_calls"] if self.spin else 0
        sc = self.cfg.get("spin", {}).get("scan_calls", 0)
        pre = self.sm.calls
        instr = self.sm.step({"presence": pres, "kroi": kroi, "codes": confirmed, "bins": bins, "look_calls": n, "raw": {t for _, t in seen},
                              "show_calls": len(self.show["calls"]) if self.show else 0,
                              "showset_calls": len(self.sset["calls"]) + self.cfg["home_calls"] if self.sset else self.cfg["home_calls"],
                              "spin_push": push, "spin_calls": push + sc if self.spin else 1,
                              "slide_calls": len(self.slide["wps"]) + self.cfg["home_calls"] + self.cfg["slide"]["settle_calls"] if self.slide else 0})  # no feasible push: the try is skipped
        if before == "SCAN" and self.sm.state == "LOOK":  # parcel point from the changed pixels of the zone, once
            self.plan_look(front, cfg)
        if self.sm.state == "SHOW" and self.sm.calls == 0:  # grasp-and-show: nothing to grasp -> the next search step at once
            self.plan_show(front, cfg)
            if not self.show:
                self.sm.end_show()
                instr = self.sm.instruction()
        if before == "SHOW" and self.sm.state == "SHOWSET":  # the parcel is in hand (or not yet gripped): put it back where the delivery arm can take it
            self.show_done = pre + 1
            self.plan_showset()
        if self.sm.state == "SPIN" and self.sm.calls == 0:  # a new SPIN try (arm home, parcel as it lies now)
            self.plan_spin(front, cfg)
        if self.sm.state == "SLIDE" and self.sm.calls == 0:  # a new stroke (arm home, parcel as it lies now): nothing to push -> the delivery arm acts
            self.plan_slide(front, cfg)
            if self.sm.state == "SLIDE" and not self.slide:
                self.sm.end_slide()
            instr = self.sm.instruction()
        if before == "IDLE":  # a new parcel starts with no QR history and no preferred yaw direction
            self.confirmer.counts.clear()
            self.spin_sigma = None
        if before == "IDLE" and self.sm.state == "SCAN":  # object arrived, arms home: snapshot every bin
            self.bin_refs = {c: scene.zone_ref(front, cfg, "bin_rois", c) for c in cfg["bin_rois"]}
        self.state_name, self.done = self.sm.state, self.sm.state == "DONE"
        self.subgoal_changed = self.sm.state != before
        if self.runlog:
            self.runlog.write(call=self.calls, state=before, instruction=instr, codes=codes, wrist_codes=wrist,
                              confirmed=confirmed, lane=self.sm.lane, presence=pres, bin_change=bins, kraft_roi=kroi,
                              left_gripper=state["left_ee"]["gripper"],
                              right_gripper=state["right_ee"]["gripper"], transitions=self.sm.transitions,
                              flags=self.sm.flags, parcels=self.sm.parcels, look_view=self.look_index, look=self.look_info,
                              spent_s=self.sm.spent, spin=self.spin_info, slide=self.slide_info, show=self.show_info)
        return instr

    def parcel_hw(self, front, cfg, lc):
        """Half-width from the changed-pixel bbox (FULL-image pixels, as look.footprint expects); lc["hw"] when the box is missing,
        covers nearly the whole zone (stale reference) or is tiny."""
        box, (x0, y0, x1, y1) = scene.changed_box(front, self.ref, cfg), scene.roi_box(front, cfg)
        if box is not None and not 0.02 <= (box[2] - box[0]) * (box[3] - box[1]) / ((x1 - x0) * (y1 - y0)) <= 0.9:
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
        """Oriented parcel footprint from the head-camera change mask (None: nothing there / a stale reference / implausible)."""
        assert self.ref is not None
        mask = scene.changed_mask(front, self.ref, cfg)
        x0, y0, x1, y1 = scene.roi_box(front, cfg)
        frac = (mask > 0).sum() / ((x1 - x0) * (y1 - y0))
        return look.fit_parcel(mask, front.shape[1::-1], sc) if 0.02 <= frac <= 0.9 else None

    def homes(self):
        return {a: (np.asarray(self.home[a]["position"], float), np.asarray(self.home[a]["rotation_rpy"], float)) for a in look.SIGN}

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
        if self.sm.state in ("SPIN", "SLIDE"):  # push waypoints (gripper closed), then home; the read calls after that hold still
            sp, i = (self.spin if self.sm.state == "SPIN" else self.slide), self.sm.calls
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
