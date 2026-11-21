import re
import time

from labelloop.brain.instructions import instruction

# IDLE -> SCAN -> SHOW -> SPIN -> (LOOK -> LOOKHOME) -> INSPECT -> route   (label search; each step is skipped when disabled / tries 0)
# IDLE -> SCAN also on a valid parcel-format code seen by the head camera inside the induction ROI (outside the bins), even with NO presence
#   change: the parcel lay on the table when the episode began, so it is part of the empty reference (trial 1).
# SCAN: head ROI codes; none after scan_max_calls -> SHOW.  SHOW (show.enabled, not big, once per parcel): scripted (no WALLX): one arm grasps the
#   parcel, lifts it in front of the head camera and turns each face toward it (all cameras decode); the first confirmed code (or the last face) ->
#   SHOWSET: the arm sets it down inside the delivery arm's grasp region (the delivery arm is known from the code, else the reject arm) and goes
#   home; then the code is read (route) or the search goes on. No plan (parcel too wide for the gripper, nothing reachable) -> SPIN at once.
#   SPIN: scripted (no WALLX): one arm pushes the parcel so it yaws, goes home,
#   then the head camera re-reads (spin.tries tries). LOOK (look.enabled): scripted wrist-camera look-around. INSPECT: WALLX flips it.
# event_fast (real profile): an object detected on the table (ev["obj"], no readable label) is CENTERed first - WALLX "slide the box to the middle
#   of the table with the {arm} arm" (nearest arm, max_calls 8, tries 2) - then re-SCANned, then the flip path (INSPECT), then REJECT (black bin).
# Nothing found after the last step -> reject with "no_label" only if every face that can carry a label was observed (flat parcel: top + bottom; box: all
#   6; the top is seen at SCAN, SHOW adds the faces it held to the head, an INSPECT try the bottom), else "unread" (also when the budget's deliver reserve is hit
#   first). There is no colour shortcut: every object gets the full search. A big parcel searches too (SHOW refuses it when too wide, then SPIN pushes it).
# route: PLACE (lane arm: A left, B right) | REJECT (right arm) | LIFT2 (big, both arms); with handover: true a HANDOVER precedes
#   PLACE / REJECT. A code routes immediately, even mid-INSPECT. A bad code -> REJECT.
# PLACE/REJECT/LIFT2 -> HOME -> VERIFY -> IDLE (parcel counted) | retry the plan | IDLE after the last try (failure logged)
# Budget: estimated REAL seconds per CASE (every call: WALLX ~wallx_call_s, scripted / hold / IDLE ~scripted_call_s); a search try or WALLX step
#   needs its cost + deliver_reserve + home to fit, else route (budget).
# SLIDE (slide.enabled, not big, no handover): scripted (no WALLX) hand-off before PLACE / REJECT: the harness checks whether the delivery arm can
#   grasp the parcel where it lies; if not, the other arm pushes it toward the delivery arm's hand-off point (a stroke per SLIDE entry, up to
#   slide.max_strokes, each re-fitted by the head camera); then the delivery arm places as usual (also when the slide failed).
# SWIPE (reject_mode floor_right, the default; a reject has no bin): scripted (no WALLX) strokes that push the parcel along -y off the table's right
#   edge, each entry re-fitting it (swipe.max_strokes); HOME -> VERIFY (the zone is empty) as for the other actions. reject_mode tray: REJECT puts it in the black bin.
# PUSH stays an action state but no plan routes to it. DONE only when max_parcels is set.
ACTIONS = ("PUSH", "PLACE", "REJECT", "LIFT2")  # success-checked
COUNTED = ACTIONS + ("INSPECT", "SPIN", "HANDOVER", "SLIDE", "SHOW", "SWIPE", "CENTER")  # tries counted per parcel
WALLX = ACTIONS + ("INSPECT", "HANDOVER", "CENTER")  # states that send an instruction to WALLX (HOME too unless scripted_home)


SCRIPTED = ("SWIPE", "SLIDE", "SHOW", "SHOWSET", "SPIN", "LOOK")


class StateMachine:
    """step(ev) -> instruction | None (None = hold still).
    ev = {"presence": zone changed fraction, "kroi": kraft fraction of the zone, "codes": confirmed code texts,
          "bins": {bin colour incl. black: ROI changed fraction vs the snapshot at SCAN}, optional "big": bool override of the presence test,
          "raw": decoded texts this call, confirmed or not, "spin_push" / "spin_calls": calls of the scripted push (codes are ignored until it is done) / of the whole spin try},
          event_fast adds "present" (the zone was not empty on the very first frame: a parcel was already there), "obj" (a plausible
          object is on the table, label readable or not) and "near" ("left" / "right": the arm nearest it, for CENTER)."""

    def __init__(self, cfg, clock=time.monotonic):
        self.cfg, self.pattern, self.clock = cfg, re.compile(cfg["code_pattern"]), clock
        self.lane_of = {z: lane for lane, zips in cfg["sort_plan"].items() for z in zips}
        self.reset()

    def reset(self):
        self.state, self.calls, self.parcels, self.results, self.spent = "IDLE", 0, 0, [], 0.0
        self.t0 = None  # event_fast: case start on the clock
        self.transitions, self.flags, self.rebase, self.sorted = [], [], False, set()  # sorted: codes done this run
        self._fresh()

    def _fresh(self):
        self.code = self.lane = self.reason = self.plan = self.pend = None
        self.big, self.obj, self.tries, self.ok, self.seen, self.prev = False, "block", {}, 0, 0, 0.0
        self.binhit = 0  # event_fast: consecutive calls with a non-target bin moved
        self.where = None  # event_fast: "front left" ... (harness where_words), for {where}
        self.near = None  # event_fast: the arm nearest the detected object ("left" / "right"), for CENTER's wording
        self.landed = False  # event_fast: the action's success check held
        self.looked = self.showed = False  # LOOK / SHOW done
        self.faces, self.flat = {"top"}, self.cfg.get("show", {}).get("flat_default", True)  # label faces observed (top: seen at SCAN); flat parcel = only top / bottom carry a label

    def _go(self, state, why):
        fast = self.cfg.get("event_fast")  # event_fast: WALLX does lifts / rejects too, only scripted states stay latched
        if self.cfg.get("real_safe") and (state in SCRIPTED + ("LOOKHOME",) or not fast and (state in ("LIFT2", "HANDOVER") or state == "REJECT" and why not in ("unknown_zip", "bad_code"))):  # a read-but-unroutable code goes to WALLX REJECT (black bin)  # interlock: never dispatched, latch HOLD for the operator
            state, why = "HOLD", "operator_hold: %s (%s)" % (why, state)
        if state in WALLX and self.remaining() < self.cfg["budget"]["wallx_call_s"]:  # not even one more WALLX call fits
            state, why = "HOME", "budget"
        self.transitions.append((self.state, state, why))
        self.state, self.calls, self.ok, self.gp = state, 0, 0, 0
        if state in COUNTED:
            self.tries[state] = self.tries.get(state, 0) + 1

    def instruction(self):
        s = self.state
        if s in WALLX + ("HOME",):
            key = "INSPECT_BIG" if s == "INSPECT" and self.big else s
            arm = self.near if s == "CENTER" and self.near else self._arm()  # CENTER: the arm nearest the object
            return instruction(key, self.cfg, self.tries.get("INSPECT", 1) - 1 if s == "INSPECT" else 0,
                               mat=self.cfg["lanes"].get(self.lane, "black"), obj=self.obj, arm=arm, where=self.where or "middle")

    def _arm(self):  # the arm that puts the parcel in its bin: the lane's (handover: false) or the receiving right arm
        return "right" if self.cfg.get("handover") else self.cfg["arms"].get(self.lane or "reject", "right")

    def _cost(self, s):  # estimated real seconds of one call in state s
        b = self.cfg["budget"]
        return b["wallx_call_s"] if s in WALLX or (s == "HOME" and not self.cfg.get("scripted_home")) else b["scripted_call_s"]

    def begin(self):
        self.t0 = self.clock() if self.t0 is None else self.t0

    def remaining(self):
        b = self.cfg["budget"]
        if self.cfg.get("event_fast"):  # real elapsed time on the harness clock
            self.begin()
            return b["case_s"] - (self.clock() - self.t0)
        return min(b["case_s"], b.get("sim_case_s") or b["case_s"]) - self.spent

    def _admit(self, s):  # a search try fits: its estimated cost + the deliver reserve + the way home remain
        b, c = self.cfg["budget"], self.cfg
        cost = c["max_calls"].get(s, 1) * b["wallx_call_s"] if s in WALLX else c[s.lower()]["est_calls"] * b["scripted_call_s"]
        return self.remaining() - cost - c["home_calls"] * b["scripted_call_s"] >= b["deliver_reserve_s"]

    def _bin_moved(self, ev, level=0.8, calls=3):
        """A bin region (the target one only while the parcel has not landed) changed >= level for `calls` calls in a row: the bin itself moved (real agent trial: WALLX lifted the
        black bin into the red one: 0.93+, persistent; an arm hovering over a bin reads 0.4-0.6)."""
        target = self.cfg["lanes"].get(self.lane, "black")  # the target bin too while the drop has not succeeded (agent run 3: WALLX tipped the red TARGET bin)
        hit = any(v >= level for k, v in ev.get("bins", {}).items() if k != target or not self._success(ev))
        self.binhit = self.binhit + 1 if hit else 0
        return self.binhit >= calls

    def _center(self):  # event_fast: the object is there but its label cannot be read -> WALLX slides it into the open middle for a fresh look
        c = self.cfg
        if not (c.get("event_fast") and self.tries.get("CENTER", 0) < c["tries"].get("CENTER", 0) and self._admit("CENTER")):
            return False
        self._go("CENTER", "object, no code")
        return True

    def _bad_grasp(self, ev):
        """event_fast: abort the attempt at once when the acting gripper (the lane arm; ponytail: left for LIFT2) closes on nothing (below grasp.empty_below),
        or reopens (after closing on the parcel) while the parcel is still in its table spot. Phases gp: 0 wait open, 1 open, 2 holding."""
        if not self.cfg.get("grasp", {}).get("check", True):  # real robot: gripper units unverified (rests at ~0) -> off until calibrated
            return False
        g = ev.get("grip", {}).get("left" if self.state == "LIFT2" else self._arm())
        if g is None or ev["presence"] < self.cfg["scene"]["presence_min"]:
            return False
        lo, op = self.cfg["grasp"]["empty_below"], self.cfg["show"]["grip_open"] * 0.9
        if self.gp == 0:
            self.gp = int(g >= op)
        elif g < lo:
            self.flags.append("grasp:missed")
            return True
        elif self.gp == 1:
            self.gp = 1 if g >= op else 2
        elif g >= op:
            self.flags.append("grasp:dropped")
            return True
        return False

    def _next_action(self):  # retry the plan only while EVERY step of it has tries left (HANDOVER too) and time for a delivery remains
        ok = all(self.tries.get(a, 0) < self.cfg["tries"].get(a, 1 << 30) for a in self.plan)
        return self.plan[0] if ok and self.remaining() >= self.cfg["budget"]["deliver_reserve_s"] else None

    def _search_next(self):  # the current label-search step is exhausted: the next enabled one if it is affordable, else reject
        c, sp, lk = self.cfg, self.cfg.get("spin", {}), self.cfg.get("look", {})
        if self.state == "INSPECT":
            self.faces |= {"bottom"}  # WALLX turned it over (not verifiable)
        nxt = ("SHOW" if c.get("show", {}).get("enabled") and not self.showed else
               "SPIN" if sp.get("enabled") and self.tries.get("SPIN", 0) < sp["tries"] else
               "LOOK" if lk.get("enabled") and not self.looked else
               "INSPECT" if self.tries.get("INSPECT", 0) < c["tries"].get("INSPECT", 0) else None)
        if nxt and not self._admit(nxt):
            self._route(None, "budget" if self.covered() else "unread")
        elif nxt:
            self.looked |= nxt == "LOOK"
            self.showed |= nxt == "SHOW"
            self._go(nxt, "no code")
        else:
            self._route(None, "no_label" if self.covered() else "unread")

    def covered(self):  # every face that can carry a label was observed
        return ({"top", "bottom"} if self.flat else {"top", "bottom", "y+", "y-", "z+", "z-"}) <= self.faces

    def abort_show(self, retry):  # the grasp failed: set the parcel down (SHOWSET), then SHOW again (a fresh fit) if retry, else the next search step
        self.showed = not retry
        self._go("SHOWSET", "grasp failed")

    def _success(self, ev):  # the zone is empty and the parcel's bin changed (it landed there, not just left the zone)
        sc = self.cfg["scene"]
        if not self.lane and self.cfg.get("reject_mode", "tray") == "floor_right":
            return ev["presence"] < sc["presence_min"]  # swiped off the table: the zone is empty
        bin_ = self.cfg["lanes"][self.lane] if self.lane else "black"
        return ev["presence"] < sc["presence_min"] and ev["bins"][bin_] >= sc["bin_gain"]

    def _route(self, lane, reason, code=None, obj="block", slide=True):
        self.lane, self.code, self.reason, self.obj = lane, code, reason, obj
        if reason == "unread":
            self.flags.append("unread:faces=%s" % sorted(self.faces))
        floor = not lane and self.cfg.get("reject_mode", "tray") == "floor_right"
        last = "LIFT2" if self.big and not floor else "PLACE" if lane else "SWIPE" if floor else "REJECT"
        self.plan = [last] if floor or self.big or not self.cfg.get("handover") else ["HANDOVER", last]
        hand = slide and not floor and self.cfg.get("slide", {}).get("enabled") and not self.big and not self.cfg.get("handover") \
            and reason != "budget" and self._admit("SLIDE")
        self._go("SLIDE" if hand else self.plan[0], reason)

    def end_swipe(self, why="no swipe"):  # no stroke is feasible: give up this try
        self._go("HOME", why)

    def end_slide(self, why="slide done"):  # no (further) stroke: the delivery arm acts
        self._go(self.plan[0], why)

    def reroute_wide(self):  # the narrow side does not fit the gripper: both arms if allowed, else reject
        if self.cfg["tries"].get("LIFT2", 0) > 0:
            self.big, self.plan, self.reason = True, ["LIFT2"], "too_wide"
            self._go("LIFT2", "too_wide")
        else:
            self._route(None, "too_wide", self.code, self.obj, slide=False)

    def _parse(self, codes):  # (codes with the parcel-format ones first, their pattern match, the lane of the first)
        codes = sorted(codes, key=lambda c: not self.pattern.match(c))
        m = codes and self.pattern.match(codes[0])
        return codes, m, m and self.lane_of.get(m.group(1)[:3])

    def delivery_arm(self, codes=()):  # "left_ee" / "right_ee": the arm that will place the parcel once `codes` are read (none: the reject bin)
        lane = self._parse(codes)[2]
        return ("right" if self.cfg.get("handover") else self.cfg["arms"].get(lane or "reject", "right")) + "_ee"

    def end_show(self):  # nothing to grasp / show: the next search step
        self._search_next()

    def _read(self, codes):
        codes, m, lane = self._parse(codes)
        self._route(lane, "routed" if lane else "unknown_zip" if m else "bad_code", codes[0])

    def _done(self, outcome):
        if outcome == "ok" and self.code:
            self.sorted.add(self.code)  # only a parcel verified in a bin blinds the same text (a failed delivery must not)
        self.results.append({"code": self.code, "lane": self.lane, "reason": self.reason, "outcome": outcome})
        self.parcels += 1
        self._fresh()
        stop = self.cfg.get("max_parcels") and self.parcels >= self.cfg["max_parcels"]
        self._go("DONE" if stop else "IDLE", outcome)

    def step(self, ev):
        cfg, sc, s = self.cfg, self.cfg["scene"], self.state
        self.transitions, self.flags = [], []
        self.near, self.calls = ev.get("near"), self.calls + 1
        self.where = ev.get("where") or self.where  # event_fast: last known parcel place in words, for {where}
        self.spent += self._cost(s)
        over = self.remaining() < cfg["budget"]["wallx_call_s"]  # one more WALLX call would exceed the case budget
        f, codes = ev["presence"], sorted(c for c in ev["codes"] if c not in self.sorted)
        if s == "IDLE":
            if ev.get("present") or any(self.pattern.match(c) for c in codes):  # parcel already on the table at the start / its label is visible now
                self._go("SCAN", "object present" if ev.get("present") else "code visible")
            else:
                if f < sc["presence_min"] or not ev.get("obj", True):  # event_fast: a change vs the reference with no object in the image is stale
                    self.seen = 0
                else:  # present: stable only while the fraction stops moving (a hand sliding it in keeps changing it)
                    self.seen = self.seen + 1 if self.seen and abs(f - self.prev) <= cfg["settle_tol"] else 1
                if self.seen >= cfg["stable_calls"]:
                    self._go("SCAN", "object in zone")
            self.prev = f
        elif s == "SCAN":
            self.big = ev.get("big", f >= sc["big_presence"])
            if any(self.pattern.match(c) for c in codes):  # a valid parcel-format decode is never overridden by the kraft test
                self._read(codes)
            elif codes:
                self._read(codes)
            elif self.calls >= cfg["scan_max_calls"] or self.remaining() < cfg["budget"]["deliver_reserve_s"]:
                if not (ev.get("obj") and self._center()):  # an object with no readable label: slide it out first
                    self._search_next()
        elif s == "CENTER":  # WALLX slid the parcel into the open middle: look again here, then the flip path takes over
            if codes:
                self._read(codes)
            elif self.calls >= cfg["max_calls"][s] or self.remaining() < cfg["budget"]["deliver_reserve_s"]:
                self._go("SCAN", "centred")
        elif s == "HOLD":  # real_safe latch: arms still, no WALLX call, until the operator ends the episode
            pass
        elif s == "SPIN":
            if codes and self.calls >= ev.get("spin_push", 0):  # the arm is home again
                self._read(codes)
            elif self.calls >= ev.get("spin_calls", 0):
                self._search_next()
        elif s == "SHOW":  # scripted grasp-and-show; show_calls = calls of the plan (0 = none)
            if codes:
                self.pend = codes
                self._go("SHOWSET", "code")
            elif self.calls >= ev.get("show_calls", 0):
                self._go("SHOWSET", "shown")
        elif s == "SHOWSET":  # the parcel is down and the arm home
            if self.calls >= ev.get("showset_calls", 0):
                self._read(self.pend) if self.pend else self._search_next()
        elif s == "SLIDE":  # a stroke is done (arm home, parcel re-fitted next entry); 0 = nothing planned
            n = ev.get("slide_calls", 0)
            if self.calls >= n:
                if n and self.tries["SLIDE"] < cfg["slide"]["max_strokes"] and self._admit("SLIDE"):
                    self._go("SLIDE", "next stroke")
                else:
                    self.end_slide()
        elif s == "SWIPE":  # a stroke is done (arm home, parcel re-fitted next entry); 0 = nothing planned
            n = ev.get("swipe_calls", 0)
            if self.calls >= n:
                if n and f >= sc["presence_min"] and self.tries["SWIPE"] < cfg["swipe"]["max_strokes"] and self._admit("SWIPE"):
                    self._go("SWIPE", "next stroke")
                else:
                    self._go("HOME", "swiped" if f < sc["presence_min"] else "swipe failed")
        elif s == "LOOK":
            if codes or self.calls >= ev.get("look_calls", 0):
                self.pend = codes
                self._go("LOOKHOME", "code" if codes else "looked")
        elif s == "LOOKHOME":
            if self.calls >= cfg["home_calls"]:
                self._read(self.pend) if self.pend else self._search_next()
        elif s == "INSPECT":
            if codes:
                self._read(codes)
            elif self.remaining() < cfg["budget"]["deliver_reserve_s"]:
                self._route(None, "budget")
            elif self.calls >= cfg["max_calls"][s]:  # a retry is a new try: its calls restart at 0, the tries count (and the budget) bound it
                self._search_next()
        elif s == "HANDOVER":
            if over:
                self._go("HOME", "budget")
            elif self.calls >= cfg["max_calls"][s]:
                self._go(self.plan[1], "handed over")
        elif cfg.get("event_fast") and s in WALLX and self._bin_moved(ev):  # WALLX grabbed a bin: latch HOLD, no retry / HOME until a human resets
            self.flags.append("bin_moved")
            self._go("HOLD", "bin_moved")
        elif s in ACTIONS and cfg.get("event_fast"):  # verified DURING the calls; a retry goes straight back to the action, HOME only to end the parcel
            self.ok = self.ok + 1 if self._success(ev) else 0
            if self.ok >= cfg["ok_calls"]:
                self.landed = True
                self._go("HOME", "outcome ok")
            elif self.calls >= cfg["max_calls"][s] or over or self._bad_grasp(ev):
                nxt = self._next_action()
                self._go(nxt or "HOME", "retry" if nxt else ("out of tries" if self.remaining() >= cfg["budget"]["deliver_reserve_s"] else "budget"))
        elif s == "HOME" and cfg.get("event_fast"):
            if self.calls >= cfg["home_calls"]:
                if not self.landed:
                    self.flags.append("failed:%s:%s" % (self.code, self.reason))
                    self.rebase = True
                self._done("ok" if self.landed else "failed")
        elif s in ACTIONS:
            self.ok = self.ok + 1 if self._success(ev) else 0
            if self.ok >= cfg["ok_calls"] or self.calls >= cfg["max_calls"][s] or over:
                self._go("HOME", "outcome ok" if self.ok else "budget")
        elif s == "HOME":
            if self.calls >= cfg["home_calls"]:
                self._go("VERIFY", "home")
        elif s == "VERIFY" and self.calls >= cfg["verify_calls"]:
            if self._success(ev):
                self._done("ok")
            elif self._next_action():
                self._go(self._next_action(), "retry")
            else:
                self.flags.append("failed:%s:%s" % (self.code, self.reason))
                self.rebase = True  # the stuck object becomes background so the line keeps moving
                self._done("failed")
        return self.instruction()
