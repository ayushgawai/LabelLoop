import re

from labelloop.brain.instructions import instruction

# IDLE -> SCAN -> SPIN -> (LOOK -> LOOKHOME) -> INSPECT -> route   (label search; each step is skipped when disabled / tries 0)
# SCAN: head ROI codes; none after scan_max_calls -> SPIN.  SPIN: scripted (no WALLX): one arm pushes the parcel so it yaws, goes home,
#   then the head camera re-reads (spin.tries tries). LOOK (look.enabled): scripted wrist-camera look-around. INSPECT: WALLX flips it.
# Nothing found after the last step (or the budget's deliver reserve is hit) -> no_label / budget reject. A big parcel skips SPIN / LOOK.
# route: PLACE (lane arm: A left, B right) | REJECT (right arm) | LIFT2 (big, both arms); with handover: true a HANDOVER precedes
#   PLACE / REJECT. A code routes immediately, even mid-INSPECT. garbage / bad code -> REJECT.
# PLACE/REJECT/LIFT2 -> HOME -> VERIFY -> IDLE (parcel counted) | retry the plan | IDLE after the last try (failure logged)
# Budget: spent estimated REAL seconds per parcel (WALLX call ~wallx_call_s, scripted / hold call ~scripted_call_s).
# PUSH stays an action state but no plan routes to it. DONE only when max_parcels is set.
ACTIONS = ("PUSH", "PLACE", "REJECT", "LIFT2")  # success-checked
COUNTED = ACTIONS + ("INSPECT", "SPIN", "HANDOVER")  # tries counted per parcel
WALLX = ACTIONS + ("INSPECT", "HANDOVER")  # states that send an instruction to WALLX (HOME too unless scripted_home)


class StateMachine:
    """step(ev) -> instruction | None (None = hold still).
    ev = {"presence": zone changed fraction, "kroi": kraft fraction of the zone, "codes": confirmed code texts,
          "bins": {bin colour incl. black: ROI changed fraction vs the snapshot at SCAN}, optional "big": bool override of the presence test,
          "spin_push" / "spin_calls": calls of the scripted push (codes are ignored until it is done) / of the whole spin try}."""

    def __init__(self, cfg):
        self.cfg, self.pattern = cfg, re.compile(cfg["code_pattern"])
        self.lane_of = {z: lane for lane, zips in cfg["sort_plan"].items() for z in zips}
        self.reset()

    def reset(self):
        self.state, self.calls, self.parcels, self.results = "IDLE", 0, 0, []
        self.transitions, self.flags, self.rebase, self.sorted = [], [], False, set()  # sorted: codes done this run
        self._fresh()

    def _fresh(self):
        self.code = self.lane = self.reason = self.plan = self.pend = None
        self.big, self.obj, self.tries, self.ok, self.seen, self.prev = False, "block", {}, 0, 0, 0.0
        self.spent, self.looked = 0.0, False  # estimated real seconds spent on this parcel; LOOK done

    def _go(self, state, why):
        self.transitions.append((self.state, state, why))
        self.state, self.calls, self.ok = state, 0, 0
        if state in COUNTED:
            self.tries[state] = self.tries.get(state, 0) + 1

    def instruction(self):
        s = self.state
        if s in WALLX + ("HOME",):
            key = "INSPECT_BIG" if s == "INSPECT" and self.big else s
            return instruction(key, self.cfg, self.tries.get("INSPECT", 1) - 1 if s == "INSPECT" else 0,
                               mat=self.cfg["lanes"].get(self.lane, "black"), obj=self.obj, arm=self._arm())

    def _arm(self):  # the arm that puts the parcel in its bin: the lane's (handover: false) or the receiving right arm
        return "right" if self.cfg.get("handover") else self.cfg["arms"].get(self.lane or "reject", "right")

    def _cost(self, s):  # estimated real seconds of one call in state s
        b = self.cfg["budget"]
        return b["wallx_call_s"] if s in WALLX or (s == "HOME" and not self.cfg.get("scripted_home")) else b["scripted_call_s"]

    def remaining(self):
        return self.cfg["budget"]["case_s"] - self.spent

    def _next_action(self):  # retry the plan only while EVERY step of it has tries left (HANDOVER too) and time for a delivery remains
        ok = all(self.tries.get(a, 0) < self.cfg["tries"].get(a, 1 << 30) for a in self.plan)
        return self.plan[0] if ok and self.remaining() >= self.cfg["budget"]["deliver_reserve_s"] else None

    def _search_next(self):  # the current label-search step is exhausted: the next enabled one, else reject
        c, sp, lk = self.cfg, self.cfg.get("spin", {}), self.cfg.get("look", {})
        if self.remaining() < c["budget"]["deliver_reserve_s"]:
            self._route(None, "budget")
        elif sp.get("enabled") and not self.big and self.tries.get("SPIN", 0) < sp["tries"]:
            self._go("SPIN", "no code")
        elif lk.get("enabled") and not self.big and not self.looked:
            self.looked = True
            self._go("LOOK", "no code")
        elif self.tries.get("INSPECT", 0) < c["tries"].get("INSPECT", 0):
            self._go("INSPECT", "no code")
        else:
            self._route(None, "no_label")

    def _success(self, ev):  # the zone is empty and the parcel's bin changed (it landed there, not just left the zone)
        sc = self.cfg["scene"]
        bin_ = self.cfg["lanes"][self.lane] if self.lane else "black"
        return ev["presence"] < sc["presence_min"] and ev["bins"][bin_] >= sc["bin_gain"]

    def _route(self, lane, reason, code=None, obj="block"):
        self.lane, self.code, self.reason, self.obj = lane, code, reason, obj
        last = "LIFT2" if self.big else "PLACE" if lane else "REJECT"
        self.plan = [last] if self.big or not self.cfg.get("handover") else ["HANDOVER", last]
        self._go(self.plan[0], reason)

    def _read(self, codes):
        codes = sorted(codes, key=lambda c: not self.pattern.match(c))  # parcel-format codes first
        m = self.pattern.match(codes[0])
        lane = m and self.lane_of.get(m.group(1)[:3])
        self._route(lane, "routed" if lane else "unknown_zip" if m else "bad_code", codes[0])

    def _done(self, outcome):
        self.sorted.add(self.code)
        self.results.append({"code": self.code, "lane": self.lane, "reason": self.reason, "outcome": outcome})
        self.parcels += 1
        self._fresh()
        stop = self.cfg.get("max_parcels") and self.parcels >= self.cfg["max_parcels"]
        self._go("DONE" if stop else "IDLE", outcome)

    def step(self, ev):
        cfg, sc, s = self.cfg, self.cfg["scene"], self.state
        self.transitions, self.flags = [], []
        self.calls += 1
        self.spent += self._cost(s) if s not in ("IDLE", "DONE") else 0
        over = self.spent + cfg["budget"]["wallx_call_s"] > cfg["budget"]["case_s"]  # one more WALLX call would exceed the case budget
        f, codes = ev["presence"], sorted(c for c in ev["codes"] if c not in self.sorted)
        if s == "IDLE":
            if f < sc["presence_min"]:
                self.seen = 0
            else:  # present: stable only while the fraction stops moving (a hand sliding it in keeps changing it)
                self.seen = self.seen + 1 if self.seen and abs(f - self.prev) <= cfg["settle_tol"] else 1
            self.prev = f
            if self.seen >= cfg["stable_calls"]:
                self._go("SCAN", "object in zone")
        elif s == "SCAN":
            self.big = ev.get("big", f >= sc["big_presence"])
            if any(self.pattern.match(c) for c in codes):  # a valid parcel-format decode is never overridden by the kraft test
                self._read(codes)
            elif ev["kroi"] < sc["kraft_min"]:
                self._route(None, "garbage", obj="object")
            elif codes:
                self._read(codes)
            elif self.calls >= cfg["scan_max_calls"] or self.remaining() < cfg["budget"]["deliver_reserve_s"]:
                self._search_next()
        elif s == "SPIN":
            if codes and self.calls >= ev.get("spin_push", 0):  # the arm is home again
                self._read(codes)
            elif self.calls >= ev.get("spin_calls", 0):
                self._search_next()
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
