import re

from labelloop.brain.instructions import instruction

# IDLE -> SCAN -> INSPECT -> route                (SCAN: head ROI + wrist codes; no code after scan_max_calls -> INSPECT)
# SCAN | INSPECT -> HANDOVER -> PLACE | REJECT    (code routes immediately, even mid-INSPECT; garbage / no_label / bad code -> REJECT)
#                -> LIFT2                         (big parcel: both arms do it all, lane or reject)
# INSPECT: left arm picks the parcel up and turns it, tries.INSPECT tries; last one without a code -> no_label reject.
# PLACE/REJECT/LIFT2 -> HOME -> VERIFY -> IDLE (parcel counted) | retry from HANDOVER | IDLE after the last try (failure logged)
# PUSH stays an action state but no plan routes to it. DONE only when max_parcels is set.
ACTIONS = ("PUSH", "PLACE", "REJECT", "LIFT2")  # success-checked
COUNTED = ACTIONS + ("INSPECT",)  # tries counted per parcel
class StateMachine:
    """step(ev) -> instruction | None (None = hold still).
    ev = {"presence": zone changed fraction, "kroi": kraft fraction of the zone, "codes": confirmed code texts,
          "kmats": {mat colour: kraft area fraction inside its bbox}, "reject": reject-bin changed fraction, optional "big": bool override of the presence test}."""

    def __init__(self, cfg):
        self.cfg, self.pattern = cfg, re.compile(cfg["code_pattern"])
        self.lane_of = {z: lane for lane, zips in cfg["sort_plan"].items() for z in zips}
        self.reset()

    def reset(self):
        self.state, self.calls, self.parcels, self.results = "IDLE", 0, 0, []
        self.transitions, self.flags, self.rebase, self.sorted = [], [], False, set()  # sorted: codes done this run
        self._fresh()

    def _fresh(self):
        self.code = self.lane = self.reason = self.plan = None
        self.big, self.obj, self.tries, self.base, self.ok, self.seen, self.prev = False, "block", {}, {}, 0, 0, 0.0

    def _go(self, state, why):
        self.transitions.append((self.state, state, why))
        self.state, self.calls, self.ok = state, 0, 0
        if state in COUNTED:
            self.tries[state] = self.tries.get(state, 0) + 1

    def instruction(self):
        s = self.state
        if s in COUNTED + ("HANDOVER", "HOME"):
            key = "INSPECT_BIG" if s == "INSPECT" and self.big else s
            return instruction(key, self.cfg, self.tries.get("INSPECT", 1) - 1 if s == "INSPECT" else 0,
                               mat=self.cfg["lanes"].get(self.lane, "black"), obj=self.obj)

    def _next_action(self):  # retry the whole plan while its last (place) step has tries left
        return self.plan[0] if self.tries.get(self.plan[-1], 0) < self.cfg["tries"][self.plan[-1]] else None

    def _success(self, ev):
        sc = self.cfg["scene"]
        gain = {c: ev["kmats"][c] - self.base.get(c, 0) for c in ev["kmats"]}
        empty = ev["presence"] < sc["presence_min"]
        if self.lane:
            return empty and gain[self.cfg["lanes"][self.lane]] >= sc["mat_gain"]
        if "reject_bin_roi" in sc:  # the object must have landed in the black bin, not just left the zone
            return empty and ev.get("reject", 0.0) >= sc["reject_gain"]
        return empty and max(gain.values()) < sc["mat_gain"]

    def _route(self, lane, reason, code=None, obj="block"):
        self.lane, self.code, self.reason, self.obj = lane, code, reason, obj
        last = "LIFT2" if self.big else "PLACE" if lane else "REJECT"
        self.plan = [last] if self.big else ["HANDOVER", last]
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
        f, codes = ev["presence"], sorted(c for c in ev["codes"] if c not in self.sorted)
        if s == "IDLE":
            if f < sc["presence_min"]:
                self.seen = 0
            else:  # present: stable only while the fraction stops moving (a hand sliding it in keeps changing it)
                self.seen = self.seen + 1 if self.seen and abs(f - self.prev) <= cfg["settle_tol"] else 1
            self.prev = f
            if self.seen >= cfg["stable_calls"]:
                self.base = dict(ev["kmats"])
                self._go("SCAN", "object in zone")
        elif s == "SCAN":
            self.big = ev.get("big", f >= sc["big_presence"])
            if ev["kroi"] < sc["kraft_min"]:
                self._route(None, "garbage", obj="object")
            elif codes:
                self._read(codes)
            elif self.calls >= cfg["scan_max_calls"]:
                self._go("INSPECT", "no code")
        elif s == "INSPECT":
            if codes:
                self._read(codes)
            elif self.calls >= cfg["max_calls"][s]:
                if self.tries[s] < cfg["tries"][s]:
                    self._go(s, "next try")
                else:
                    self._route(None, "no_label")
        elif s == "HANDOVER":
            if self.calls >= cfg["max_calls"][s]:
                self._go(self.plan[1], "handed over")
        elif s in ACTIONS:
            self.ok = self.ok + 1 if self._success(ev) else 0
            if self.ok >= cfg["ok_calls"] or self.calls >= cfg["max_calls"][s]:
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
