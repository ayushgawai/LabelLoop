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


def norm_code(text):
    """Barcode / QR text -> '420/NNNNN/NNN': '420/90291/001', '(420)90291', '42090291', or a bare 5-digit pincode '90291'; else None."""
    t = (text or "").strip()
    if re.fullmatch(r"420/\d{5}/\d{3}", t):
        return t
    m = re.search(r"\(?420\)?\D?(\d{5})", t) or re.fullmatch(r"(\d{5})", t)
    return "420/%s/000" % m.group(1) if m else None


BOARD = __import__("pathlib").Path(__file__).resolve().parents[1] / "runs" / "package" / "board.json"  # the phone scanner's board


def phone_code(max_age=180.0, path=None, since=None):
    """The code the phone scanner read last (dashboard /scan -> runs/package/board.json), if seen within max_age seconds; else None."""
    import json
    import time
    try:
        cur = json.loads((path or BOARD).read_text()).get("current") or {}
    except (OSError, ValueError):
        return None
    t = float(cur.get("last_seen") or cur.get("scanned") or 0)
    if since is not None and t < since:  # an old scan (an earlier parcel) never counts
        return None
    return norm_code(cur.get("code")) if cur.get("code") and time.time() - t <= max_age else None


def any_barcodes(img):
    """(text, (cx, cy)) of every 1D / 2D barcode zxing finds (all formats), with an upscaled + sharpened retry for small labels."""
    import cv2
    import zxingcpp
    def c(b, s=1.0):
        p = b.position
        return ((p.top_left.x + p.bottom_right.x) / 2 / s, (p.top_left.y + p.bottom_right.y) / 2 / s)
    out = [(b.text, c(b)) for b in zxingcpp.read_barcodes(img)]
    if not out:
        s = min(3.0, 3840 / max(img.shape[:2]))
        g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        out = [(b.text, c(b, s)) for b in zxingcpp.read_barcodes(cv2.addWeighted(g, 1.8, cv2.GaussianBlur(g, (0, 0), 2), -0.8, 0))]
    return out


ZONE_BOXES = ((0.10, 0.15, 0.38, 0.50), (0.64, 0.15, 0.90, 0.50), (0.78, 0.52, 1.0, 1.0))  # zones 1/2/3 (trays / bins sit there)


TABLE_POLY = ((0.15, 0.19), (0.86, 0.19), (0.97, 0.80), (0.02, 0.80))  # the table top in the head image, incl. a parcel standing at the
# back edge (it sticks up over the wall); stops above the robot's front strip where the arm bases and parked claws are


def table_free_mask(img_rgb, hsv=None):
    """1 where a loose parcel can be: the table polygon minus the zone boxes (trays, bins) and the parked left claw."""
    import cv2
    H, W = img_rgb.shape[:2]
    inside = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(inside, np.array([[int(x * W), int(y * H)] for x, y in TABLE_POLY], np.int32), 1)
    for x0, y0, x1, y1 in ZONE_BOXES + ((0.0, 0.60, 0.20, 1.0),):
        inside[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)] = 0
    return inside


def locate_dark(img_rgb, vmax=70, min_area=1500):
    """(x, y) image fractions of the largest dark object ON THE TABLE (inside the convex hull of the green mat; parked-claw corners
    masked), e.g. the black cap; None if nothing big enough."""
    import cv2
    H, W = img_rgb.shape[:2]
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    inside = table_free_mask(img_rgb, hsv)
    m = cv2.morphologyEx(((hsv[..., 2] < vmax) & (inside > 0)).astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, _, st, cen = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    i = 1 + int(np.argmax(st[1:, 4]))
    return None if st[i][4] < min_area else (float(cen[i][0] / W), float(cen[i][1] / H))


def dark_in_box(img_rgb, box, vmax=70):
    """Area (px) of the largest dark blob inside an image box (fractions): the dark parcel sitting in a zone's tray."""
    import cv2
    H, W = img_rgb.shape[:2]
    x0, y0, x1, y1 = box
    hsv = cv2.cvtColor(img_rgb[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)], cv2.COLOR_RGB2HSV)
    m = cv2.morphologyEx((hsv[..., 2] < vmax).astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(m)
    return int(st[1:, 4].max()) if n > 1 else 0


def light_mask(img_rgb):
    """White / light parcel (the mailer bag): low saturation, bright."""
    import cv2
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    return ((hsv[..., 1] < 60) & (hsv[..., 2] > 170)).astype(np.uint8)


def locate_parcel(img_rgb, kind="dark", min_area=1500):
    """(x, y) image fractions of the parcel on the free table: 'dark' (the cap) or 'light' (the white mailer)."""
    if kind != "light":
        return locate_dark(img_rgb, min_area=min_area)
    import cv2
    H, W = img_rgb.shape[:2]
    m = cv2.morphologyEx(light_mask(img_rgb) & table_free_mask(img_rgb), cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    n, _, st, cen = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    i = 1 + int(np.argmax(st[1:, 4]))
    return None if st[i][4] < min_area else (float(cen[i][0] / W), float(cen[i][1] / H))


def parcel_in_box(img_rgb, box, kind="dark", base=None):
    """Area (px) of the parcel's largest blob inside a zone box. kind 'change': what differs from `base` (the box's own
    image when the step started): the held parcel (+ claw) arriving over the bin, whatever its colour."""
    if kind == "change":
        import cv2
        H, W = img_rgb.shape[:2]
        x0, y0, x1, y1 = box
        cur = img_rgb[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)]
        if base is None or base.shape != cur.shape:
            return 0
        m = (cv2.absdiff(cv2.GaussianBlur(cur, (7, 7), 0), cv2.GaussianBlur(base, (7, 7), 0)).max(axis=2) > 40).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        n, _, st, _ = cv2.connectedComponentsWithStats(m)
        return int(st[1:, 4].max()) if n > 1 else 0
    if kind != "light":
        return dark_in_box(img_rgb, box)
    import cv2
    H, W = img_rgb.shape[:2]
    x0, y0, x1, y1 = box
    m = cv2.morphologyEx(light_mask(img_rgb[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)]), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(m)
    return int(st[1:, 4].max()) if n > 1 else 0


# head-camera pixel -> table point in the RIGHT arm's home-relative frame (least squares over recorded grasps: scenario 1 bag,
# scenario 3 bag after the flip, scenario 2 Miro box, scenario 4 Bhindi box). The left arm's base is 0.51 m to the left
# (same as the two-arm carry), so left = right-frame point - (0, 0.51). ponytail: affine from 4 grasps (~4 cm), a proper
# calibration (claw tip at a grid of points) if picks keep landing beside the parcel.
PIX2RIGHT = None
BASE_SEP = 0.51
GRASP_RPY = {"left": [-0.13, 0.80, -0.29], "right": [0.11, 1.01, 0.38]}  # recorded top-down grasp orientations


def pix_to_arm(u, v, arm):
    global PIX2RIGHT
    if PIX2RIGHT is None:
        P = np.array([[648, 341], [651, 383], [501, 373], [964, 310]], float)
        Y = np.array([[0.374, 0.277], [0.32, 0.245], [0.30, 0.295], [0.303, 0.026]])
        PIX2RIGHT = np.linalg.lstsq(np.c_[P, np.ones(len(P))], Y, rcond=None)[0]
    x, y = np.array([u, v, 1.0]) @ PIX2RIGHT
    return (float(x), float(y) - (BASE_SEP if arm == "left" else 0.0))


def find_parcel_px(img_rgb, roi=None):
    """(u, v) pixel centre of the loose parcel on the table (not trays / bins / claws), or None; roi = x-range fractions."""
    import cv2
    from labelloop.dashboard.track import find_parcel
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    if roi:
        W = img.shape[1]
        img = img.copy()
        img[:, :int(roi[0] * W)] = (90, 140, 40)   # paint outside the range as mat
        img[:, int(roi[1] * W):] = (90, 140, 40)
    b = find_parcel(img)
    return (b[0] + b[2] / 2, b[1] + b[3] / 2) if b else None


def where_words(x, y):
    """Image fractions -> table grid words for the WALLX prompt: rows back / middle / front, columns left / middle / right."""
    col = "left" if x < 0.38 else "middle" if x < 0.62 else "right"
    row = "back" if y < 0.42 else "middle" if y < 0.62 else "front"
    return "the centre" if (row, col) == ("middle", "middle") else "the %s %s" % (row, col)


class Sequence:
    def __init__(self, spec, runlog=None):
        self.spec, self.runlog = spec, runlog
        self.reset()

    def reset(self):
        self.i, self.calls, self.code, self.mat, self.source = 0, 0, None, "black", None
        self.hold_grip, self.gcmd = {}, {}  # arm -> gripper command that holds a delicate object (phone) without squeezing it
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
        if a == "picked":  # the arm the autopick used
            return getattr(self, "pick_arm", "right")
        return {"blue": "left"}.get(self.mat, "right") if a == "lane" else a

    def lock_arms(self):
        s = getattr(self, "active", None) or self.step_spec  # the step that produced THIS call's instruction: on a say step's last call
        # the index has already advanced, and locking by step_spec let WALLX move the idle arm for that one call (sc5i left-claw bug)
        arm = self._arm(s) if s and "say" in s else None
        return tuple(x for x in ("left", "right") if arm and x != arm)

    def _read(self, images, step=None):
        """A read step may override "roi" (where on the head image to look; also filters barcodes) and "code_hint" (two parcels: one per
        half of the table)."""
        opt = lambda k, d=None: (step or {}).get(k, self.spec.get(k, d))  # noqa: E731
        front, roi = images["front"], opt("roi", [0.02, 0.48, 0.78, 1.0])
        H, W = front.shape[:2]
        inside = lambda c: step is None or "roi" not in step or (roi[0] * W <= c[0] <= roi[2] * W and roi[1] * H <= c[1] <= roi[3] * H)  # noqa: E731
        codes = [c for c in (norm_code(t) for t, ctr in locate(front) + any_barcodes(front) if inside(ctr)) if c]
        source = "qr" if codes else None
        if not codes:
            zips = [z for z, _ in label_ocr.zips(front, roi)]
            if not zips and opt("zip_vote"):  # small labels: multi-crop OCR vote, known pincodes preferred
                v = label_ocr.zip_vote(front, roi)
                zips = [v[0]] if v else []
            codes, source = (["420/%s/000" % zips[0]], "zip") if zips else ([], None)
        if not codes and opt("code_hint"):
            codes, source = [opt("code_hint")], "demo"  # DEMO: the label counts as read
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
        while self.step_spec is not None and ("read" in self.step_spec or "locate" in self.step_spec):  # cost no robot call
            if self.step_spec.get("from_phone"):  # the label comes from the phone scanner: hold (arms still) until it has read one
                self._phone_since = getattr(self, "_phone_since", None) or __import__("time").time() - 2.0
                code = phone_code(self.step_spec.get("max_age", 180), since=self._phone_since)  # a scan made for THIS parcel only
                if not code and self.calls < self.step_spec.get("wait_calls", 0):
                    self.calls += 1
                    self.waiting_phone = True
                    break
                self.waiting_phone = False
                if code:
                    self.code, self.source, self.mat = code, "phone", bin_for(code)
                    self.i, self.calls = self.i + 1, 0
                    continue
                if self.step_spec.get("required", True):  # no scan in time: never guess a zone; skip to the last step (home)
                    self.no_scan = True
                    self.i, self.calls = len(self.spec["steps"]) - 1, 0
                    continue
            if "locate" in self.step_spec:  # where is the parcel now (it may have been moved): grid words for {where}
                xy = locate_parcel(images["front"], self.spec.get("parcel", "dark"))
                self.where = where_words(*xy) if xy else self.step_spec.get("default", "the centre")
                self.where_xy = [round(v, 3) for v in xy] if xy else None
            else:
                self._read(images, self.step_spec)
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
        elif s is not None and "wait" in s:  # hold still (e.g. a person puts the phone / parcel into a claw), grips kept
            instr = "(wait %d)" % self.i
            self.calls += 1
            if self.calls >= s.get("calls", 10):
                self.i, self.calls = self.i + 1, 0
        elif s is not None and "autopick" in s:  # find the parcel on the head camera, then: above it (claw open) -> down -> close -> lift
            c = s["autopick"]
            if self.calls == 0:
                px = find_parcel_px(images["front"], c.get("roi"))
                if px is None:  # nothing found: skip the pick (and the step after it, the WALLX place) to the end
                    self.no_parcel = True
                    self.i, self.calls = len(self.spec["steps"]) - 1, 0
                    instr = "(autopick %d: no parcel)" % self.i
                else:
                    arm = c.get("arm", "auto")
                    if arm == "auto":
                        arm = "left" if px[0] < images["front"].shape[1] / 2 else "right"
                    x, y = pix_to_arm(px[0], px[1], arm)
                    x, y = x + c.get("dx", 0.0), y + c.get("dy", 0.0)
                    zg, zh, rpy = c.get("z", 0.0), c.get("z_high", 0.15), c.get("rpy", GRASP_RPY[arm])
                    self.pick_arm, self.pick_px = arm, [round(px[0]), round(px[1])]
                    p0 = list(state[arm + "_ee"]["position"])
                    self.pick_plan = [([p0[0], p0[1], zh], rpy, 1.5, 3),  # straight up first (never sweep low across the table)
                                      ([x, y, zh], rpy, 1.5, c.get("approach_calls", 7)), ([x, y, zg], rpy, 1.5, c.get("down_calls", 4)),
                                      ([x, y, zg], rpy, 0.0, 2), ([x, y, zh], rpy, 0.0, 3)]
            if getattr(self, "pick_plan", None) and self.step_spec is s:
                instr = "(autopick %d)" % self.i
                self.calls += 1
                self.pick_k = self.calls  # (scripted_reply runs after the index may have advanced and calls reset)
                if self.calls >= sum(p[3] for p in self.pick_plan):
                    self.i, self.calls = self.i + 1, 0
                    self.pick_plan_done = self.pick_plan
        elif s is not None and "wait_parcel" in s:  # arms at rest, the harness keeps looking: the parcel back on the table
            instr = "(watch %d)" % self.i                # (a person took it out of the bin) -> next step (WALLX again)
            self.calls += 1
            kind = self.spec.get("parcel", "dark")
            if "bin_box" in s:  # first the parcel must LEAVE the bin (its area there drops), only then count it on the table
                area = parcel_in_box(images["front"], ZONE_BOXES[s["bin_box"]], kind)
                if self.calls == 1:
                    self.bin_full = area
                    self.left_bin = area < s.get("full_area", 5000)  # already out of the bin when watching starts: fine
                self.left_bin = self.left_bin or area < self.bin_full - s.get("out_area", 2500)
            on_table = locate_parcel(images["front"], kind) and ("bin_box" not in s or self.left_bin)
            self.seen_n = getattr(self, "seen_n", 0) + 1 if on_table else 0
            if self.seen_n >= 3:
                self.seen_n, self.i, self.calls, self.parcel_back, self.left_bin = 0, self.i + 1, 0, True, False
            elif self.calls >= s.get("calls", 120):  # nobody took it out: finish (skip the remaining steps)
                self.i, self.calls = len(self.spec["steps"]) - 1, 0
        elif s is not None and "grip" in s:  # gentle grip: close slowly until the fingers stop on the object, then hold just there
            instr = "(grip %d)" % self.i
            self.calls += 1
            if self._grip_update(s, state) or self.calls >= s.get("calls", 40):
                for arm in s["grip"]:
                    self.hold_grip[arm] = self.hold_grip.get(arm, self.gcmd.get(arm, 0.0))
                self.i, self.calls, self.gcmd = self.i + 1, 0, {}
        elif s is not None and "move" in s:  # scripted move (organisers allowed simple scripted steps): scripted_reply drives it
            instr = "(scripted move %d)" % self.i
            self.calls += 1
            if self.calls >= s.get("calls", 3):
                self.i, self.calls = self.i + 1, 0
        elif s is not None and ("read" in s or "locate" in s):  # waiting for the phone scan: hold
            instr = None
        elif s is not None:
            moved = False
            claw = float(state.get((self._arm(s) or "right") + "_ee", {}).get("gripper", 1.0) or 0.0)
            if s.get("relocate") and self.calls < s.get("relocate_calls", 45) and claw >= 0.3:  # (claw closed = holding it: no re-target)
                # the harness keeps watching the parcel: if it was
                xy = locate_parcel(images["front"], self.spec.get("parcel", "dark"))  # moved to another grid cell (same cell 3 calls in a row), the prompt is cancelled
                cell = where_words(*xy) if xy else None  # and WALLX gets a new one with the new location (history reset)
                self._cand, self._candn = (cell, getattr(self, "_candn", 0) + 1) if cell and cell == getattr(self, "_cand", None) else (cell, 1)
                if cell and cell != getattr(self, "where", None) and self._candn >= 3:
                    self.where, self.where_xy, self.moved_n = cell, [round(v, 3) for v in xy], getattr(self, "moved_n", 0) + 1
                    moved = True
                elif s.get("reissue_every") and self.calls and self.calls % s["reissue_every"] == 0:  # no progress: re-scan and
                    if cell:  # give WALLX the task again from scratch (history reset), with where the parcel is now
                        self.where, self.where_xy = cell, [round(v, 3) for v in xy]
                    self.reissued = getattr(self, "reissued", 0) + 1
                    moved = "reissue"
            tmpl = s["say_moved"] if s.get("say_moved") and getattr(self, "moved_n", 0) else s["say"]
            instr = tmpl.format(mat=self.mat, zone=self.spec.get("zones", {}).get(self.mat, ""), arm=self._arm(s) or "right",
                                    where=getattr(self, "where", "the centre"))
            self.calls += 1
            if s.get("until") == "in_bin":  # the harness keeps looking: parcel in the target zone's tray and gone from the table
                box = ZONE_BOXES[s["bin_box"] if "bin_box" in s else {"blue": 0, "red": 1}.get(self.mat, 2)]  # bin_box: fixed target
                kind = s.get("parcel", self.spec.get("parcel", "dark"))
                if kind == "change":  # held parcel over the target bin (whatever its colour): compare with the bin at step start
                    H, W = images["front"].shape[:2]
                    if self.calls == 1 or getattr(self, "bin_img", None) is None:
                        self.bin_img = images["front"][int(box[1] * H):int(box[3] * H), int(box[0] * W):int(box[2] * W)].copy()
                    area = parcel_in_box(images["front"], box, "change", self.bin_img)
                    z = float(state.get((self._arm(s) or "right") + "_ee", {}).get("position", [0, 0, 1])[2])
                    low = z < s.get("bin_z", 0.10)  # the arm lowered the parcel into the bin (not just carrying it past)
                    self.bincount = getattr(self, "bincount", 0) + 1 if area >= s.get("bin_area", 6000) and low else 0
                else:
                    area = parcel_in_box(images["front"], box, kind)
                    if self.calls == 1 or getattr(self, "bin_base", None) is None:  # how the bin looked when this round started
                        self.bin_base = area if kind == "light" else 0  # (white tape / wall already in the box: count only what is new)
                        H, W = images["front"].shape[:2]
                        self.bin_img = images["front"][int(box[1] * H):int(box[3] * H), int(box[0] * W):int(box[2] * W)].copy()
                    hit = area - self.bin_base >= s.get("bin_area", 7000) and not locate_parcel(images["front"], kind)
                    if s.get("also_change"):  # OR: the bin looks different and the arm is lowered into it (parcel being put there)
                        z = float(state.get((self._arm(s) or "right") + "_ee", {}).get("position", [0, 0, 1])[2])
                        hit = hit or (parcel_in_box(images["front"], box, "change", self.bin_img) >= s.get("change_area", 6000)
                                      and z < s.get("bin_z", 0.10))
                    self.bincount = getattr(self, "bincount", 0) + 1 if hit else 0
                if self.bincount >= s.get("bin_calls", 3):  # 3 calls in a row: done, stop WALLX, release + home
                    self.i, self.calls, self.bincount, self.task_done = self.i + 1, 0, 0, True
                    self.subgoal_changed = True
            if s.get("until") == "in_zone":  # cut WALLX off as soon as the parcel is over its zone (m1: it kept carrying it around)
                inside, c = self._bag_in_zone(images["front"])
                self.seen_out = getattr(self, "seen_out", False) or (c is not None and not inside)  # only after the parcel was outside
                self.zonecount = getattr(self, "zonecount", 0) + 1 if inside and self.seen_out else 0  # (run 3: it started half inside)
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
        self.subgoal_changed = self.i != before or self.calls == 1 or (s is not None and "say" in s and locals().get("moved", False))
        self.done = s is None
        self.state_name = "SEQ%d" % before
        self.last_log = {"step": before, "instruction": instr, "code": self.code, "source": self.source, "bin": self.mat,
                         "lock": list(self.lock_arms()), "where": getattr(self, "where_xy", None), "pick": getattr(self, "pick_px", None), "waiting_phone": getattr(self, "waiting_phone", False),
                         "flags": (["re-scan: parcel at %s, task given again" % self.where] if locals().get("moved") == "reissue"
                                   else ["parcel moved to %s: new WALLX prompt" % self.where] if locals().get("moved")
                                   else ["no phone scan arrived: no zone guessed, going home"] if getattr(self, "no_scan", False) and self.i != before
                                   else ["parcel taken out of the bin: WALLX gets the task again"] if getattr(self, "parcel_back", False) and self.i != before
                                   else ["parcel in the target bin: task done, going home"] if getattr(self, "task_done", False) and self.i != before else []), "left_gripper": state["left_ee"]["gripper"], "right_gripper": state["right_ee"]["gripper"]}
        if "parcel in the target bin: task done, going home" in self.last_log["flags"]:
            self.task_done = False  # log it once
        if "parcel taken out of the bin: WALLX gets the task again" in self.last_log["flags"]:
            self.parcel_back = False
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

    def _grip_update(self, s, state):
        """One call of a gentle grip: each arm's command steps down by `rate`; when the measured opening stops following (the
        fingers touch the object), hold at measured - squeeze[arm]. True when every arm has contact."""
        rate, done = s.get("rate", 0.05), True
        for arm, squeeze in s["grip"].items():
            if arm in self.hold_grip:
                continue
            g = float(state[arm + "_ee"]["gripper"])
            cmd = self.gcmd.get(arm, g)
            if cmd < 1.4 and g - cmd > s.get("contact", 0.10):  # blocked by the object: contact
                self.hold_grip[arm] = max(0.0, g - squeeze)
                continue
            self.gcmd[arm] = max(0.0, cmd - rate)
            done = False
        return done

    def _with_grips(self, state, targets):
        """Every arm holding a delicate object keeps its hold command (a plain 'hold still' would command the measured opening:
        no grip force, the phone could slip)."""
        for arm, g in self.hold_grip.items():
            k = arm + "_ee"
            t = targets.setdefault(k, {"position": list(state[k]["position"]), "rotation_rpy": list(state[k]["rotation_rpy"])})
            t.setdefault("gripper", g)
        return targets

    def scripted_reply(self, embodiment, state):
        s = self.active
        if s and "grip" in s:
            t = {a + "_ee": {"position": list(state[a + "_ee"]["position"]), "rotation_rpy": list(state[a + "_ee"]["rotation_rpy"]),
                             "gripper": self.hold_grip.get(a, self.gcmd.get(a, float(state[a + "_ee"]["gripper"])))} for a in s["grip"]}
            return move_response(embodiment, state, self._with_grips(state, t))
        if s and ("wait" in s or "wait_parcel" in s or (("read" in s or "locate" in s) and getattr(self, "waiting_phone", False))):
            return move_response(embodiment, state, self._with_grips(state, {}))
        if s and "release" in s:  # open the acting arm's claw fully where it is, then lift it (release over the zone)
            arm = ({"blue": "left"}.get(self.mat, "right") if s["release"] == "lane" else
                   getattr(self, "pick_arm", "right") if s["release"] == "picked" else s["release"])
            k = arm + "_ee"
            pos = np.asarray(state[k]["position"], float)
            if self.calls == 0 or self.calls > s.get("open_calls", 2):  # step advanced (calls reset) or lifting phase
                pos = pos + np.array([0, 0, min(0.04, s.get("lift", 0.08))])
            g = s.get("to", 1.5) if s.get("open", True) else float(state[k]["gripper"])  # "open": false = lift clear, claw as it is
            if s.get("open", True):
                self.hold_grip.pop(arm, None)  # let go: no more hold command for this arm
            return move_response(embodiment, state, self._with_grips(state, {k: {"position": pos.tolist(), "rotation_rpy": list(state[k]["rotation_rpy"]), "gripper": g}}))
        if s and "autopick" in s and getattr(self, "pick_plan", None):
            k, n = getattr(self, "pick_k", self.calls), 0
            for pos, rpy, g, calls in self.pick_plan:  # the waypoint this call heads for
                n += calls
                if k <= n:
                    break
            key = self.pick_arm + "_ee"
            cur = np.asarray(state[key]["position"], float)
            goal = np.asarray(pos, float)
            d = np.linalg.norm(goal - cur)
            cap = s["autopick"].get("max_step", 0.06)
            nxt = goal if d <= cap else cur + (goal - cur) * cap / d
            return move_response(embodiment, state, self._with_grips(state, {key: {"position": nxt.tolist(), "rotation_rpy": rpy, "gripper": g}}))
        if s and "carry" in s:
            return self._carry(embodiment, state, s)
        if s and "replay" in s:
            span = self.replay[1] if getattr(self, "replay", None) and self.replay[0] is s else None
            if span is None:
                return None
            d, (a, b) = s["_demo"], span
            acts = {}
            u = np.linspace(0, 1, 32)
            ease = u * u * (3 - 2 * u) if s.get("ease", True) else u  # each chunk eases in / out: the client pauses between chunks, no hard stop
            for arm in ("left", "right"):
                seg = np.asarray(d[arm][a:b], float)
                src = np.linspace(0, 1, len(seg)) if len(seg) > 1 else np.zeros(1)
                rows = [[float(np.interp(e, src, seg[:, k])) for k in range(seg.shape[1])] for e in ease] if len(seg) > 1 else [seg[0].tolist()] * 32
                acts[arm + "_ee"] = {"position": [r[:3] for r in rows], "rotation_rpy": [r[3:6] for r in rows], "gripper": [r[6] for r in rows]}
                if arm in self.hold_grip:  # a delicate object (phone) in this claw: the gentle hold, never the recorded command
                    acts[arm + "_ee"]["gripper"] = [float(self.hold_grip[arm])] * len(rows)
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
        return move_response(embodiment, state, self._with_grips(state, targets))

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
