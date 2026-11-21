"""Scripted fake scene for offline tests: no simulator, no robot. Front image is 1280x720 so QR codes decode."""
import functools

import cv2
import numpy as np
import qrcode
from PIL import Image

from labelloop import look

A, B, UNKNOWN, BAD = "420/95123/001", "420/90001/002", "420/10001/003", "LL|ZONE-A|001"
# bin_rois of the config on 1280x720: x0, y0, x1, y1; zone (induction_roi) is x 179..1075, y 324..705
BINS = {"blue": (243, 0, 640, 202), "red": (653, 0, 1075, 230), "black": (947, 94, 1242, 338)}  # black = the old tray (reject_mode tray only)
COLORS = {"blue": (40, 57, 73), "red": (235, 30, 30), "black": (20, 20, 20)}  # the bins' colours (sim/qr_sort_task/props.py)
KRAFT, RED, ZONE, BIG = (165, 125, 85), (200, 30, 30), (670, 430), (600, 515)


@functools.lru_cache(maxsize=None)
def qr_image(code, size=70):
    return np.asarray(qrcode.make(code).convert("RGB").resize((size, size), Image.NEAREST))


def parcel(code, fail_places=0, hidden=0, big=False, look=None, spin=None, xy=None, half=(0.06, 0.04), face=None, h=0.08, yaw=0.0):
    """spin = K: the label shows on the head camera once K scripted pushes were done (call FakeScene.push(); the test loop does it when the harness reaches the push end). look = K: the label shows on the left wrist while the scripted look-around is at view K (set FakeScene.view). xy = root (x, y) of the parcel centre: drawn as a yaw-0 box of face half-extents `half` (m) at that point (default: a square at ZONE), scripted slides move it (FakeScene.slide). face = F: the label is on face F ("y+", "z-", "bottom", ...: box-frame names of look.FACES) and shows on both wrists while the scripted SHOW holds that face toward the head (FakeScene.show_face; the head never sees it). hidden = N: the label is off the head camera and shows on the left wrist only once the Nth INSPECT try has started."""
    return {"kind": "parcel", "code": code, "fail_places": fail_places, "hidden": hidden, "big": big, "inspects": 0, "look": look, "spin": spin, "pushes": 0, "xy": xy and np.array(xy, float), "half": half, "face": face, "h": h, "yaw": np.radians(yaw), "lift": 0.0}


def nolabel(big=False):
    return {**parcel(None, big=big), "kind": "nolabel"}


def garbage():
    return {**parcel(None), "kind": "garbage"}


class FakeScene:
    """Fake WALLX world. `queue` objects arrive in the induction zone one at a time (a hand is visible for the first
    call), after the zone has been empty `gap` calls. apply(instruction) obeys PUSH ("onto the blue mat") / PLACE
    ("into the blue bin ... right arm") / LIFT2 ("both arms and put it into the blue bin") and REJECT / LIFT2 ("black bin")
    on the 2nd consecutive call; the first fail_places PLACE/LIFT2 attempts of an object do nothing. INSPECT ("turn it")
    and HANDOVER ("hand the") only move the label into view; big objects are drawn large."""

    def __init__(self, queue, gap=6, prefill=None):
        """prefill = {bin colour: [parcel codes]}: parcels already sorted into a bin, their QR facing up in the head view (never in `log`)."""
        self.queue, self.gap, self.waited, self.prefill = list(queue), gap, 0, prefill or {}
        self.grip_state = {}  # {arm: gripper value} overriding the default state (the test loop feeds what the scripted SHOW commanded)
        self.zone, self.hand, self.at, self.log = None, False, {}, []  # at: id(obj) -> mat colour | "black"
        self.kind, self.n, self.grip, self.text, self.view = None, 0, 1.0, None, None
        self.show_face = None  # the face a scripted SHOW currently holds toward the head

    def on(self, where):
        return [o for o in self.log if self.at[id(o)] == where]

    def observe(self):
        front = np.full((720, 1280, 3), (225, 228, 226), np.uint8)
        front[:90] = (10, 60, 20)
        for color, (x0, y0, x1, y1) in BINS.items():
            front[y0:y1, x0:x1] = COLORS[color]
            for i, o in enumerate([{"kind": "parcel", "code": c} for c in self.prefill.get(color, [])] + self.on(color)):
                x, y = x0 + 8 + 64 * (i % 3), y0 + 8 + 64 * (i // 3 % 2)
                front[y:y + 60, x:x + 60] = KRAFT if o["kind"] != "garbage" else RED
                if o["code"]:
                    front[y + 2:y + 58, x + 2:x + 58] = qr_image(o["code"], 56)  # sorted parcels keep their label up, in view
        if self.hand:
            front[360:520, 520:760] = (235, 190, 170)
        o = self.zone
        if o:
            (x, y), h = (BIG, 190) if o["big"] else (ZONE, 60)  # big: ~40% of the zone
            col = RED if o["kind"] == "garbage" else KRAFT
            if o["xy"] is not None:
                hx, hy = o["half"]
                front[look.box_mask(o["xy"], o["yaw"], hx, hy, o["h"], dz=o["lift"]) > 0] = col  # the 3D box: top + the visible walls
                x, y = (int(v) for v in look.table_to_pixel((*o["xy"], look.TABLE_Z + o["lift"] + o["h"])))
            else:
                front[y - h:y + h, x - h:x + h] = col
            if o["code"] and not o["hidden"] and o["face"] is None and (o["spin"] is None or o["pushes"] >= o["spin"]):
                front[y - 35:y + 35, x - 35:x + 35] = qr_image(o["code"])
        wrist = {k: np.full((480, 640, 3), 128, np.uint8) for k in ("left_wrist", "right_wrist")}
        if o and o["hidden"] and self.kind == "INSPECT" and o["inspects"] >= o["hidden"]:  # the left arm turned the label up
            wrist["left_wrist"][100:380, 180:460] = qr_image(o["code"], 280)
        if o and o["look"] is not None and self.view == o["look"]:
            wrist["left_wrist"][100:380, 180:460] = qr_image(o["code"], 280)
        if o and o["face"] is not None and self.show_face == o["face"]:  # the held parcel shows its label face to the head: both wrists see it
            for k in wrist:
                wrist[k][100:380, 180:460] = qr_image(o["code"], 280)
        ee = {"position": [0.1185, 0, 0.181], "rotation_rpy": [0, 0, 0]}  # the home pose (q = 0)
        return ({"front": front, **wrist},
                {"left_ee": {**ee, "gripper": self.grip_state.get("left_ee", 0.2)}, "right_ee": {**ee, "gripper": self.grip_state.get("right_ee", self.grip)}})

    def slide(self, plan):
        """A scripted slide stroke was driven: the parcel moves along the push by plan["dist"] (to plan["c_after"])."""
        if self.zone and self.zone["xy"] is not None:
            self.zone["xy"] = np.array(plan["c_after"], float)

    def swipe(self, plan):
        """A scripted reject stroke was driven: the parcel moves to plan["c_after"]; past the table's right edge it is gone (on the floor)."""
        o = self.zone
        if o and o["xy"] is not None:
            o["xy"] = np.array(plan["c_after"], float)
            if plan["off"] or o["xy"][1] < -0.5642:
                self.at[id(o)], self.zone = "floor", None

    def setdown(self, plan):
        """A scripted SHOW set the parcel down (a look.set_down plan): it lies at plan["spot"] (yaw ignored: the mock draws yaw-0 boxes)."""
        if self.zone and self.zone["xy"] is not None and plan["spot"] is not None:
            self.zone["xy"], self.zone["lift"] = np.array(plan["spot"], float), 0.0

    def push(self):
        if self.zone:
            self.zone["pushes"] += 1

    def apply(self, instruction):
        self.hand = False
        t = (instruction or "").lower()
        kind = ("LIFT2" if "both arms and put" in t else "INSPECT" if "turn it" in t else "HANDOVER" if "hand the" in t
                else "REJECT" if "black bin" in t else "PUSH" if "push" in t else "PLACE" if " bin" in t else None)
        self.n = self.n + 1 if kind and t == self.text else 1
        self.kind, self.text, o = kind, t, self.zone
        if o and kind == "INSPECT" and self.n == 1:  # each try has its own wording: a new INSPECT run
            o["inspects"] += 1
        if o and kind in ("PUSH", "PLACE", "REJECT", "LIFT2") and self.n >= 2:
            lifts = kind in ("PLACE", "LIFT2") and "black" not in t
            o["places"] = o.get("places", 0) + (lifts and self.n == 2)  # one attempt per run
            if not lifts or o["places"] > o["fail_places"]:
                self.at[id(o)] = "black" if "black" in t else "blue" if "blue" in t else "red"
                self.zone = None
        if self.zone is None and self.queue and not kind:
            self.waited += 1
            if self.waited >= self.gap:
                self.zone, self.waited, self.hand = self.queue.pop(0), 0, True
                self.log.append(self.zone)
                self.at[id(self.zone)] = "zone"
