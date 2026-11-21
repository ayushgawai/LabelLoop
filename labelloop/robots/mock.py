"""Scripted fake scene for offline tests: no simulator, no robot. Front image is 1280x720 so QR codes decode."""
import functools

import numpy as np
import qrcode
from PIL import Image

A, B, UNKNOWN, BAD = "420/95123/001", "420/90001/002", "420/10001/003", "LL|ZONE-A|001"
# bin_rois of the config on 1280x720: x0, y0, x1, y1; zone (induction_roi) is x 179..1075, y 324..705
BINS = {"blue": (371, 0, 627, 317), "red": (653, 0, 954, 317), "black": (947, 94, 1242, 338)}
COLORS = {"blue": (40, 57, 73), "red": (235, 30, 30), "black": (20, 20, 20)}  # the bins' colours (sim/qr_sort_task/props.py)
KRAFT, RED, ZONE, BIG = (165, 125, 85), (200, 30, 30), (670, 430), (600, 515)


@functools.lru_cache(maxsize=None)
def qr_image(code, size=70):
    return np.asarray(qrcode.make(code).convert("RGB").resize((size, size), Image.NEAREST))


def parcel(code, fail_places=0, hidden=0, big=False, look=None, spin=None):
    """spin = K: the label shows on the head camera once K scripted pushes were done (call FakeScene.push(); the test loop does it when the harness reaches the push end). look = K: the label shows on the left wrist while the scripted look-around is at view K (set FakeScene.view). hidden = N: the label is off the head camera and shows on the left wrist only once the Nth INSPECT try has started."""
    return {"kind": "parcel", "code": code, "fail_places": fail_places, "hidden": hidden, "big": big, "inspects": 0, "look": look, "spin": spin, "pushes": 0}


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

    def __init__(self, queue, gap=6):
        self.queue, self.gap, self.waited = list(queue), gap, 0
        self.zone, self.hand, self.at, self.log = None, False, {}, []  # at: id(obj) -> mat colour | "black"
        self.kind, self.n, self.grip, self.text, self.view = None, 0, 1.0, None, None

    def on(self, where):
        return [o for o in self.log if self.at[id(o)] == where]

    def observe(self):
        front = np.full((720, 1280, 3), (225, 228, 226), np.uint8)
        front[:90] = (10, 60, 20)
        for color, (x0, y0, x1, y1) in BINS.items():
            front[y0:y1, x0:x1] = COLORS[color]
            for i, o in enumerate(self.on(color)):
                x, y = x0 + 8 + 64 * (i % 3), y0 + 8 + 64 * (i // 3 % 2)
                front[y:y + 60, x:x + 60] = KRAFT if o["kind"] != "garbage" else RED
        if self.hand:
            front[360:520, 520:760] = (235, 190, 170)
        o = self.zone
        if o:
            (x, y), h = (BIG, 190) if o["big"] else (ZONE, 60)  # big: ~40% of the zone
            front[y - h:y + h, x - h:x + h] = RED if o["kind"] == "garbage" else KRAFT
            if o["code"] and not o["hidden"] and (o["spin"] is None or o["pushes"] >= o["spin"]):
                front[y - 35:y + 35, x - 35:x + 35] = qr_image(o["code"])
        wrist = {k: np.full((480, 640, 3), 128, np.uint8) for k in ("left_wrist", "right_wrist")}
        if o and o["hidden"] and self.kind == "INSPECT" and o["inspects"] >= o["hidden"]:  # the left arm turned the label up
            wrist["left_wrist"][100:380, 180:460] = qr_image(o["code"], 280)
        if o and o["look"] is not None and self.view == o["look"]:
            wrist["left_wrist"][100:380, 180:460] = qr_image(o["code"], 280)
        ee = {"position": [0.1185, 0, 0.181], "rotation_rpy": [0, 0, 0]}  # the home pose (q = 0)
        return ({"front": front, **wrist},
                {"left_ee": {**ee, "gripper": 0.2}, "right_ee": {**ee, "gripper": self.grip}})

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
