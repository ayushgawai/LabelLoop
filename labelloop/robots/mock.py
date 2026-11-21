"""Scripted fake scene for offline tests: no simulator, no robot. Front image is 1280x720 so QR codes decode."""
import functools

import numpy as np
import qrcode
from PIL import Image

A, B, UNKNOWN, BAD = "420/95123/001", "420/90001/002", "420/10001/003", "LL|ZONE-A|001"
MATS = {"blue": (80, 300, 300, 560), "red": (1100, 300, 1270, 560)}  # bins: x0, y0, x1, y1; zone (induction_roi) is x 589..1075, y 346..662
COLORS = {"blue": (40, 57, 73), "red": (235, 30, 30)}  # the shoe boxes' outside colours (sim/qr_sort_task/props.py)
KRAFT, RED, ZONE = (165, 125, 85), (200, 30, 30), (670, 430)
BIN = (928, 108, 1126, 288)  # black reject bin = reject_bin_roi (0.725, 0.15, 0.88, 0.40) of 1280x720


@functools.lru_cache(maxsize=None)
def qr_image(code, size=70):
    return np.asarray(qrcode.make(code).convert("RGB").resize((size, size), Image.NEAREST))


def parcel(code, fail_places=0):
    return {"kind": "parcel", "code": code, "fail_places": fail_places}


def nolabel():
    return {"kind": "nolabel", "code": None, "fail_places": 0}


def garbage():
    return {"kind": "garbage", "code": None, "fail_places": 0}


class FakeScene:
    """Fake WALLX world. `queue` objects arrive in the induction zone one at a time (a hand is visible for the first
    call), after the zone has been empty `gap` calls. apply(instruction) obeys PUSH ("onto the blue mat") / PLACE
    ("into the blue bin") and REJECT ("into the black bin") on the 2nd consecutive call; the first fail_places PLACE
    attempts of an object do nothing."""

    def __init__(self, queue, gap=6):
        self.queue, self.gap, self.waited = list(queue), gap, 0
        self.zone, self.hand, self.at, self.log = None, False, {}, []  # at: id(obj) -> mat colour | "reject"
        self.kind, self.n, self.grip = None, 0, 1.0

    def on(self, where):
        return [o for o in self.log if self.at[id(o)] == where]

    def observe(self):
        front = np.full((720, 1280, 3), (225, 228, 226), np.uint8)
        front[:90] = (10, 60, 20)
        for color, (x0, y0, x1, y1) in MATS.items():
            front[y0:y1, x0:x1] = COLORS[color]
            for i, o in enumerate(self.on(color)):
                cx = (x0 + x1) // 2 + 30 * (i % 3) - 30
                front[y1 - 100:y1 - 40, cx - 30:cx + 30] = KRAFT if o["kind"] != "garbage" else RED
        bx0, by0, bx1, by1 = BIN
        front[by0:by1, bx0:bx1] = (20, 20, 20)
        for i, o in enumerate(self.on("reject")):
            x, y = bx0 + 8 + 64 * (i % 3), by0 + 8 + 64 * (i // 3 % 2)
            front[y:y + 60, x:x + 60] = KRAFT if o["kind"] != "garbage" else RED
        if self.hand:
            front[360:520, 520:760] = (235, 190, 170)
        o = self.zone
        if o:
            x, y = ZONE
            front[y - 60:y + 60, x - 60:x + 60] = RED if o["kind"] == "garbage" else KRAFT
            if o["code"]:
                front[y - 35:y + 35, x - 35:x + 35] = qr_image(o["code"])
        wrist = {k: np.full((480, 640, 3), 128, np.uint8) for k in ("left_wrist", "right_wrist")}
        ee = {"position": [0, 0, 0], "rotation_rpy": [0, 0, 0]}
        return ({"front": front, **wrist},
                {"left_ee": {**ee, "gripper": 0.2}, "right_ee": {**ee, "gripper": self.grip}})

    def apply(self, instruction):
        self.hand = False
        t = (instruction or "").lower()
        kind = "REJECT" if "black bin" in t else "PUSH" if "push" in t else "PLACE" if " bin" in t else None
        self.n = self.n + 1 if kind and kind == self.kind else 1
        self.kind = kind
        o = self.zone
        if o and kind and self.n >= 2:
            o["places"] = o.get("places", 0) + (kind == "PLACE" and self.n == 2)  # one attempt per PLACE run
            if kind != "PLACE" or o["places"] > o["fail_places"]:
                self.at[id(o)] = "reject" if kind == "REJECT" else "blue" if "blue" in t else "red"
                self.zone = None
        if self.zone is None and self.queue and not kind:
            self.waited += 1
            if self.waited >= self.gap:
                self.zone, self.waited, self.hand = self.queue.pop(0), 0, True
                self.log.append(self.zone)
                self.at[id(self.zone)] = "zone"
