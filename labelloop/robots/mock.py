"""Scripted fake scene for offline tests: no simulator, no robot. Front image is 1280x720 so QR codes decode."""
import functools
import re

import numpy as np
import qrcode
from PIL import Image

A, B, C = "LL|ZONE-A|001", "LL|ZONE-B|002", "LL|ZONE-A|003"
BINS = {"orange": (385, 107, 583, 279), "blue": (703, 109, 907, 283)}  # x0, y0, x1, y1 as in the sim head camera
COLORS = {"orange": (225, 165, 40), "blue": (30, 100, 230)}
KRAFT = (165, 125, 85)
SLOTS = [200, 640, 1060]  # table x positions, one per image third (3 cubes max); y = 470
CENTRE = {"on the left": 213, "in the middle": 640, "on the right": 1067}


@functools.lru_cache(maxsize=None)
def qr_image(code, size=200):
    img = qrcode.make(code).convert("RGB").resize((size, size), Image.NEAREST)
    return np.asarray(img)


def cube(code=None, label="top"):
    """label: 'top' (front camera reads it), 'side' (only a wrist camera reads it, once picked), None (decoy)."""
    return {"code": code, "label": label if code else None, "at": "table", "x": None}


class FakeScene:
    """Fake WALLX world. The right arm does every grasp (hold = 0.46 plateau). It obeys the position and bin named
    in the instruction; wrong_bin=True makes it drop the cube into the other bin."""

    def __init__(self, cubes=None, wrong_bin=False):
        self.cubes = cubes if cubes is not None else [cube(A), cube(B, "side"), cube()]
        for i, c in enumerate(self.cubes):
            c["x"] = SLOTS[i]
        self.wrong_bin, self.kind, self.n, self.grip = wrong_bin, None, 0, 1.0
        self.bin_of, self.picks, self.carry = {}, 0, False  # code -> bin it was dropped into; number of pick-ups

    def at(self, where):
        return [c for c in self.cubes if c["at"] == where]

    def observe(self):
        front = np.full((720, 1280, 3), (225, 228, 226), np.uint8)
        front[:90] = (10, 60, 20)
        for color, (x0, y0, x1, y1) in BINS.items():
            front[y0:y1, x0:x1] = COLORS[color]
        for color, (x0, y0, x1, y1) in BINS.items():
            for i, c in enumerate(self.at(color)):
                cx = (x0 + x1) // 2 + 30 * (i % 2) - 15
                front[y1 - 100:y1 - 40, cx - 30:cx + 30] = KRAFT  # sorted cubes land label-down: no QR visible
        for c in self.at("table") + self.at("edge"):
            x, y = (40, 400) if c["at"] == "edge" else (c["x"], 470)
            front[y - 50:y + 50, x - 50:x + 50] = KRAFT
            if c["label"] == "top":
                front[y - 40:y + 40, x - 40:x + 40] = qr_image(c["code"], 80)
        wrist = {k: np.full((480, 640, 3), 128, np.uint8) for k in ("left_wrist", "right_wrist")}
        hand = self.at("hand")
        if hand and hand[0]["label"] == "side":
            wrist["right_wrist"][140:340, 220:420] = qr_image(hand[0]["code"], 200)
        ee = {"position": [0, 0, 0], "rotation_rpy": [0, 0, 0]}
        return ({"front": front, **wrist},
                {"left_ee": {**ee, "gripper": 0.2}, "right_ee": {**ee, "gripper": self.grip}})

    def apply(self, instruction):
        if instruction is None:
            return
        t = instruction.lower()
        kind = ("DONE" if "home" in t else "REJECT" if "edge" in t else "PLACE" if " bin" in t else "PICK")
        self.n = self.n + 1 if kind == self.kind else 1
        if self.n == 1:
            self.carry = bool(self.at("hand"))  # already holding when this instruction started
        self.kind = kind
        where = re.search(r"on the left|in the middle|on the right", t)
        bin_ = re.search(r"blue|orange", t)
        hand, table = self.at("hand"), self.at("table")
        if kind == "DONE":
            self.grip = 1.0
        elif not hand:  # approach, open, grasp the cube named by position at the 2nd call
            if kind != "REJECT" and table and self.n >= 2:
                target = min(table, key=lambda c: abs(c["x"] - CENTRE[where.group() if where else "in the middle"]))
                target["at"] = "hand"
                self.picks += 1
            self.grip = 0.46 if self.at("hand") else 1.0
        elif kind != "PICK" and self.n >= (2 if self.carry or kind == "REJECT" else 3):  # release
            self.grip = 1.0
            dest = "edge" if kind == "REJECT" else bin_.group()
            if kind == "PLACE" and self.wrong_bin:
                dest = "orange" if dest == "blue" else "blue"
            hand[0]["at"] = dest
            if kind == "PLACE":
                self.bin_of[hand[0]["code"]] = dest
        else:
            self.grip = 0.46


class FakeAgent:
    """Answers from scene ground truth."""

    def __init__(self, scene):
        self.scene = scene

    def ask(self, images, state_name, context):
        left = len(self.scene.at("table")) + len(self.scene.at("hand"))
        return {"done": False, "target": "", "bin_side": {"blue": "right", "orange": "left"},
                "unsorted_boxes": left, "reason": "fake"}
