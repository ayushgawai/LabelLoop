"""Scripted fake scene for offline tests: no simulator, no robot."""
import functools

import numpy as np
import qrcode
from PIL import Image

A, B = "LL|ZONE-A|001", "LL|ZONE-B|002"


@functools.lru_cache(maxsize=None)
def qr_image(code, size=200):
    img = qrcode.make(code).convert("RGB").resize((size, size), Image.NEAREST)
    return np.asarray(img)


def paste(img, code, x, y, size=200):
    img[y:y + size, x:x + size] = qr_image(code, size)


class FakeScene:
    """Box A: top label (ZONE-A). Box B: side label (ZONE-B), readable only while presented. Box C: unlabeled."""

    def __init__(self):
        self.boxes = [{"code": A, "top": True, "side": False, "at": "table"},
                      {"code": B, "top": False, "side": True, "at": "table"},
                      {"code": None, "top": False, "side": False, "at": "table"}]
        self.grip, self.kw, self.n = 1.5, None, 0
        self.bin_of = {}  # code -> bin word seen in the PLACE instruction

    def at(self, where):
        return [b for b in self.boxes if b["at"] == where]

    def unsorted(self):
        return len(self.at("table")) + len(self.at("hand"))

    def observe(self):
        imgs = {k: np.full((480, 640, 3), 128, np.uint8) for k in ("front", "left_wrist", "right_wrist")}
        for i, b in enumerate(self.at("bin")):  # sorted boxes stay visible to the front camera
            if b["code"]:
                paste(imgs["front"], b["code"], 20 + 220 * (i % 2), 250, 160)
        table, hand = self.at("table"), self.at("hand")
        if self.kw == "LOOK" and self.n >= 2 and table and table[0]["top"]:
            paste(imgs["right_wrist"], table[0]["code"], 220, 140)
        if self.kw == "PRESENT" and self.n >= 3 and hand and hand[0]["side"]:
            paste(imgs["right_wrist"], hand[0]["code"], 220, 140)
        ee = {"position": [0, 0, 0], "rotation_rpy": [0, 0, 0]}
        return imgs, {"left_ee": {**ee, "gripper": self.grip}, "right_ee": {**ee, "gripper": 1.5}}

    def apply(self, instruction):
        if instruction is None:
            return
        kw = self._kw(instruction)
        self.n = self.n + 1 if kw == self.kw else 1
        self.kw = kw
        table, hand = self.at("table"), self.at("hand")
        if kw == "GRASP":
            self.grip = 1.5 if self.n < 2 else 0.35
            if self.n >= 2 and table:
                table[0]["at"] = "hand"
        elif kw in ("PLACE", "REJECT") and self.n >= 2:
            self.grip = 1.5
            if hand:
                hand[0]["at"] = "bin" if kw == "PLACE" else "left"
                if kw == "PLACE":
                    self.bin_of[hand[0]["code"]] = "blue" if "blue" in instruction else "orange"

    @staticmethod
    def _kw(instruction):
        t = instruction.lower()
        for kw, words in (("PLACE", ("bin",)), ("REJECT", ("edge", "far left")), ("DONE", ("home",)),
                          ("PRESENT", ("turn", "rotate", "different")), ("GRASP", ("pick up", "grasp"))):
            if any(w in t for w in words):
                return kw
        return "LOOK"


class FakeAgent:
    """Answers from scene ground truth."""

    def __init__(self, scene):
        self.scene = scene

    def ask(self, images, state_name, context):
        s = self.scene
        done = {"GRASP": bool(s.at("hand")), "PLACE": not s.at("hand"), "REJECT": not s.at("hand")}.get(state_name, False)
        return {"done": done, "target": "", "bin_side": {"blue": "left", "orange": "right"},
                "unsorted_boxes": s.unsorted(), "reason": "fake"}
