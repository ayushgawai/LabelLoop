from pathlib import Path

import cv2

from labelloop.config import load_yaml
from labelloop.perception import scene

DATA = Path(__file__).parent / "data"
# tests/data frames are from the OLD sim props, so the zone is pinned to where the cube sat then; kraft / presence come from the live config.
CFG = {**load_yaml(Path(__file__).parent.parent / "configs" / "task_qr_sort.yaml")["scene"], "induction_roi": [0.46, 0.48, 0.84, 0.92]}


def frame(name, size=None):
    img = cv2.cvtColor(cv2.imread(str(DATA / name)), cv2.COLOR_BGR2RGB)
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA) if size else img


def test_presence_and_kraft_in_zone_at_both_resolutions():
    for size in (None, (640, 480)):
        empty, full = frame("c1_in_blue.jpg", size), frame("c1_start.jpg", size)  # cube on the table / in the bin
        ref = scene.zone_ref(empty, CFG)
        assert scene.presence(empty, ref, CFG) < CFG["presence_min"] < scene.presence(full, ref, CFG)
        assert scene.kraft_in_roi(full, CFG) > CFG["kraft_min"] > scene.kraft_in_roi(empty, CFG)
