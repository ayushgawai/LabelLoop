from pathlib import Path

import cv2

from labelloop.config import load_yaml
from labelloop.perception import scene

DATA = Path(__file__).parent / "data"
CFG = load_yaml(Path(__file__).parent.parent / "configs" / "task_qr_sort.yaml")["scene"]


def frame(name, size=None):
    img = cv2.cvtColor(cv2.imread(str(DATA / name)), cv2.COLOR_BGR2RGB)
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA) if size else img


def test_mats_found_at_full_and_policy_resolution():
    for size in (None, (640, 480)):
        img = frame("c1_start.jpg", size)
        b = scene.mats(img, CFG)
        w = img.shape[1]
        assert b["orange"][2] < b["blue"][0] and 0.28 * w < b["orange"][0] and b["blue"][2] < 0.75 * w
        assert all(box[3] - box[1] > 0.15 * img.shape[0] for box in b.values())


def test_kraft_in_mat_rises_when_cube_lands_in_blue():
    start, after = frame("c1_start.jpg"), frame("c1_in_blue.jpg")
    k0 = scene.kraft_in_mats(start, scene.mats(start, CFG), CFG)
    k1 = scene.kraft_in_mats(after, scene.mats(after, CFG), CFG)
    assert k1["blue"] - k0["blue"] > CFG["mat_gain"] and k1["orange"] - k0["orange"] < CFG["mat_gain"]


def test_presence_and_kraft_in_zone_at_both_resolutions():
    for size in (None, (640, 480)):
        empty, full = frame("c1_in_blue.jpg", size), frame("c1_start.jpg", size)  # cube on the table / in the bin
        ref = scene.zone_ref(empty, CFG)
        assert scene.presence(empty, ref, CFG) < CFG["presence_min"] < scene.presence(full, ref, CFG)
        assert scene.kraft_in_roi(full, CFG) > CFG["kraft_min"] > scene.kraft_in_roi(empty, CFG)
