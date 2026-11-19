from pathlib import Path

import cv2

from labelloop.config import load_yaml
from labelloop.perception import qr, scene

DATA = Path(__file__).parent / "data"
CFG = load_yaml(Path(__file__).parent.parent / "configs" / "task_qr_sort.yaml")["scene"]


def frame(name, size=None):
    img = cv2.cvtColor(cv2.imread(str(DATA / name)), cv2.COLOR_BGR2RGB)
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA) if size else img


def test_bins_found_at_full_and_policy_resolution():
    for size in (None, (640, 480)):
        img = frame("c1_start.jpg", size)
        b = scene.bins(img, CFG)
        w = img.shape[1]
        assert b["orange"][2] < b["blue"][0] and 0.28 * w < b["orange"][0] and b["blue"][2] < 0.75 * w
        assert all(box[3] - box[1] > 0.15 * img.shape[0] for box in b.values())


def test_blocks_counted_c1_and_c6_and_cube_in_bin_excluded():
    for size in (None, (640, 480)):
        for name, n in (("c1_start.jpg", 1), ("c6_start.jpg", 4), ("c1_in_blue.jpg", 0)):
            img = frame(name, size)
            assert len(scene.blocks(img, scene.bins(img, CFG), CFG)) == n, (name, size)


def test_kraft_in_bin_rises_when_cube_lands_in_blue():
    start, after = frame("c1_start.jpg"), frame("c1_in_blue.jpg")
    k0 = scene.kraft_in_bins(start, scene.bins(start, CFG), CFG)
    k1 = scene.kraft_in_bins(after, scene.bins(after, CFG), CFG)
    assert k1["blue"] - k0["blue"] > CFG["bin_gain"] and k1["orange"] - k0["orange"] < CFG["bin_gain"]


def test_qr_located_on_sim_frames_and_where():
    (text, (x, y)), = qr.locate(frame("c1_start.jpg"))
    assert text == "LL|ZONE-A|001" and 700 < x < 800 and 350 < y < 420
    assert scene.where(x, 1280) == "in the middle"
    found = dict(qr.locate(frame("c6_start.jpg")))
    assert set(found) == {"LL|ZONE-A|001", "LL|ZONE-B|001"}
    assert scene.where(found["LL|ZONE-A|001"][0], 1280) == "on the left"
    assert scene.where(found["LL|ZONE-B|001"][0], 1280) == "on the right"
