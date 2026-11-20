from pathlib import Path

import cv2

from labelloop.config import load_yaml
from labelloop.perception import scene

DATA = Path(__file__).parent / "data"
# tests/data frames are from the OLD sim props (172 mm blue + orange bins), so the bin ranges are pinned to what those frames show.
# The kraft / zone settings come from the live config. Replace the frames (and this override) with head frames of the shoe boxes.
OLD_FRAME_BINS = {"blue": [[[95, 120, 80], [125, 255, 255]]], "orange": [[[10, 150, 150], [30, 255, 255]]]}
CFG = {**load_yaml(Path(__file__).parent.parent / "configs" / "task_qr_sort.yaml")["scene"], "bins": OLD_FRAME_BINS}


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


def test_live_bin_ranges_find_the_shoe_box_colours():
    """Live config ranges vs the colours sampled from the shoe-box photos: each lane bin is found where it is painted, the
    black reject box and the kraft / white linings are not mistaken for a lane bin, and red is found on both sides of H 0."""
    import sys
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim" / "qr_sort_task"))
    from props import BINS
    live = load_yaml(Path(__file__).parent.parent / "configs" / "task_qr_sort.yaml")["scene"]
    img = np.full((720, 1280, 3), (200, 200, 200), np.uint8)
    paint = {"bin_blue": (100, 200, 400, 500), "bin_red": (500, 200, 800, 500), "reject_bin": (900, 200, 1200, 500)}  # x0, y0, x1, y1
    for b, (x0, y0, x1, y1) in paint.items():
        img[y0:y1, x0:x1] = BINS[b]["outside"]
        img[y0 + 60:y1 - 60, x0 + 60:x1 - 60] = BINS[b]["inside"]  # the lining seen through the opening
    img[200:230, 500:800] = (224, 4, 8)  # red sampled at H 179 (the other side of the wrap)
    # the perfume box's light-blue foil (package 4) as photographed, bigger than the blue bin: it must not win the blue blob
    from props import PARCELS
    for i, rgb in enumerate([PARCELS["garbage"]["color"], (43, 51, 62), (49, 57, 70), (130, 153, 161), (92, 110, 117)]):
        img[540 + 30 * i:570 + 30 * i, 100:1200] = rgb
    img[520:540, 100:1200] = PARCELS["z10001"]["tape"]["rgb"]  # ... nor the mailer's blue tear band (package 3)
    found = scene.mats(img, live)
    assert set(found) == {"blue", "red"}
    near = lambda a, b: all(abs(u - v) <= 2 for u, v in zip(a, b))  # noqa: E731 (the even opening kernel shifts edges ~1 px)
    assert near(found["blue"], paint["bin_blue"]) and near(found["red"], paint["bin_red"])


def test_presence_and_kraft_in_zone_at_both_resolutions():
    for size in (None, (640, 480)):
        empty, full = frame("c1_in_blue.jpg", size), frame("c1_start.jpg", size)  # cube on the table / in the bin
        ref = scene.zone_ref(empty, CFG)
        assert scene.presence(empty, ref, CFG) < CFG["presence_min"] < scene.presence(full, ref, CFG)
        assert scene.kraft_in_roi(full, CFG) > CFG["kraft_min"] > scene.kraft_in_roi(empty, CFG)
