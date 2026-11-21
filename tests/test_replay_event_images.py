import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import qrcode

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import replay_event_images as r


def qr_img(text, n=140):
    return np.array(qrcode.make(text).convert("RGB").resize((n, n)))


def test_bin_codes_ignored_table_codes_accepted(tmp_path):
    ep = tmp_path / "ep1"
    ep.mkdir()
    empty = np.full((480, 640, 3), 90, np.uint8)
    cur = empty.copy()
    cur[10:150, 130:270] = qr_img("BINCODE")   # centre (200, 80): inside blue bin ROI
    cur[300:440, 380:520] = qr_img("TABLECODE")  # centre (450, 370): on the table
    for name, im in (("0000", empty), ("0001", cur)):
        cv2.imwrite(str(ep / f"{name}_front.jpg"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(ep / "0001_left_wrist.jpg"), cv2.cvtColor(qr_img("WRIST", 200), cv2.COLOR_RGB2BGR))
    rows = r.run(tmp_path)
    res = dict((c, x) for _, c, x in rows)["0001"]
    assert [t for t, _ in res["accepted"]] == ["TABLECODE"]
    assert [t for t, _ in res["ignored"]] == ["BINCODE"]
    assert res["frac"] > 0 and (ep / "annotated" / "0001_front.png").exists()


def write_episode(root, frames):
    ep = root / "ep1"
    ep.mkdir()
    for name, im in frames:
        cv2.imwrite(str(ep / f"{name}_front.jpg"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    return ep


def test_profile_overrides_match_the_yaml():
    y = r.load_yaml(r.CFG)
    real, sim = r.scene_cfg(profile="real"), r.scene_cfg(profile="sim")
    assert sim == y["scene"]
    assert {k: real[k] for k in ("bin_rois", "induction_roi", "whole_table")} == {k: y["profiles"]["real"][k] for k in ("bin_rois", "induction_roi", "whole_table")}
    assert real["presence_thr"] == y["scene"]["presence_thr"]  # the rest of the scene section is kept


def test_real_profile_uses_the_real_bin_regions(tmp_path):
    empty = np.full((480, 640, 3), 90, np.uint8)
    cur = empty.copy()
    cur[146:286, 314:454] = qr_img("REALBIN")  # centre (384, 216) = (0.60, 0.45): in the real red bin, on the table in sim
    cur[74:214, 474:614] = qr_img("SIMBIN")    # centre (544, 144) = (0.85, 0.30): in the sim black tray, on the table in real
    write_episode(tmp_path, (("0000", empty), ("0001", cur)))
    for prof, acc, ign in (("real", "SIMBIN", "REALBIN"), ("sim", "REALBIN", "SIMBIN")):
        res = dict((c, x) for _, c, x in r.run(tmp_path, profile=prof))["0001"]
        assert ([t for t, _ in res["accepted"]], [t for t, _ in res["ignored"]]) == ([acc], [ign]), prof


def test_grid_skipped_when_table_grid_is_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "labelloop.perception.table_grid", None)  # import fails as before Task 1 is merged
    ep = write_episode(tmp_path, (("0000", np.full((480, 640, 3), 90, np.uint8)),))
    rows = r.run(tmp_path, grid=True)
    assert rows[0][2]["quad"] is None and (ep / "annotated" / "0000_front.png").exists()
    assert "not available" in capsys.readouterr().err


def test_grid_overlay(tmp_path):
    pytest.importorskip("labelloop.perception.table_grid")
    corners = np.array([[330, 95], [975, 90], [1190, 600], [95, 615]], np.float64)  # far-left, far-right, near-right, near-left
    img = np.full((720, 1280, 3), (70, 72, 75), np.uint8)
    cv2.fillConvexPoly(img, corners.astype(np.int32), (46, 178, 150))
    ep = write_episode(tmp_path, (("0000", img),))
    plain = (r.run(tmp_path), cv2.imread(str(ep / "annotated" / "0000_front.png")))
    rows = r.run(tmp_path, grid=True)
    assert np.abs(rows[0][2]["quad"] - corners).max() < 4
    assert plain[0][0][2]["quad"] is None and (cv2.imread(str(ep / "annotated" / "0000_front.png")) != plain[1]).any()
