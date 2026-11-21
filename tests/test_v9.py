"""Harness v9: 3D box fit, grasp checks, joint-state seeding, code association (bins never read), whole-table detection, reject swipe, unread vs no_label."""
import numpy as np
import pytest

from labelloop import look
from labelloop.brain.state_machine import StateMachine
from labelloop.harness import Harness
from labelloop.robots.mock import A, B, BINS, FakeScene, nolabel, parcel, qr_image
from test_v6 import BASE, FLOOR, LC, ev, scan
from test_v8 import SC, run

SIZES = {"cube76": (0.038, 0.038, 0.076), "flat165": (0.0825, 0.051, 0.038), "flat267": (0.1335, 0.051, 0.064), "stand229": (0.1145, 0.0445, 0.152)}


@pytest.mark.parametrize("name", SIZES)
def test_3d_fit_recovers_size_height_and_yaw_from_synthetic_masks(name):
    hx, hy, h = SIZES[name]
    for yaw in (0, 30, 70, 110):
        f = look.fit_parcel(look.box_mask((0.30, 0.05), np.radians(yaw), hx, hy, h), (1280, 720), LC)
        assert f is not None and abs(f["h"] - h) < 0.02 and np.allclose(f["half"], sorted((hx, hy), reverse=True), atol=0.012) and np.allclose(f["c"], (0.30, 0.05), atol=0.008)
        if hx != hy:
            d = (f["yaw"] - np.radians(yaw)) % np.pi
            assert min(d, np.pi - d) < np.radians(5)


def test_config_bin_rois_are_the_projected_rotated_bin_footprints_with_walls():
    for k, (x0, x1, y0, y1) in {"blue": BASE["slide"]["bins"]["blue"], "red": BASE["slide"]["bins"]["red"]}.items():
        px = np.array([look.table_to_pixel((x, y, z)) for x in (x0, x1) for y in (y0, y1) for z in (look.TABLE_Z, look.TABLE_Z + 0.127)]) / [1280, 720]
        r = BASE["scene"]["bin_rois"][k]
        assert r[0] <= px[:, 0].min() + 0.011 and r[2] >= px[:, 0].max() - 0.011 and r[3] >= px[:, 1].max() - 0.011


def test_held_low_corner_form_and_homes_use_the_arm_frame_state_convention():
    h = Harness(BASE)
    h.home = {a: {"position": [0.1185, 0, 0.181], "rotation_rpy": [0, 0, 0], "gripper": 0.0} for a in look.SIGN}
    assert np.allclose(h.homes()["left_ee"][0], [0.1185, 0, 0.181]) and np.allclose(look.fk(np.zeros(6))[0], [0.1185, 0, 0.181], atol=1e-3)  # no frame offset added
    for fit in ({"c": np.zeros(2)},):
        g = look.grasp_plan("left_ee", look.fit_parcel(look.box_mask((0.30, 0.0), 0, 0.045, 0.038, 0.08), (1280, 720), LC), SC)
        assert g and abs(((g["yaw"] - look.grasp_plan("left_ee", look.fit_parcel(look.box_mask((0.30, 0.0), 0, 0.045, 0.038, 0.08), (1280, 720), LC), SC)["yaw"]) + np.pi) % (2 * np.pi) - np.pi) < 1e-9
    n = np.array([0.0, 1.0])  # the two jaw orientations are y0 and y0 + pi
    y0 = np.arctan2(-n[0], n[1])
    assert np.isclose(abs((float(np.angle(np.exp(1j * (y0 + np.pi)))) - y0 + np.pi) % (2 * np.pi) - np.pi), np.pi)


def test_failed_return_planning_never_releases_in_the_air(monkeypatch):
    fit = look.fit_parcel(look.box_mask((0.30, 0.0), 0, 0.038, 0.038, 0.08), (1280, 720), LC)
    p = look.show_plan("left_ee", fit, SC)
    real = look.commanded_ok
    monkeypatch.setattr(look, "commanded_ok", lambda *a, **k: None if k.get("ok_pose") else real(*a, **k))  # every checked return route fails
    s = look.set_down(p, "right_ee", len(p["calls"]), SC)
    rel = [c for c in s["calls"] if c["grip"] == SC["grip_open"]][0]
    assert s["where"] == "orig" and rel["tag"] == "release" and np.allclose(rel["ee"], p["calls"][2]["ee"])  # released at the grasp pose, on the table
    assert [c["tag"] for c in s["calls"]][-3:] == ["down", "release", "retreat"]


def test_missed_grasp_retries_once_then_goes_on_and_a_parcel_that_does_not_follow_the_test_lift_aborts():
    cube = dict(xy=(0.30, 0.0), half=(0.038, 0.038))
    h, sc, sent = run([parcel(B, hidden=99, **cube)], BASE, miss=True)
    assert h.entered.count("SHOW") == 2 and h.show_done == 3 and "SPIN" in h.entered
    h, sc, sent = run([parcel(B, hidden=99, **cube)], BASE, stuck=True)
    assert h.entered.count("SHOW") == 1 and h.show_info["grasp"] == "no_comotion" and h.sset["where"] == "orig" and h.show_done == 4
    assert h.sset["calls"][0]["tag"] == "down"  # back down to the table before the jaws open


def test_flat_parcel_needs_top_and_bottom_seen_for_no_label_else_unread():
    sm = StateMachine({**BASE, "spin": {"enabled": False}, "show": {**BASE["show"], "enabled": False}, "tries": {**BASE["tries"], "INSPECT": 0}})
    scan(sm)
    while sm.state == "SCAN":
        sm.step(ev())
    assert sm.reason == "unread" and any(f.startswith("unread") for f in sm.flags) and not sm.covered()
    sm = StateMachine({**BASE, "spin": {"enabled": False}, "show": {**BASE["show"], "enabled": False}, "tries": {**BASE["tries"], "INSPECT": 1}})
    scan(sm)
    while sm.state in ("SCAN", "INSPECT"):
        sm.step(ev())
    assert sm.reason == "no_label" and sm.covered()  # INSPECT turned it over: top + bottom
    sm.flat = False
    assert not sm.covered()  # a box also needs its 4 sides


def test_codes_of_parcels_in_the_bins_are_never_read_and_a_new_parcel_anywhere_on_the_table_routes_by_its_own_label():
    pre = {"blue": [B], "red": [A]}  # sorted parcels, labels up and in view: B (a red-lane text) in the blue bin, A in the red one
    for xy, code, lane in (((0.40, 0.25), A, "blue"), ((0.50, -0.30), B, "red"), ((0.30, 0.0), None, None)):  # (the unlabelled one is swiped off the table)
        sc = FakeScene([parcel(code, xy=xy, h=0.04, half=(0.05, 0.04)) if code else {**nolabel(), "xy": np.array(xy), "half": (0.05, 0.04)}], prefill=pre)
        h = Harness({**FLOOR, "max_parcels": 1})
        for _ in range(150):
            if h.done:
                break
            instr = h.step(*sc.observe())
            if h.sm.state == "SWIPE" and h.swipe and h.sm.calls == 3:
                sc.swipe(h.swipe)
            sc.apply(instr)
        r = h.sm.results[0]
        assert r["outcome"] == "ok" and r["code"] == code  # the new parcel's own label, never a bin's
        assert r["lane"] == {"blue": "A", "red": "B", None: None}[lane] and sc.at[id(sc.log[0])] == (lane or "floor")
        assert code or r["reason"] in ("unread", "no_label")


def test_reject_is_a_scripted_swipe_off_the_right_edge_and_the_old_tray_path_stays_behind_the_flag():
    fit = look.fit_parcel(look.box_mask((0.40, -0.30), 0, 0.05, 0.04, 0.04), (1280, 720), LC)
    sw = {**LC, "dir_min_dot": 0.5, **FLOOR["swipe"]}
    p = look.swipe_plan(fit, sw)
    assert p and p["normal"][1] > 0.9 and p["c_after"][1] < fit["c"][1] - 0.2 and p["arm"] == "right_ee"
    p2 = look.swipe_plan(look.fit_parcel(look.box_mask((0.40, -0.45), 0, 0.05, 0.04, 0.04), (1280, 720), LC), sw)
    assert p2 and p2["off"]
    sm = StateMachine({**FLOOR, "spin": {"enabled": False}, "show": {**FLOOR["show"], "enabled": False}})
    scan(sm)
    sm.step(ev(codes=["LL|BAD"]))
    assert sm.state == "SWIPE" and sm.plan == ["SWIPE"] and sm.instruction() is None
    sm = StateMachine({**BASE, "spin": {"enabled": False}, "show": {**BASE["show"], "enabled": False}})
    scan(sm)
    sm.step(ev(codes=["LL|BAD"]))
    assert sm.state in ("REJECT", "SLIDE")
