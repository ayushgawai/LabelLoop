"""Harness v7: reach-based arm coordination: grasp-feasibility, SLIDE hand-off, closer-arm SPIN."""
import numpy as np
import pytest

from labelloop import look
from labelloop.harness import Harness
from labelloop.robots.mock import A, B, UNKNOWN, FakeScene, parcel
from test_v6 import BASE, LC, fit_of

SC = {**BASE["look"], **BASE["spin"], **BASE["grasp"], **BASE["slide"]}
NARROW = {**SC, "grip_max": 0.0975}


def run(objs, cfg=BASE):
    """Full episode in the mock; the scripted stroke displaces the parcel when its push waypoint at the end of the stroke is reached."""
    h, sc, sent, moved = Harness({**cfg, "max_parcels": 1}), FakeScene(objs), [], []
    h.entered, h.plans = [], []
    for _ in range(400):
        if h.done:
            break
        instr = h.step(*sc.observe())
        sent.append((h.sm.state, instr))
        h.entered += [t[1] for t in h.sm.transitions]
        if h.sm.state == "SLIDE" and h.slide and h.sm.calls == 3 and not any(h.slide is p for p in h.plans):
            h.plans.append(h.slide)
            sc.slide(h.slide)
        sc.apply(instr)
        h.log_reply(0.01)
    return h, sc, sent


def test_grasp_check_is_per_arm_and_respects_the_97mm_gripper():
    cases = {0.20: (True, False), 0.0: (True, True), -0.20: (False, True)}  # y of the parcel: (left can, right can)
    for y, want in cases.items():
        fit = fit_of((0.30, y), 0)
        assert tuple(bool(look.grasp_plan(a, fit, NARROW)) for a in ("left_ee", "right_ee")) == want
    f0 = fit_of((0.30, 0.0), 0)
    g = look.grasp_plan("left_ee", f0, NARROW)
    assert g["pre"][2] > g["grasp"][2] and g["grasp"][2] == pytest.approx(look.TABLE_Z + f0["h"] - SC["grasp_depth"])  # the grasp height follows the FITTED height
    # fingers close across the narrow side: tool y (-sin yaw, cos yaw) is parallel to it
    fit = fit_of((0.30, 0.0), 0)
    n = fit["axes"][int(np.argmin(fit["half"]))]
    assert abs(np.dot([-np.sin(g["yaw"]), np.cos(g["yaw"])], n)) > 0.99
    assert look.grasp_plan("left_ee", fit_of((0.30, 0.0), 0, half=(0.06, 0.046)), NARROW) is None  # 92 mm narrow side + margin > 97.5 mm
    assert look.grasp_plan("left_ee", fit_of((0.30, 0.0), 0, half=(0.06, 0.041)), NARROW) is not None
    assert look.grasp_plan("left_ee", fit_of((0.30, 0.0), 0, half=(0.07, 0.07)), NARROW) is None  # too wide both ways


def test_spin_prefers_the_closer_arm_when_both_can_push():
    for y, arm in ((0.08, "left_ee"), (-0.08, "right_ee")):
        assert look.spin_plan(fit_of((0.30, y), 0), LC)["arm"] == arm


def test_slide_plan_pushes_through_the_face_centre_toward_the_handoff_and_stays_off_bins_and_edges():
    fit = fit_of((0.30, 0.20), 0)
    p = look.slide_plan(fit, SC["handoff"]["right"], "right_ee", SC)
    assert p["arm"] == "left_ee" and p["dist"] == SC["stroke"]
    assert np.allclose(p["contact"], fit["c"] + fit["half"][int(np.argmax(np.abs(p["normal"] @ fit["axes"].T)))] * p["normal"], atol=1e-6)
    assert np.allclose(p["c_after"], fit["c"] - 0.08 * p["normal"]) and p["c_after"][1] < fit["c"][1]  # moved toward -y
    # (d) a stroke whose predicted footprint comes within 3 cm of a bin (red: x from 0.613) or the table edge is not made
    assert look.slide_plan(fit_of((0.50, 0.12), 0), (0.70, 0.12), "right_ee", SC) is None  # straight at the blue bin
    assert look.slide_plan(fit_of((0.22, 0.0), 0), (0.10, 0.0), "left_ee", SC) is None  # toward the near table edge (x 0.184)
    assert look.footprint_clear(np.array([0.30, 0.0]), fit_of((0.30, 0.0), 0), SC)
    assert not look.footprint_clear(np.array([0.60, 0.10]), fit_of((0.60, 0.10), 0), SC)  # x half 0.06: 0.66 > bin x0 (0.651) - 0.03 and y inside blue


def test_red_parcel_far_left_is_slid_by_the_left_arm_then_placed_by_the_right_arm():  # (a)
    h, sc, sent = run([parcel(B, xy=(0.30, 0.20))])
    assert sc.at[id(sc.log[0])] == "red" and h.sm.results[0]["outcome"] == "ok"
    assert h.plans and all(p["arm"] == "left_ee" for p in h.plans) and len(h.plans) <= SC["max_strokes"]
    assert [i for s, i in sent if s == "PLACE"][0] == "put the block into the red bin with the right arm"
    assert all(i is None for s, i in sent if s == "SLIDE")  # scripted: no WALLX call
    order = [t for t in h.entered if t in ("SLIDE", "PLACE")]
    assert order.index("PLACE") > order.index("SLIDE") and sc.log[0]["xy"][1] < 0.20


def test_blue_parcel_on_the_right_is_slid_by_the_right_arm_then_placed_by_the_left_arm():  # (b)
    h, sc, sent = run([parcel(A, xy=(0.30, -0.20))])
    assert sc.at[id(sc.log[0])] == "blue" and h.plans and all(p["arm"] == "right_ee" for p in h.plans)
    assert [i for s, i in sent if s == "PLACE"][0] == "put the block into the blue bin with the left arm"
    assert sc.log[0]["xy"][1] > -0.20


def test_no_slide_when_the_delivery_arm_can_grasp_directly_and_slide_failure_falls_back_to_the_delivery_arm():  # (c)
    h, sc, sent = run([parcel(B, xy=(0.30, 0.0))])
    assert not h.plans and "SLIDE" in h.entered and sc.at[id(sc.log[0])] == "red" and h.slide_info["why"] == "graspable"
    h, sc, sent = run([parcel(B, xy=(0.30, 0.20))], {**BASE, "slide": {**BASE["slide"], "max_strokes": 1}})  # one stroke only: place anyway
    assert len(h.plans) == 1 and sc.at[id(sc.log[0])] == "red"
    h, sc, sent = run([parcel(UNKNOWN, xy=(0.30, 0.20))])  # reject bin = right arm too
    assert h.plans and h.plans[0]["arm"] == "left_ee" and sc.at[id(sc.log[0])] == "black"


def test_too_wide_parcel_goes_to_lift2_else_reject():  # (e)
    wide = dict(xy=(0.30, 0.0), half=(0.06, 0.07))
    h, sc, sent = run([parcel(A, **wide)])
    assert h.sm.results[0]["reason"] == "too_wide" and not h.plans
    assert [i for s, i in sent if s == "LIFT2"][0] == "pick up the block with both arms and put it into the blue bin"
    h, sc, sent = run([parcel(A, **wide)], {**BASE, "tries": {**BASE["tries"], "LIFT2": 0}})
    assert h.sm.results[0]["reason"] == "too_wide" and sc.at[id(sc.log[0])] == "black" and "SLIDE" not in h.entered[2:]


def test_slide_is_admitted_only_within_the_case_budget():
    tight = {**BASE, "budget": {**BASE["budget"], "case_s": 140.0}}  # reserve 130: no stroke fits
    h, sc, sent = run([parcel(B, xy=(0.30, 0.20))], tight)
    assert "SLIDE" not in h.entered and not h.plans
