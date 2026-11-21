"""Harness v6: review bug fixes, arm roles, SPIN, push-path feasibility, time budget, calibration test mode."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from labelloop import look
from labelloop.brain.state_machine import StateMachine
from labelloop.config import load_yaml
from labelloop.eval.runlog import RunLog
from labelloop.harness import Harness
from labelloop.perception import scene
from labelloop.perception.qr import Confirmer
from labelloop.pose import interp_rpy, rpy_to_matrix, sim_to_wallx, wallx_to_sim
from labelloop.robots.mock import A, B, UNKNOWN, ZONE, FakeScene, garbage, nolabel, parcel, qr_image
from labelloop.wallx_client import move_response
from test_look import urdf_fk

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim" / "qr_sort_task"))
import build_task as bt  # noqa: E402

ROOT = Path(__file__).parent.parent
BASE = {**load_yaml(ROOT / "configs/task_qr_sort.yaml"), "reject_mode": "tray"}  # old tray flow; floor_right: test_v9
FLOOR = load_yaml(ROOT / "configs/task_qr_sort.yaml")
ROBOT = load_yaml(ROOT / "configs/ex001_6r.yaml")
BIG = {"case_s": 1e9, "deliver_reserve_s": 0, "wallx_call_s": 11.0, "scripted_call_s": 2.2}
NOSLIDE = {**BASE, "slide": {**BASE["slide"], "enabled": False}, "show": {**BASE["show"], "enabled": False}}  # the hand-off (SLIDE) has its own tests in test_v7, SHOW in test_v8
NOSPIN = {**NOSLIDE, "spin": {"enabled": False}, "budget": BIG}
GAIN = 0.05
LC = {**BASE["look"], **BASE["spin"]}


def synth_mask(c, yaw_deg, half, wh=(1280, 720), h=None):
    """Head-camera change mask of a box (centre xy root, yaw, footprint half-extents m, height h = parcel_h): the filled silhouette of its 8 corners."""
    return look.box_mask(c, np.radians(yaw_deg), half[0], half[1], LC["parcel_h"] if h is None else h, wh)


def fit_of(c, yaw_deg, half=(0.06, 0.04), h=None):
    return look.fit_parcel(synth_mask(c, yaw_deg, half, h=h), (1280, 720), LC)


def ev(presence=0.06, kroi=0.08, codes=(), bins=None, **kw):
    return {"presence": presence, "kroi": kroi, "codes": set(codes), "bins": {"blue": 0.0, "red": 0.0, "black": 0.0, **(bins or {})}, **kw}


def scan(sm, **kw):  # IDLE -> SCAN (two stable calls)
    sm.step(ev(**kw)), sm.step(ev(**kw))
    assert sm.state == "SCAN"


# --- review bug fixes -------------------------------------------------------------------------------------------------------------
def test_footprint_box_is_full_image_pixels_at_any_resolution_and_implausible_boxes_fall_back():
    w, c = 0.06, np.array([0.30, 0.05])  # a 12 cm square parcel; its top face at table + target_h
    corners = [(*(c + [sx * w, sy * w]), look.TABLE_Z + BASE["look"]["target_h"]) for sx in (-1, 1) for sy in (-1, 1)]
    for wh in ((1280, 720), (640, 480)):
        px = np.array([look.table_to_pixel(p, wh) for p in corners])
        box = (*px.min(0), *px.max(0))
        assert look.footprint(box, wh, BASE["look"]) == pytest.approx(w * np.sqrt(2), rel=0.25)  # half the diagonal: conservative
    assert look.footprint(None, (1280, 720), BASE["look"]) == BASE["look"]["hw"]
    assert look.footprint((0, 0, 1279, 719), (1280, 720), BASE["look"]) == BASE["look"]["hw"]  # whole frame = stale reference


def test_harness_hw_falls_back_on_a_stale_reference_and_uses_the_mask_otherwise():
    h, sc = Harness(NOSPIN), FakeScene([parcel(A)])
    empty = sc.observe()[0]["front"]
    cfg = NOSPIN["scene"]
    h.ref = scene.zone_ref(empty, cfg)
    sc.zone = parcel(None)
    sc.zone["code"] = None
    front = sc.observe()[0]["front"]
    hw = h.parcel_hw(front, cfg, NOSPIN["look"])
    assert 0.04 <= hw <= 0.25 and hw != NOSPIN["look"]["hw"]
    stale = np.zeros_like(front)  # reference taken before the parcel / lighting changed: everything differs
    assert h.parcel_hw(stale, cfg, NOSPIN["look"]) == NOSPIN["look"]["hw"]


def test_inspect_retries_are_bounded_by_tries_and_zero_tries_skips_inspect():
    cfg = {**NOSPIN, "tries": {**BASE["tries"], "INSPECT": 2}}
    sm = StateMachine(cfg)
    scan(sm)
    n = 0
    while sm.state in ("SCAN", "INSPECT"):
        sm.step(ev())
        n += sm.state == "INSPECT"
    assert sm.tries["INSPECT"] == 2 and sm.reason == "no_label" and n == 2 * cfg["max_calls"]["INSPECT"]
    sm = StateMachine({**NOSPIN, "tries": {**BASE["tries"], "INSPECT": 0}})
    scan(sm)
    while sm.state == "SCAN":
        sm.step(ev())
    assert sm.state == "REJECT" and sm.reason == "unread" and "INSPECT" not in sm.tries  # only the top was seen: never "no_label"


def test_verify_retries_are_bounded_over_the_whole_plan_including_handover():
    cfg = {**NOSPIN, "handover": True, "tries": {**BASE["tries"], "HANDOVER": 2}}
    sm = StateMachine(cfg)
    scan(sm)
    sm.step(ev(codes=[A]))
    entered = [t[1] for t in sm.transitions]
    for _ in range(2000):
        if sm.state == "IDLE":
            break
        sm.step(ev())  # the parcel never leaves
        entered += [t[1] for t in sm.transitions]
    assert sm.state == "IDLE" and sm.results[0]["outcome"] == "failed"
    assert entered.count("HANDOVER") == 2 and entered.count("PLACE") == 2  # HANDOVER's 2 tries stop the retries, not PLACE's 3


def test_scripted_targets_share_the_wallx_state_frame_round_trip_through_pose():
    lc = LC
    for c in ((0.30, 0.05), (0.30, -0.05)):
        plan = look.spin_plan(fit_of(c, 0), lc, None)
        arm = plan["arm"]
        home = sim_to_wallx(np.zeros(7), np.zeros(7), ROBOT)
        for (ee, rpy), tip in zip(plan["wps"], plan["tips"]):
            rel = wallx_to_sim(move_response("e", home, {arm: {"position": ee.tolist(), "rotation_rpy": rpy.tolist()}})["actions"], ROBOT)[-1]
            row = rel[:7] if arm == "left_ee" else rel[7:]
            back = sim_to_wallx(row, np.zeros(7), ROBOT)["left_ee"] if arm == "left_ee" else sim_to_wallx(np.zeros(7), row, ROBOT)["right_ee"]
            assert np.allclose(back["position"], ee, atol=1e-5)  # sim wire -> WALLX absolute returns the arm-frame target
            assert np.allclose(rpy_to_matrix(back["rotation_rpy"]), rpy_to_matrix(rpy), atol=1e-5)
            assert np.allclose(look.tip_of(arm, back["position"], back["rotation_rpy"], lc), tip, atol=1e-5)  # ... whose tip is the root point
    # the LOOK via-point: root z = table + 0.30 once the arm-frame target z is shifted back by the arm base
    via = look.to_arm("left_ee", [0, 0, look.TABLE_Z + 0.30], lc)
    assert via[2] + lc["arm_z"] == pytest.approx(look.TABLE_Z + 0.30)


def test_qr_confirmation_is_per_camera_per_parcel_and_a_valid_decode_beats_the_garbage_test():
    c = Confirmer(2)
    assert c.update({("left_wrist", A)}) == set() and c.update({("right_wrist", A)}) == set()  # same text, two different cameras
    assert c.update({("right_wrist", A)}) == {("right_wrist", A)}
    h, sc = Harness(NOSPIN), FakeScene([])
    h.confirmer.counts = {("front", A): 1}
    h.step(*sc.observe())  # an IDLE call: the next parcel starts with no QR history
    assert h.confirmer.counts == {}
    sm = StateMachine(NOSPIN)
    scan(sm)
    sm.step(ev(kroi=0.0, codes=[A]))  # label seen but the kraft test says garbage: the valid code wins
    assert sm.reason == "routed" and sm.lane == "A"
    sm = StateMachine(NOSPIN)
    scan(sm, kroi=0.0)
    sm.step(ev(kroi=0.0))
    assert sm.state == "SCAN"
    sm.step(ev(kroi=0.0))
    assert sm.reason is None and sm.state == "SCAN"  # no colour shortcut: a low kraft fraction never rejects, the search goes on
    sm = StateMachine(NOSPIN)
    scan(sm, kroi=0.0)
    sm.step(ev(kroi=0.0, raw=[A]))  # a raw valid decode on the first SCAN frame (count 1, unconfirmed): wait for its confirmation frame
    assert sm.state == "SCAN"
    sm.step(ev(kroi=0.0, raw=[A]))
    assert sm.state == "SCAN"  # still being seen: never garbage
    sm.step(ev(kroi=0.0, codes=[A], raw=[A]))
    assert sm.reason == "routed" and sm.lane == "A"
    h = Harness(NOSPIN)  # SCAN: head ROI codes only; a wrist camera's code does not count
    h.sm.state = "SCAN"
    imgs, st = FakeScene([]).observe()
    imgs["left_wrist"][100:380, 180:460] = qr_image(A, 280)
    h.step(imgs, st)
    assert h.confirmer.counts == {}
    h = Harness(NOSPIN)
    h.sm.state = "INSPECT"  # end to end: a wrist-only code in one camera, then another camera, never confirms
    imgs, st = FakeScene([]).observe()
    for cam in ("left_wrist", "right_wrist", "left_wrist"):
        imgs["left_wrist"][:] = imgs["right_wrist"][:] = 128
        imgs[cam][100:380, 180:460] = qr_image(A, 280)
        h.step(imgs, st)
        assert h.confirmer.counts.get((cam, A)) == 1


# --- arm roles ----------------------------------------------------------------------------------------------------------------
def test_arm_roles_without_handover_and_the_handover_path_when_enabled():
    for code, want in ((A, "put the block into the blue bin with the left arm"), (B, "put the block into the red bin with the right arm"),
                       (UNKNOWN, "put the block into the black bin with the right arm")):
        sm = StateMachine(NOSPIN)
        scan(sm)
        assert sm.step(ev(codes=[code])) == want and sm.plan == ["PLACE" if code != UNKNOWN else "REJECT"]
    sm = StateMachine(NOSPIN)
    scan(sm, presence=0.4)
    assert sm.step(ev(0.4, codes=[A])) == "pick up the block with both arms and put it into the blue bin"  # LIFT2 unchanged
    sm = StateMachine({**NOSPIN, "handover": True})
    scan(sm)
    assert sm.step(ev(codes=[A])) == "hand the block from the left arm to the right arm" and sm.plan == ["HANDOVER", "PLACE"]
    while sm.state == "HANDOVER":
        out = sm.step(ev())
    assert out == "put the block into the blue bin with the right arm"


# --- SPIN in the mock ---------------------------------------------------------------------------------------------------------
def run_spin(objs, cfg=NOSLIDE, runlog=None):
    h, sc, sent = Harness({**cfg, "max_parcels": 1}, runlog=runlog), FakeScene(objs), []
    h.entered = []
    for _ in range(600):
        if h.done:
            break
        instr = h.step(*sc.observe())
        sent.append((h.sm.state, instr))
        h.entered += [t[1] for t in h.sm.transitions]
        if h.spin_index == 3:  # the push waypoint at the parcel's far end reached: the parcel has yawed once more
            sc.push()
        sc.apply(instr)
        h.log_reply(0.01)
    return h, sc, sent


def test_spin_pushes_until_the_label_shows_then_routes_with_the_lane_arm(tmp_path):
    h, sc, sent = run_spin([parcel(A, spin=2)], runlog=RunLog(str(tmp_path / "r.jsonl")))
    states = [s for i, (s, _) in enumerate(sent) if not i or s != sent[i - 1][0]]
    assert states[:5] == ["IDLE", "SCAN", "SPIN", "PLACE", "HOME"] and h.entered.count("SPIN") == 2 and sc.zone is None
    assert sc.at[id(sc.log[0])] == "blue" and h.sm.results[0]["code"] == A and "INSPECT" not in states
    assert [i for s, i in sent if s == "PLACE"][0] == "put the block into the blue bin with the left arm"
    assert all(i is None for s, i in sent if s == "SPIN")  # no WALLX call during the push or the re-read
    lines = [json.loads(l) for l in (tmp_path / "r.jsonl").read_text().splitlines()]
    spent = [l["spent_s"] for l in lines if "spent_s" in l]
    assert max(spent) > 5 * BASE["budget"]["scripted_call_s"] and any(l.get("spin") for l in lines)


def test_spin_never_shows_a_label_then_unread_reject_with_right_arm():
    h, sc, sent = run_spin([parcel(A, spin=99)])
    assert h.entered.count("SPIN") == BASE["spin"]["tries"] and h.sm.results[0]["reason"] == "unread"  # the bottom was never seen
    assert sc.at[id(sc.log[0])] == "black" and "INSPECT" not in [s for s, _ in sent]
    assert [i for s, i in sent if s == "REJECT"][0] == "put the block into the black bin with the right arm"
    assert h.spin is not None  # the parcel at ZONE is reachable: the pushes really were planned


def test_spin_inspect_after_spin_and_big_parcels_search_too():
    h, sc, sent = run_spin([nolabel()], {**NOSLIDE, "tries": {**BASE["tries"], "INSPECT": 1}})
    ss = [s for i, (s, _) in enumerate(sent) if not i or s != sent[i - 1][0]]
    assert ss[:5] == ["IDLE", "SCAN", "SPIN", "INSPECT", "REJECT"]
    h, sc, sent = run_spin([parcel(A, big=True, hidden=99)])  # big parcels search too (no skip); INSPECT tries 0 -> unread reject
    assert "SPIN" in [s for s, _ in sent] and h.sm.results[0]["reason"] == "unread"
    h, sc, sent = run_spin([parcel(A, big=True, hidden=99)], {**NOSLIDE, "tries": {**BASE["tries"], "INSPECT": 1}})
    assert [i for s, i in sent if s == "INSPECT"][0] == "lift the block with both arms and turn it over"


def test_unreachable_parcel_spin_tries_are_skipped_quickly_to_reject():
    h = Harness({**NOSLIDE, "max_parcels": 1})
    h.ref = np.zeros((10, 10), np.uint8)
    sm, spin_cfg = h.sm, {**BASE["look"], **BASE["spin"]}
    assert look.spin_plan({"c": np.array([1.5, 0.0]), "axes": np.eye(2), "half": np.array([0.06, 0.04])}, spin_cfg) is None
    scan(sm)
    sm.step(ev())
    while sm.state == "SCAN":
        sm.step(ev())
    assert sm.state == "SPIN"
    n = 0
    while sm.state == "SPIN":
        sm.step(ev(spin_push=0, spin_calls=1))  # what the harness sends when no push is feasible
        n += 1
    assert n == BASE["spin"]["tries"] and sm.state == "REJECT" and sm.reason == "unread"


# --- push paths ---------------------------------------------------------------------------------------------------------------
def test_spin_plans_are_ik_feasible_along_the_whole_commanded_trajectory_through_the_urdf_chain():
    home = (np.array([0.1185, 0, 0.181]), np.zeros(3))
    for c, yaw in (((0.30, 0.05), 0), ((0.30, -0.05), 30), ((0.32, 0.10), 60)):
        fit = fit_of(c, yaw)
        plan = look.spin_plan(fit, LC)
        assert plan and len(plan["wps"]) == 6
        arm, tips, P, n = plan["arm"], plan["tips"], plan["contact"], plan["normal"]
        assert np.allclose(tips[1:5, 2], plan["tip_z"]) and 0.015 <= plan["tip_z"] - look.TABLE_Z <= 0.03 and np.allclose(tips[[0, 5], 2], look.TABLE_Z + LC["approach_h"])
        assert np.allclose(tips[1, :2], P[:2] + LC["out"] * n) and np.allclose(tips[3, :2], P[:2] - LC["push_len"] * n)  # in and out along the normal
        assert n @ (look.frame(arm, LC)[:2] - fit["c"]) > 0  # a face that faces the arm
        # torque of the inward push (F = -n at P) turns the parcel in the planned direction
        r, F = P - fit["c"], -n
        assert np.sign(r[0] * F[1] - r[1] * F[0]) == plan["sigma"]
        poses = [home, *plan["wps"], home]
        q = None
        for (p0, r0), (p1, r1) in zip(poses[:-1], poses[1:]):  # the commanded legs through the URDF chain, every 2 cm
            for u in np.linspace(0, 1, int(np.ceil(np.linalg.norm(p1 - p0) / 0.02)) + 1)[1:]:
                p, rp = p0 + u * (p1 - p0), interp_rpy(r0, r1, u)
                q, e, _ = look.ik(p, rp, tries=1, q0=np.zeros(6) if q is None else q)
                T = urdf_fk(q)  # raises on a joint-limit violation
                assert np.linalg.norm(T[:3, 3] - p) < 0.011
                assert (T[:3, 3] + T[:3, :3] @ look.TCP + look.frame(arm, LC))[2] >= look.TABLE_Z + 0.01
    # the next try keeps turning the same way, on a face facing the arm that is now in front
    first = look.spin_plan(fit_of((0.30, 0.0), 0), LC)
    again = look.spin_plan(fit_of((0.30, 0.0), 20), LC, first["sigma"])
    assert again and again["sigma"] == first["sigma"]


def test_commanded_ok_checks_intermediate_samples_not_only_the_waypoints():
    a, b = (look.reach("left_ee", p, np.radians(75), LC) for p in ((0.28, 0.1, 0.1), (0.36, 0.1, 0.1)))
    poses, seen = [a[:2], b[:2]], []

    def block_mid(leg, tip):
        seen.append(tip[0])
        return not 0.31 < tip[0] < 0.33  # both endpoints are fine, the middle is not

    assert look.commanded_ok("left_ee", poses, LC)
    assert not look.commanded_ok("left_ee", poses, LC, block_mid) and any(0.31 < x < 0.33 for x in seen) and 0.28 < seen[0] < 0.36
    assert look.reach("left_ee", [0.30, 0.0, 0.015], np.pi / 2, LC) is None  # fingertip under table + 0.01: refused
    assert not look.commanded_ok("left_ee", [a[:2], (np.array([0.9, 0.0, 0.1]), a[1])], LC)  # a transit leg that leaves the workspace


def test_planned_contact_lies_on_the_fitted_face_of_a_rotated_box_and_implausible_fits_refuse():
    for yaw in (0, 30):
        for c in ((0.30, 0.0), (0.30, 0.05)):
            fit = fit_of(c, yaw, (0.06, 0.04))
            assert fit and np.allclose(sorted(fit["half"]), (0.04, 0.06), atol=0.006) and np.allclose(fit["c"], c, atol=0.006)
            d = (fit["yaw"] - np.radians(yaw)) % (np.pi / 2)
            assert min(d, np.pi / 2 - d) < 0.04
            plan = look.spin_plan(fit, LC)
            assert plan, (yaw, c)
            P, n = plan["contact"][:2] - fit["c"], plan["normal"]
            i = int(np.argmax(np.abs(fit["axes"] @ n)))
            assert abs(P @ n - fit["half"][i]) < 0.005  # on the face plane
            assert abs(P @ fit["axes"][1 - i]) <= fit["half"][1 - i] + 0.005  # within the face
            assert abs(P @ fit["axes"][1 - i]) == pytest.approx(LC["reach_frac"] * fit["half"][1 - i], abs=0.005)  # near one end
    for half in ((0.01, 0.01), (0.03, 0.2), (0.3, 0.3)):  # tiny, needle, huge
        assert fit_of((0.30, 0.0), 0, half) is None
    assert look.fit_parcel(np.zeros((720, 1280), np.uint8), (1280, 720), LC) is None


# --- budget -------------------------------------------------------------------------------------------------------------------
def test_budget_is_per_case_counts_idle_and_admits_a_search_try_only_if_it_leaves_the_reserve():
    mc = {**BASE["max_calls"], "INSPECT": 3}  # an INSPECT try costs 3 x 11 s
    mk = lambda case_s, **kw: StateMachine({**NOSPIN, "budget": {**BIG, "case_s": case_s, "deliver_reserve_s": 60.0, **kw}, "max_calls": mc,
                                            "tries": {**BASE["tries"], "INSPECT": 3}})
    sm = mk(1000.0)
    scan(sm)
    assert sm.spent == pytest.approx(2 * 2.2)  # the two IDLE calls count
    for _ in range(4):
        sm.step(ev())  # SCAN: scripted calls, 2.2 s each
    assert sm.spent == pytest.approx(6 * 2.2) and sm.state == "SCAN"
    sm.calls = 10**3
    sm.step(ev())
    assert sm.state == "INSPECT"
    spent = sm.spent
    while sm.state == "INSPECT":
        sm.step(ev())
    assert sm.tries["INSPECT"] == 3 and sm.reason == "no_label" and sm.spent > spent + 8 * 11  # the case budget runs through the parcel
    sm = mk(100.0)  # 100 - 13 spent - 33 (try) - 4.4 (home) = 49.6 < 60 reserve: not admitted
    scan(sm)
    sm.calls = 10**3
    sm.step(ev())
    assert sm.state == "REJECT" and sm.reason == "unread" and "INSPECT" not in sm.tries  # the search ran out of budget before every face was seen
    sm = StateMachine({**BASE, "budget": {**BIG, "case_s": 90.0, "deliver_reserve_s": 60.0}, "tries": {**BASE["tries"], "INSPECT": 0}})
    scan(sm)
    sm.calls = 10**3  # a SPIN try (12 x 2.2 s) + home + reserve does not fit either
    sm.step(ev())
    assert sm.state == "REJECT" and sm.reason == "unread" and "SPIN" not in sm.tries
    sm = mk(1000.0, sim_case_s=100.0)  # the sim's own episode end caps it
    assert sm.remaining() == 100.0


def test_spent_carries_over_parcels_and_no_wallx_step_is_entered_without_time_for_a_call():
    sm = StateMachine({**NOSPIN, "budget": {**BIG, "case_s": 400.0}})
    scan(sm)
    sm.step(ev(codes=[A]))
    while sm.state != "IDLE":
        sm.step(ev(bins={"blue": GAIN}, presence=0.0) if sm.state in ("PLACE", "VERIFY") else ev(presence=0.0))
    carried = sm.spent
    assert sm.parcels == 1 and carried > 3 * 11.0 and sm.remaining() == pytest.approx(400.0 - carried)  # not reset per parcel
    sm = StateMachine({**NOSPIN, "budget": {**BIG, "case_s": 12.0}})
    scan(sm)
    sm.step(ev(codes=[A]))
    assert sm.state == "HOME" and sm.transitions[-1][2] == "budget"  # 7.6 s left < one 11 s call


def test_budget_stops_a_placement_that_would_exceed_it_and_does_not_retry():
    b = {"case_s": 60.0, "deliver_reserve_s": 20.0, "wallx_call_s": 11.0, "scripted_call_s": 2.2}
    sm = StateMachine({**NOSPIN, "budget": b})
    scan(sm)
    sm.step(ev(codes=[A]))
    assert sm.state == "PLACE"
    n, entered = 0, ["PLACE"]
    while sm.state == "PLACE":
        sm.step(ev())
        n += 1
    assert n < NOSPIN["max_calls"]["PLACE"] and sm.transitions[-1][2] == "budget" and sm.spent + b["wallx_call_s"] > b["case_s"]
    for _ in range(50):
        sm.step(ev())
        entered += [t[1] for t in sm.transitions]
        if sm.state == "IDLE":
            break
    assert sm.state == "IDLE" and sm.results[0]["outcome"] == "failed" and entered.count("PLACE") == 1  # no retry: no time left


# --- calibration test mode ----------------------------------------------------------------------------------------------------
def test_table_to_pixel_inverts_pixel_to_table():
    rng = np.random.default_rng(1)
    for wh in ((1280, 720), (640, 480), (640, 360)):
        for _ in range(20):
            u, v, z = rng.uniform(100, wh[0] - 100), rng.uniform(wh[1] * 0.3, wh[1] - 20), rng.uniform(0.0, 0.2)
            p = look.pixel_to_table(u, v, wh, z)
            assert np.allclose(look.table_to_pixel(p, wh), (u, v), atol=1e-6)


def run_calib(cfg, tmp_path, extra=10):
    h, sc = Harness({**cfg, "calib_test": True}, runlog=RunLog(str(tmp_path / "c.jsonl"))), FakeScene([parcel(A)])
    replies = []
    for i in range(500):
        imgs, st = sc.observe()
        instr = h.step(imgs, st)
        assert instr is None  # never a WALLX instruction, even after the schedule
        replies.append((h.cal_cur, h.scripted_reply("e", st)))
        assert replies[-1][1] is not None
        if i >= len(h.calib) + extra:
            break
    return h, replies, [json.loads(l) for l in (tmp_path / "c.jsonl").read_text().splitlines()]


def test_calib_mode_never_asks_wallx_commands_only_feasible_points_logs_pixels_and_latches_still(tmp_path):
    h, replies, lines = run_calib(BASE, tmp_path)
    lc = BASE["look"]
    cal = [l["calib"] for l in lines if "calib" in l]
    done = [c for c in cal if "measured_ee" in c]
    assert h.sm.calls == 0 and h.sm.parcels == 0 and h.sm.state == "IDLE" and h.state_name == "CALIB"  # the state machine never ran
    for c in cal:
        assert np.allclose(c["tip_px"], look.table_to_pixel(c["tip_root"], (1280, 720)), atol=1e-6)  # the commanded tip's expected pixel
        assert np.allclose(c["floor_px"], look.table_to_pixel([*c["tip_root"][:2], look.TABLE_Z], (1280, 720)), atol=1e-6)  # ... and the table point below
        if not c["refused"]:
            assert c["ik_err"] < 0.01 and np.allclose(look.tip_of(c["arm"], c["ee"], c["rpy"], lc), c["tip_root"], atol=0.011)
    assert len(done) == sum(not c["refused"] for c in cal) and done  # a measured line per commanded target
    sched = [r for r in replies if "hold" not in r[0]]
    for e, r in sched:  # only feasible targets are ever commanded; the other arm stays home
        if "arm" in e:
            assert any(c["arm"] == e["arm"] and np.allclose(c["ee"], e["ee"]) for c in done)
            other = "right_ee" if e["arm"] == "left_ee" else "left_ee"
            assert np.allclose(r["actions"][other]["position"][-1], h.home[other]["position"])
        else:
            assert np.allclose(r["actions"]["left_ee"]["position"][-1], h.home["left_ee"]["position"])
    tail = replies[len(sched):]
    assert sched[-1][0].get("home") and len(tail) == 11 and all(e == {"hold": True} for e, _ in tail)  # then hold still, never sorts
    assert all(np.allclose(r["actions"]["left_ee"]["position"][-1], h.home["left_ee"]["position"]) for _, r in tail)


def test_all_points_refused_still_latches_hold_and_the_empty_case_builds(tmp_path):
    h, replies, _ = run_calib({**BASE, "calib": {**BASE["calib"], "points": [[1.5, 0.0]]}}, tmp_path)
    assert h.sm.state == "IDLE" and h.sm.parcels == 0 and replies[-1][0] == {"hold": True}
    c = bt.CASES[31]
    assert c["late"] == {} and c["timeout"] == 90
    assert [o["id"] for o in bt.scene(31, c)["objects"] if "qr_box" in o["semantic_tags"]] == []
    d = bt.dag(31, c)["graph_task_generator"]
    assert "event_component" not in d["definition"]["component_sections"]
    assert d["nodes"][0]["direct_task_generator"]["component_sections"]["termination_component"]["terms"]["success"]["_func"].endswith(".never")  # X2Real needs one


def test_calib_mode_refuses_infeasible_points(tmp_path):
    cfg = {**BASE, "calib": {**BASE["calib"], "points": [[1.5, 0.0], [0.35, 0.10]]}}
    h, replies, lines = run_calib(cfg, tmp_path)
    cal = [l["calib"] for l in lines if "calib" in l]
    assert [c["refused"] for c in cal if c["tip_root"][0] == 1.5] == [True, True]  # both arms refuse x = 1.5 m
    commanded = {tuple(np.round(e["ee"], 6)) for e, _ in replies if "ee" in e}
    assert all(np.linalg.norm(np.array(e) + look.frame("left_ee", BASE["look"])) < 0.8 for e in commanded)  # nothing near x = 1.5
    assert all(c["tip_root"][0] != 1.5 for c in cal if "measured_ee" in c)


def test_empty_views_log_a_runlog_warning(tmp_path, monkeypatch):
    h, sc = Harness({**NOSPIN, "look": {**BASE["look"], "enabled": True}}, runlog=RunLog(str(tmp_path / "w.jsonl"))), FakeScene([])
    imgs, _ = sc.observe()
    h.ref = scene.zone_ref(imgs["front"], NOSPIN["scene"])
    monkeypatch.setattr(look, "views", lambda *a, **k: [])
    h.plan_look(imgs["front"], NOSPIN["scene"])
    lines = [json.loads(l) for l in (tmp_path / "w.jsonl").read_text().splitlines()]
    assert any("no feasible view" in l.get("warning", "") for l in lines)
