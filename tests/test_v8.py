"""Harness v8: SHOW (scripted grasp-and-show label search) and its set-down. Planning is slow (IK), so show_plan is cached across the tests."""
import numpy as np
import pytest

from labelloop import look
from labelloop.harness import Harness
from labelloop.pose import interp_rpy
from labelloop.robots.mock import A, B, FakeScene, parcel
from test_look import urdf_fk
from test_v6 import BASE, fit_of

SC = {**BASE["look"], **BASE["spin"], **BASE["grasp"], **BASE["slide"], **BASE["show"]}
CUBE = dict(xy=(0.30, 0.0), half=(0.038, 0.038))  # a 76 mm cube
HOME = (np.array([0.1185, 0, 0.181]), np.zeros(3))
_plans, _real = {}, look.show_plan


@pytest.fixture(autouse=True, scope="module")
def cached_show_plan():
    def cached(arm, fit, cfg, home=None, grasp=None):
        k = (arm, tuple(np.round(fit["c"], 3)), tuple(np.round(fit["half"], 3)), round(fit["yaw"], 2))
        if k not in _plans:
            _plans[k] = _real(arm, fit, cfg, home, grasp)
        return _plans[k]

    look.show_plan = cached
    yield
    look.show_plan = _real


LIFT = {"testlift": 0.02, "lift": 0.08, "via": 0.08, "move": 0.08, "hold": 0.08}  # how high the mock draws the held parcel after each scripted call


def run(objs, cfg=BASE, replies=None, miss=False, stuck=False, prefill=None):
    """Full episode in the mock. The mock obeys the scripted SHOW: the head sees the face the harness holds toward it (a call's effect shows in
    the next image), and the set-down moves the parcel."""
    h, sc, sent = Harness({**cfg, "max_parcels": 1}), FakeScene(objs, prefill=prefill), []
    h.grip_log = []  # miss: the jaws close on nothing (gripper reads the command); stuck: the parcel does not follow the gripper
    h.entered = []
    for _ in range(500):
        if h.done:
            break
        imgs, state = sc.observe()
        instr = h.step(imgs, state)
        sent.append((h.sm.state, instr))
        h.entered += [t[1] for t in h.sm.transitions]
        if replies is not None and h.sm.state in ("SHOW", "SHOWSET"):
            replies.append((h.sm.state, h.sm.calls, h.scripted_reply("emb", state)))
        sc.show_face = None
        if h.sm.state != "SHOW":
            sc.grip_state.clear()
        if h.sm.state == "SHOWSET" and sc.zone:
            sc.zone["lift"] = 0.0
        if h.sm.state == "SHOW" and h.show and h.sm.calls < len(h.show["calls"]):
            c = h.show["calls"][h.sm.calls]
            held = sc.zone is not None and not miss and c["tag"] in ("close", *LIFT)
            sc.grip_state[h.show["arm"]] = 0.9 if held and c["grip"] == SC["grip_close"] else c["grip"]  # a parcel in the jaws stops them at ~0.9
            if sc.zone and held and not stuck:
                sc.zone["lift"] = LIFT.get(c["tag"], 0.0)
            sc.show_face = c.get("face")
            if sc.zone and sc.zone["face"] == "*second":
                sc.zone["face"] = h.show["faces"][1]
        if h.sm.state == "SHOWSET" and h.sm.calls == 0 and h.sset:
            sc.setdown(h.sset)
        sc.apply(instr)
        h.log_reply(0.01)
    return h, sc, sent


def drive(arm, calls, hold, start=HOME, held_from=4, q=None):
    """Replay `calls` [{ee, rpy}] through the URDF chain the way move_response commands them (every 2 cm): joint limits (urdf_fk raises),
    fingertip above the table, and, from leg `held_from` (4 = after the lift: poses[k] -> poses[k+1]), the hanging parcel's lowest point >= held_clear above the table."""
    q, poses, worst = np.zeros(6) if q is None else q, [start] + [(c["ee"], c["rpy"]) for c in calls], 9.0
    for k, ((p0, r0), (p1, r1)) in enumerate(zip(poses[:-1], poses[1:])):
        for u in np.linspace(0, 1, max(int(np.ceil(np.linalg.norm(p1 - p0) / 0.02)), int(np.ceil(np.linalg.norm(r1 - r0) / 0.15))) + 1)[1:]:
            p, r = p0 + u * (p1 - p0), interp_rpy(r0, r1, u)
            q, e, _ = look.ik(p, r, tries=1, q0=q)
            T = urdf_fk(q)
            assert np.linalg.norm(T[:3, 3] - p) < 0.011
            assert (T[:3, 3] + T[:3, :3] @ look.TCP + look.frame(arm, SC))[2] >= look.TABLE_Z + 0.01
            if k >= held_from:
                worst = min(worst, look.held_low(arm, p, r, hold, SC))
    return worst


def test_cube_show_plan_is_feasible_through_the_urdf_chain_keeps_the_parcel_clear_and_faces_the_head():
    fit = fit_of((0.30, 0.0), 0, (0.038, 0.038))
    assert max(len(look.show_plan(a, fit, SC)["faces"]) for a in ("left_ee", "right_ee")) >= 3  # honest floor: the better arm shows 3 of the 4 sides
    for arm in ("left_ee", "right_ee"):
        p = look.show_plan(arm, fit, SC)
        cs = p["calls"]
        assert p["arm"] == arm and len(p["faces"]) >= 2 and set(p["faces"]) <= {"z+", "y+", "z-", "y-"} and "bottom" in p["missing"]
        assert [c["tag"] for c in cs[:5]] == ["pre", "descend", "close", "testlift", "lift"] and len(cs) <= 18
        assert [c["grip"] for c in cs[:5]] == [SC["grip_open"], SC["grip_open"], SC["grip_close"], SC["grip_close"], SC["grip_close"]] and all(c["grip"] == SC["grip_close"] for c in cs[3:])
        assert cs[3]["ee"] [2] - cs[2]["ee"][2] == pytest.approx(SC["test_lift"], abs=1e-6) and cs[4]["rpy"].tolist() == cs[1]["rpy"].tolist()  # a 2 cm test lift first; no rotation
        assert all(c["q"] is not None for c in cs[:5])  # the joint state is carried from the start
        low = drive(arm, cs, p["hold"], held_from=5)  # from the lift on, also every transit between views
        assert low >= look.TABLE_Z + SC["held_clear"] - 1e-6
        P, d = p["P"], look.HEAD_POS - p["P"]
        d /= np.linalg.norm(d)
        u, v = look.table_to_pixel(P + [0, 0, -0.02])
        assert 0.25 * 1280 < u < 0.75 * 1280 and 0.25 * 720 < v < 0.8 * 720  # well inside the head image
        for c in cs[4:]:
            if c["tag"] == "hold":
                assert look.box_R(c["rpy"], p["hold"]) @ look.FACES[c["face"]] @ d >= SC["vis_min"] - 1e-6  # the face looks at the head
        assert [c["face"] for c in cs[4:] if c["tag"] == "hold"] == p["faces"]


def test_package1_is_too_wide_for_show_so_spin_runs():  # 229x89x152 standing at (0.30, 0.05): narrow side 89 mm > 97.5 - 10
    fit = fit_of((0.30, 0.05), 0, (0.1145, 0.0445))
    assert look.too_wide(fit, SC) and all(look.show_plan(a, fit, SC) is None for a in ("left_ee", "right_ee"))
    h, sc, sent = run([parcel(A, hidden=99, xy=(0.30, 0.0), half=(0.06, 0.07))], BASE)
    assert "SHOW" in h.entered and "SHOW" not in [s for s, _ in sent] and h.entered.index("SPIN") == h.entered.index("SHOW") + 1  # no scripted SHOW call
    assert h.show is None and h.show_info["why"] == "too_wide" and h.entered.count("SPIN") == BASE["spin"]["tries"]


def test_label_on_a_side_is_found_at_its_view_rotation_stops_early_and_the_delivery_arm_takes_it_from_the_set_down_spot():
    replies = []
    h, sc, sent = run([parcel(B, face="*second", **CUBE)], BASE, replies)
    cs, n = h.show["calls"], len(h.show["calls"])
    second = h.show["faces"][1]
    hold_idx = max(i for i, c in enumerate(cs) if c.get("face") == second and c["tag"] == "hold")
    assert h.show_done == hold_idx + 1 < n  # confirmed on the second image of its view; the remaining faces were never shown
    assert [t for t in h.entered if t in ("SHOW", "SHOWSET", "SLIDE", "PLACE")][:4] == ["SHOW", "SHOWSET", "SLIDE", "PLACE"]
    assert all(i is None for s, i in sent if s in ("SHOW", "SHOWSET"))  # scripted: no WALLX call
    assert [i for s, i in sent if s == "PLACE"][0] == "put the block into the red bin with the right arm"
    assert sc.at[id(sc.log[0])] == "red" and h.sm.results[0]["outcome"] == "ok" and h.sm.results[0]["code"] == B
    assert h.show_info["delivery"] == "right_ee" and h.sset["where"] in ("handoff", "orig")  # (a fitted height a few mm taller can leave no hand-off spot: it is put back where it was)
    assert h.slide_info["why"] == "graspable" and h.slide is None  # already within the delivery arm's reach: no push
    # scripted_reply drives the gripper values and keeps the other arm home
    arm = h.show["arm"]
    other = "right_ee" if arm == "left_ee" else "left_ee"
    shows = [(c, r) for s, c, r in replies if s == "SHOW" and c < n]
    assert shows and all(np.allclose(r["actions"][other]["position"][-1], h.home[other]["position"], atol=1e-6) for _, r in shows)
    close = next(r for c, r in shows if c == 2)
    assert close["actions"][arm]["gripper"][-1] == pytest.approx(SC["grip_close"])
    assert [r for s, c, r in replies if s == "SHOWSET" and c == len(h.sset["calls"]) - 2][0]["actions"][arm]["gripper"][-1] == pytest.approx(SC["grip_open"])  # release


def test_label_never_shown_all_faces_driven_then_reject_when_nothing_else_is_enabled():
    cfg = {**BASE, "spin": {**BASE["spin"], "enabled": False}}
    h, sc, sent = run([parcel(B, face="bottom", **CUBE)], cfg)  # the bottom is unreachable: the label is never shown
    assert h.show_done == len(h.show["calls"]) and "bottom" in h.show["missing"]
    assert h.entered.count("SHOW") == 1 and "SPIN" not in h.entered
    assert h.sm.results[0]["reason"] == "unread" and sc.at[id(sc.log[0])] == "black"  # the bottom was never seen: not "no_label"
    assert [i for s, i in sent if s == "REJECT"][0] == "put the block into the black bin with the right arm"
    assert h.show_info["delivery"] == "right_ee" and h.sset["where"] in ("handoff", "orig")  # (a fitted height a few mm taller can leave no hand-off spot: it is put back where it was)


def test_faces_the_arm_cannot_show_fall_back_to_spin_pushes():
    h, sc, sent = run([parcel(B, face="bottom", **CUBE)], BASE)
    order = [t for t in h.entered if t in ("SHOW", "SHOWSET", "SPIN")]
    assert order[:3] == ["SHOW", "SHOWSET", "SPIN"] and h.entered.count("SPIN") >= 1 and h.sm.results[0]["reason"] == "unread"


@pytest.mark.parametrize("pick", ["left_ee", "right_ee"])
@pytest.mark.parametrize("deliv", ["left_ee", "right_ee"])
def test_set_down_is_inside_the_delivery_arms_grasp_region_and_feasible_through_the_urdf_chain(pick, deliv):
    fit = fit_of((0.30, 0.0), 0, (0.038, 0.038))
    p = look.show_plan(pick, fit, SC)
    n = len(p["calls"])
    for done in (5, n):  # just lifted, after the last view
        s = look.set_down(p, deliv, done, SC)
        assert s["where"] == "handoff" and [c["tag"] for c in s["calls"]][-3:] == ["down", "release", "retreat"]
        spot, f2 = s["spot"], s["fit"]
        assert np.linalg.norm(spot - SC["handoff"][deliv[:-3]]) <= 0.05 + 1e-9
        assert look.footprint_clear(spot, f2, SC) and look.grasp_plan(deliv, f2, SC) is not None  # the delivery arm can grasp it there
        cur = p["calls"][done - 1]
        low = drive(pick, s["calls"], p["hold"], start=(cur["ee"], cur["rpy"]), held_from=0, q=cur["q"])  # (down leg ends at the table: parcel flat)
        assert low >= look.TABLE_Z - 0.004
        grips = [c["grip"] for c in s["calls"]]
        assert grips[-3:] == [SC["grip_close"], SC["grip_open"], SC["grip_open"]]
    assert look.set_down(p, deliv, 3, SC)["where"] == "orig" and look.set_down(p, deliv, 1, SC)["calls"] == []  # closed on it at the table / never gripped
    s4 = look.set_down(p, deliv, 4, SC)  # 2 cm up: back down to the grasp pose, then let go
    assert s4["where"] == "orig" and [c["tag"] for c in s4["calls"]] == ["down", "release", "retreat"] and np.allclose(s4["calls"][1]["ee"], p["calls"][2]["ee"])


def test_held_parcel_is_a_box_hanging_from_the_fingertips():
    hold = (0.076, 0.076, 0.08, np.pi / 2)
    arm, tip = "left_ee", np.array([0.3, 0.0, look.TABLE_Z + 0.08 - SC["grasp_depth"]])  # the grasp point: the parcel lies flat on the table
    rpy = np.array([0.0, np.pi / 2, 0.3])
    ee = look.to_arm(arm, tip, SC) - look.rpy_to_matrix(rpy) @ look.TCP
    assert look.held_low(arm, ee, rpy, hold, SC) == pytest.approx(look.TABLE_Z, abs=1e-9)
    # gripped at pitch 70 the parcel hangs tilted in the hand, but flat in the world at the grasp orientation
    hold70 = (*hold[:3], np.radians(70))
    rpy70 = np.array([0.0, np.radians(70), 0.3])
    ee70 = look.to_arm(arm, tip, SC) - look.rpy_to_matrix(rpy70) @ look.TCP
    assert look.held_low(arm, ee70, rpy70, hold70, SC) == pytest.approx(look.TABLE_Z, abs=1e-9)
    for pitch in (70, 75, 80, 85, 90):  # explicit world-frame corners: flat on the table at every grasp pitch, 8 distinct corners, the tip inside the box
        hp = (0.076, 0.102, 0.05, np.radians(pitch))
        r = np.array([0.0, np.radians(pitch), -0.4])
        e = look.to_arm(arm, tip, SC) - look.rpy_to_matrix(r) @ look.TCP
        cs = look.held_corners(arm, e, r, hp, SC)
        assert len({tuple(np.round(c, 6)) for c in cs}) == 8 and cs[:, 2].min() == pytest.approx(tip[2] - (hp[2] - SC["grasp_depth"]), abs=1e-9)
        assert look.held_low(arm, e, r, hp, SC) == pytest.approx(cs[:, 2].min())
