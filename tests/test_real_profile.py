import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from labelloop import look
from labelloop.brain.state_machine import StateMachine
from labelloop.config import load_yaml
from labelloop.harness import Harness, apply_profile
from labelloop.perception import scene
from test_event_server import FakeHarness, ask, env, req  # noqa: F401  (env fixture)

CFG = load_yaml(Path(__file__).parent.parent / "configs/task_qr_sort.yaml")
SAFE = {**CFG, "profile": "real", "real_safe": True}
STATE = {"left_ee": {"position": [.1, .2, .3], "rotation_rpy": [0, 0, 0], "gripper": .1},
         "right_ee": {"position": [.1, -.2, .3], "rotation_rpy": [0, 0, 0], "gripper": .2}}


@pytest.fixture(autouse=True)
def sim_after():
    yield
    look.set_profile(CFG["profiles"]["sim"])


def test_profile_round_trip_sim_values_unchanged():
    sim = lambda: (look.TABLE_Z, look.HEAD_POS.copy(), look.HEAD_R.copy(), look.FX, look.FY, look.CX, look.CY, look.CAM_OFF.copy(), look.CAM_TILT)  # noqa: E731
    a = sim()
    assert a[0] == 0.015 and a[3] == 2.97 / 5.54 * 1280 and a[4] == 2.97 / 3.09 * 720 and np.allclose(a[1], [0.084, 0, 0.517])
    assert np.allclose(a[2][:, 2] * -1, [0.574, 0, -0.819], atol=1e-3)  # sim head looks 55 deg down
    look.set_profile(CFG["profiles"]["real"])
    assert look.TABLE_Z != 0.015
    look.set_profile(CFG["profiles"]["sim"])
    b = sim()
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert apply_profile(CFG)["look"]["arm_y"] == 0.29 and apply_profile(CFG)["look"]["arm_z"] == 0.03


def test_real_head_projection_sanity():
    ab = look.set_profile(CFG["profiles"]["real"])
    assert ab["arm_y"] == 0.263 and look.FX == pytest.approx(717.9853)
    p = look.pixel_to_table(look.CX, look.CY)
    assert p[2] == pytest.approx(look.TABLE_Z) and 0.1 < p[0] < 0.5 and abs(p[1]) < 0.1  # centre ray: in front of the robot, near the midline
    assert np.allclose(look.table_to_pixel(p), [look.CX, look.CY], atol=1e-6)
    d = -look.HEAD_R[:, 2]
    assert np.degrees(np.arcsin(-d[2])) == pytest.approx(25.2, abs=0.3)  # looks ~25 deg down
    for arm in ("left_ee", "right_ee"):  # arm bases (root) land left / right in the image, below the camera's mid-height
        u, _ = look.table_to_pixel(look.frame(arm, ab) + [0.15, 0, look.TABLE_Z - ab["arm_z"] * 0])
        assert (u < 640) == (arm == "left_ee")


def test_real_safe_interlock_latches_hold():
    cfg = apply_profile(SAFE)
    assert cfg["max_parcels"] == 1 and not any(cfg[k]["enabled"] for k in ("show", "spin", "slide", "look"))
    for f in (lambda m: m._route(None, "unread"), lambda m: m._route("A", "routed", slide=True) or m._go("SWIPE", "x"),
              lambda m: m.reroute_wide(), lambda m: m._go("SPIN", "x"), lambda m: m._go("SHOW", "x"), lambda m: m._go("LOOK", "x"),
              lambda m: m._go("LIFT2", "x"), lambda m: m._go("SLIDE", "x")):
        m = StateMachine(cfg)
        f(m)
        if m.state != "HOLD":
            continue
        assert "operator_hold" in m.transitions[-1][2]
        for _ in range(5):
            assert m.step({"presence": 0.0, "kroi": 0, "codes": set(), "bins": {}, "raw": set()}) is None and m.state == "HOLD"
    for why, want in (("unknown_zip", "REJECT"), ("bad_code", "REJECT"), ("unread", "HOLD"), ("no_label", "HOLD"), ("too_wide", "HOLD")):
        m = StateMachine(cfg)
        m._route(None, why)
        assert m.state == want, why
    m = StateMachine(cfg)
    m._route("A", "routed")
    assert m.state == "PLACE" and m.instruction()  # a lane parcel goes to WALLX
    m._done("ok")
    assert m.state == "DONE"  # max_parcels 1


def test_real_safe_replies_are_hold_or_home_only():
    h = Harness(SAFE)
    img = {k: np.full((72, 128, 3), 128, np.uint8) for k in ("front", "left_wrist", "right_wrist")}
    h.step(img, STATE), h.step(img, STATE)
    assert h.home is not None
    for st in ("IDLE", "SCAN", "HOLD", "PLACE", "HOME", "VERIFY", "SHOW", "SPIN", "SWIPE", "SLIDE", "LOOK"):
        h.sm.state = st
        r = h.scripted_reply("teletwin_desktop/gripper-g", STATE)
        if st in ("SHOW", "SPIN", "SWIPE", "SLIDE", "LOOK"):  # unreachable under the interlock; documents that the replies stay inert
            continue
        assert r is None or st == "HOME"
    h.sm.state = "HOLD"
    assert h.step(img, STATE) is None and h.sm.state == "HOLD"
    assert h.wrist_cams("INSPECT", 0) == set()


def test_home_needs_two_still_observations():
    h = Harness(SAFE)
    img = {k: np.full((72, 128, 3), 128, np.uint8) for k in ("front", "left_wrist", "right_wrist")}
    moving = copy.deepcopy(STATE)
    assert h.step(img, STATE) is None and h.home is None  # first observation: never the home
    moving["left_ee"]["position"][0] += 0.02
    assert h.step(img, moving) is None and h.home is None  # moved 2 cm
    h.step(img, moving)
    assert h.home == moving  # steady twice


def test_bins_blanked_before_decode():
    img = np.full((100, 200, 3), 255, np.uint8)
    out = scene.blank_bins(img, CFG["scene"])
    x0, y0, x1, y1 = CFG["scene"]["bin_rois"]["blue"]
    assert out[int(50 * (y0 + y1)), int(200 * (x0 + x1) / 2)].sum() == 0 and out[99, 100].sum() == 765 * 1 and img.min() == 255


def test_save_images(env, tmp_path):  # noqa: F811
    stub, make = env
    from labelloop.robots import event_server
    import test_event_server as t
    ws = make(None)
    ws.recv(timeout=10)
    box = t.start_relay(event_server.Relay(stub.url, {}, None, None, str(tmp_path)))
    from websockets.sync.client import connect
    ws = connect("ws://127.0.0.1:%d" % box["port"], compression=None, max_size=None)
    ws.recv(timeout=10)
    for _ in range(12):
        ask(ws, req())
    eps = list(tmp_path.iterdir())
    assert len(eps) == 1
    names = sorted(p.name for p in eps[0].iterdir())
    assert names == sorted([f"{n}_{c}.jpg" for n in (0, 10) for c in ("front", "left_wrist", "right_wrist")] + ["0.json", "10.json"])
    j = json.loads((eps[0] / "10.json").read_text())
    assert j["state"]["left_ee"]["gripper"] == pytest.approx(.1) and j["instruction"] == "sort the boxes" and set(j) == {"instruction", "state"}
    box["loop"].call_soon_threadsafe(box["stop"].set)


# ---- event_fast: WALLX-only ----
FAST = {**CFG, "profile": "real", "real_safe": True, "event_fast": True}
EMPTY = {"presence": 0.0, "kroi": 0, "codes": set(), "bins": {"blue": 0, "red": 0, "black": 0}, "raw": set()}
ON = {**EMPTY, "presence": 0.3}
LANDED = {**EMPTY, "bins": {"blue": 0.1, "red": 0, "black": 0}}
CODE = "420/95123/001"  # ZIP3 951 -> lane A (blue)


def fast_sm():
    t = [0.0]
    m = StateMachine(apply_profile(FAST), lambda: t[0])
    return m, t


def drive(m, evs):
    """Steps m through evs; returns the (state, instruction) of every call that sent something non-None or changed state."""
    out = []
    for ev in evs:
        out.append((m.state, m.step(ev)))
    return out


def test_fast_top_label_call_sequence_and_no_scripted_reply():
    m, _ = fast_sm()
    assert not m.cfg["scripted_home"] and not any(m.cfg[k]["enabled"] for k in ("show", "spin", "slide", "look")) and m.cfg["reject_mode"] == "tray"
    out = drive(m, [ON, ON, {**ON, "codes": {CODE}}, ON, LANDED, EMPTY, EMPTY])
    # IDLE (stable 1) -> SCAN (2 calls) -> PLACE until the bin changed -> HOME (1) -> DONE
    assert [i for _, i in out if i] == ["put the box into the blue bin"] * 2 + ["put both arms back to the home position"]
    assert m.results == [{"code": CODE, "lane": "A", "reason": "routed", "outcome": "ok"}] and m.state == "DONE"
    assert all(i is None or "home position" in i or "bin" in i for _, i in out)


def test_fast_retry_skips_home():
    m, _ = fast_sm()
    drive(m, [ON, ON, {**ON, "codes": {CODE}}])
    assert m.state == "PLACE"
    for _ in range(m.cfg["max_calls"]["PLACE"]):
        ins = m.step(ON)
    assert m.state == "PLACE" and m.tries["PLACE"] == 2 and "home" not in ins and [t[2] for t in m.transitions] == ["retry"]
    m.step(LANDED)
    assert m.state == "HOME"


def test_fast_flip_read_route_and_unread_rejects_via_wallx():
    m, _ = fast_sm()
    drive(m, [ON] * 4)
    assert m.state == "SCAN"
    ins = m.step(ON)  # scan_max_calls 4
    assert m.state == "INSPECT" and ins == "flip the box over"
    assert m.step({**ON, "codes": {CODE}}) and m.state == "PLACE"  # flip -> read -> route
    m, _ = fast_sm()
    drive(m, [ON] * 5)
    assert m.state == "INSPECT"
    for _ in range(2 * m.cfg["max_calls"]["INSPECT"]):
        ins = m.step(ON)
    assert m.state == "REJECT" and ins == "put the box into the black bin" and m.tries["INSPECT"] == 2


def test_fast_budget_is_real_elapsed_time():
    m, t = fast_sm()
    drive(m, [ON, ON, {**ON, "codes": {CODE}}])
    t[0] = 470.0  # 10 s left < 20 s reserve: no retry, home
    for _ in range(m.cfg["max_calls"]["PLACE"]):
        m.step(ON)
    assert m.state == "HOME"


def test_fast_grasp_abort_retries_once():
    for seq, flag in (([1.5, 0.1], "grasp:missed"), ([1.5, 1.0, 1.5], "grasp:dropped")):  # closed on nothing / parcel never left the spot after reopening
        m, _ = fast_sm()
        m.cfg.setdefault("grasp", {})["check"] = True  # off by default on the real robot until gripper units are calibrated
        drive(m, [ON, ON, {**ON, "codes": {CODE}}])
        flags = []
        for g in seq:
            m.step({**ON, "grip": {"left": g, "right": g}})
            flags += m.flags
        assert flag in flags and m.state == "PLACE" and m.tries["PLACE"] == 2 and m.calls == 0  # retry at once, no HOME
        for g in seq:  # the second attempt may fail too: no third try
            m.step({**ON, "grip": {"left": g, "right": g}})
        assert m.state == "HOME"
    m, _ = fast_sm()  # a good grasp (closed on the parcel, then the bin changed) is not aborted
    drive(m, [ON, ON, {**ON, "codes": {CODE}}])
    for g in (1.5, 1.0, 1.0):
        m.step({**ON, "grip": {"left": g, "right": g}})
    assert m.state == "PLACE" and m.tries["PLACE"] == 1
    m.step({**LANDED, "grip": {"left": 1.5, "right": 1.5}})
    assert m.state == "HOME"


def test_hold_mode_saves_30_no_wallx(env, tmp_path):  # noqa: F811
    stub, _ = env
    from labelloop.robots import event_server
    import test_event_server as t
    from websockets.sync.client import connect
    box = t.start_relay(event_server.Relay(None, None, None, None, str(tmp_path), hold=True))
    ws = connect("ws://127.0.0.1:%d" % box["port"], compression=None, max_size=None)
    ws.recv(timeout=10)
    for _ in range(32):
        r = ask(ws, req())
        assert r["actions"]["left_ee"]["position"][31] == pytest.approx(t.STATE["left_ee"]["position"])
    assert not stub.requests and len(list(next(tmp_path.iterdir()).glob("*.json"))) == 30
    box["loop"].call_soon_threadsafe(box["stop"].set)


# ---- trial 1: the parcel lay on the table when the episode started (part 1-3) and an unreadable label (part 4) ----
MAT, LBL = (90, 140, 100), (200, 150, 110)  # RGB green work mat / kraft-ish parcel (not the mat colour)
GRIPPERS = ((0.0, 0.80, 0.18, 1.0), (0.72, 0.80, 1.0, 1.0))  # must match scene.gripper_rois


def head_frame(parcel=True, w=200, h=100, box=(0.30, 0.55, 0.55, 0.90)):
    """Synthetic real-profile head frame: the green mat, a parcel (image fractions) and the parked grippers in the bottom corners."""
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :] = MAT
    for x0, y0, x1, y1 in GRIPPERS:
        img[int(y0 * h):, int(x0 * w):int(x1 * w)] = (40, 40, 45)
    if parcel:
        x0, y0, x1, y1 = box
        img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = LBL
    return img


def frames(img):
    return {"front": img, "left_wrist": img, "right_wrist": img}


def fast_harness(empty_ref=None):
    prof = {**CFG["profiles"]["real"], "empty_ref": empty_ref}
    return Harness({**FAST, "profiles": {**CFG["profiles"], "real": prof}})


@pytest.fixture
def empty_jpg(tmp_path):
    p = tmp_path / "empty.jpg"
    assert cv2.imwrite(str(p), cv2.cvtColor(head_frame(parcel=False), cv2.COLOR_RGB2BGR))
    return str(p)


def test_idle_routes_on_a_valid_code_with_zero_presence():
    for m in (StateMachine(CFG), fast_sm()[0]):  # sim config and event_fast: the label alone gets the search going
        assert m.step({**EMPTY, "codes": {CODE}}) is None
        assert m.state == "SCAN" and m.transitions[-1][2] == "code visible"
        m.step({**EMPTY, "codes": {CODE}})  # SCAN re-reads it: route, no presence change needed
        assert m.lane == "A" and m.code == CODE and m.state in ("SLIDE", "PLACE")
    m = StateMachine(CFG)  # a text that is not a parcel label does not leave IDLE
    m.step({**EMPTY, "codes": {"not-a-label"}})
    assert m.state == "IDLE"


def test_empty_ref_is_the_initial_reference(empty_jpg):
    img = head_frame()
    h = fast_harness(empty_jpg)
    assert h.step(frames(img), STATE) is None and h.ref is None  # home not steady yet
    h.step(frames(img), STATE)
    assert h.ref_cfg and h.sm.state == "SCAN"  # not empty vs the configured image -> a parcel is already there
    assert scene.presence(img, h.ref, h.cfg["scene"]) > 0.1
    cfg = apply_profile({**FAST, "profiles": {**CFG["profiles"], "real": {**CFG["profiles"]["real"], "empty_ref": "runs/event_images/x/0_front.jpg"}}})
    assert cfg["empty_ref"].endswith("runs/event_images/x/0_front.jpg") and Path(cfg["empty_ref"]).is_absolute()  # relative to the repo root
    assert apply_profile(FAST)["empty_ref"] is None


def test_first_frame_parcel_is_present_without_a_change():
    h = fast_harness()  # no empty_ref: the parcel is part of the reference, as in trial 1
    h.step(frames(head_frame()), STATE)
    assert h.step(frames(head_frame()), STATE) is None and h.home is not None
    assert h.sm.state == "SCAN" and h.obj_box is not None and h.near == "left"
    h2 = fast_harness()
    h2.step(frames(head_frame(parcel=False)), STATE), h2.step(frames(head_frame(parcel=False)), STATE)
    assert h2.sm.state == "IDLE" and h2.obj_box is None  # an empty bench stays in IDLE


def test_object_box_ignores_the_mat_bins_and_parked_grippers():
    sc = apply_profile(FAST)["scene"]
    assert scene.object_box(head_frame(parcel=False), None, sc) is None  # mat + parked grippers only
    b = scene.object_box(head_frame(), None, sc)
    assert b is not None and 55 <= b[0] <= 65 and 105 <= b[2] <= 115 and 50 <= b[1] <= 60 and 85 <= b[3] <= 92  # the parcel at 0.30-0.55 x 0.55-0.90


def test_wrist_code_counts_for_the_near_arm(monkeypatch, empty_jpg):
    monkeypatch.setattr("labelloop.harness.locate", lambda im: [])
    monkeypatch.setattr("labelloop.harness.decode", lambda imgs: [(CODE, v) for v in imgs])
    h = fast_harness(empty_jpg)
    for _ in range(4):
        h.step(frames(head_frame()), STATE)
    assert h.sm.state == "PLACE" and h.sm.code == CODE and h.near == "left" and h.wrist_cams("SCAN", 0) == {"left_wrist"}
    h = fast_harness(empty_jpg)
    for _ in range(4):
        h.step(frames(head_frame(box=(0.50, 0.55, 0.70, 0.90))), STATE)  # image-right half -> the right wrist
    assert h.sm.state == "PLACE" and h.near == "right"


def test_wrist_code_is_ignored_when_the_head_sees_it_in_a_bin(monkeypatch, empty_jpg):
    monkeypatch.setattr("labelloop.harness.locate", lambda im: [(CODE, (78.0, 26.0))])  # centre of the real blue bin ROI
    monkeypatch.setattr("labelloop.harness.decode", lambda imgs: [(CODE, v) for v in imgs])
    h = fast_harness(empty_jpg)
    ins = [h.step(frames(head_frame()), STATE) for _ in range(6)]
    assert h.sm.lane is None and not any(i and "blue" in i for i in ins)  # an already sorted parcel in a bin blinds the wrist text


def test_hidden_label_centers_the_box_then_routes(monkeypatch, empty_jpg):
    seen = []
    monkeypatch.setattr("labelloop.harness.locate", lambda im: list(seen))
    monkeypatch.setattr("labelloop.harness.decode", lambda imgs: [])
    h = fast_harness(empty_jpg)
    ins = [h.step(frames(head_frame()), STATE) for _ in range(6)]
    assert h.sm.state == "CENTER" and ins[-1] == "slide the box to the middle of the table with the left arm"
    assert h.sm.tries["CENTER"] == 1 and h.sm.transitions[-1][2] == "object, no code"
    seen.append((CODE, (80.0, 70.0)))  # WALLX slid the box out: the label shows now
    ins = [h.step(frames(head_frame()), STATE) for _ in range(2)]
    assert h.sm.state == "PLACE" and h.sm.code == CODE and ins[-1] == "put the box into the blue bin"


def test_real_head_label_reads():  # trial 2: a ~60 px taped label on the real head camera only reads after the x3 + unsharp retry
    import cv2
    from pathlib import Path
    from labelloop.perception.qr import locate
    img = cv2.cvtColor(cv2.imread(str(Path(__file__).parent / "data" / "real_head_qr.jpg")), cv2.COLOR_BGR2RGB)
    assert [t for t, _ in locate(img)] == ["420/95126/001"]


def test_wording_override_and_where_words():  # trial agent: templates come from --wording; {where} = the parcel's place on the free table
    cfg = apply_profile({**CFG, "profile": "real", "event_fast": True, "wording": {"PLACE": "put the small box at the {where} of the table into the {mat} bin"}})
    assert cfg["instructions"]["PLACE"] == ["put the small box at the {where} of the table into the {mat} bin"]
    h = Harness({**CFG, "profile": "real", "event_fast": True})
    img = cv2.cvtColor(cv2.imread(str(Path(__file__).parent / "data" / "real_head_qr.jpg")), cv2.COLOR_BGR2RGB)
    from labelloop.perception.qr import locate
    assert h.where_words(img, h.cfg["scene"], locate(img)) == "left"
    sm = StateMachine(cfg)
    sm.lane, sm.where, sm.state = "A", "front left", "PLACE"
    assert sm.instruction() == "put the small box at the front left of the table into the %s bin" % cfg["lanes"]["A"]


@pytest.mark.skipif(not __import__("shutil").which("tesseract"), reason="tesseract not installed")
def test_real_head_zip_ocr_fallback():  # real Miro label: QR undecodable from the head camera, printed ZIP read by OCR
    from labelloop.perception import label_ocr
    from labelloop.perception.qr import locate
    img = cv2.cvtColor(cv2.imread(str(Path(__file__).parent / "data" / "real_head_miro.jpg")), cv2.COLOR_BGR2RGB)
    assert locate(img) == []
    assert "90012" in [z for z, _ in label_ocr.zips(img, apply_profile({**CFG, "profile": "real"})["scene"]["induction_roi"])]


@pytest.mark.parametrize("name,ocr", [("real_head_qr.jpg", False), ("real_head_miro.jpg", True)])
def test_parcel_present_at_start_routes_in_scan(name, ocr):  # the parcel is in the start reference: SCAN must still accept its label
    if ocr and not __import__("shutil").which("tesseract"):
        pytest.skip("tesseract not installed")
    img = cv2.cvtColor(cv2.imread(str(Path(__file__).parent / "data" / name)), cv2.COLOR_BGR2RGB)
    z = {"position": [0, 0, 0], "rotation_rpy": [0, 0, 0], "gripper": 0.0}
    h = Harness({**CFG, "profile": "real", "event_fast": True, "real_safe": True})
    states = [(h.step({"front": img, "left_wrist": img, "right_wrist": img}, {"left_ee": dict(z), "right_ee": dict(z)}), h.sm.state)[1] for _ in range(5)]
    assert "PLACE" in states and "CENTER" not in states


def test_bin_moved_latches_hold():  # agent trial 2: WALLX lifted the black bin into the red one -> stop, no retry, no HOME
    cfg = apply_profile({**CFG, "profile": "real", "event_fast": True})
    sm = StateMachine(cfg)
    sm.lane, sm.state, sm.plan = [k for k, v in cfg["lanes"].items() if v == "red"][0], "PLACE", ["PLACE"]
    ev = {"presence": 0.4, "kroi": 0, "codes": set(), "raw": set(), "bins": {"blue": 0.05, "red": 0.66, "black": 0.5}}
    for _ in range(5):  # an arm over the black bin (0.4-0.6) is not a moved bin
        sm.step(ev)
    assert sm.state == "PLACE"
    for _ in range(3):
        out = sm.step({**ev, "bins": {"blue": 0.05, "red": 0.7, "black": 0.95}})
    assert sm.state == "HOLD" and out is None and "bin_moved" in sm.flags
