from pathlib import Path

from labelloop.brain.state_machine import StateMachine
from labelloop.config import load_yaml

CFG = load_yaml(Path(__file__).parent.parent / "configs" / "task_qr_sort.yaml")
A, B, NOZIP, BAD = "420/95123/001", "420/90001/002", "420/10001/003", "LL|ZONE-A|001"
GAIN = 0.01  # > mat_gain


def ev(presence=0.0, kroi=0.0, codes=(), kmats=None):
    return {"presence": presence, "kroi": kroi, "codes": set(codes), "kmats": kmats or {"blue": 0.0, "orange": 0.0}}


def parcel(sm, code, kroi=0.08):
    """Drive IDLE -> SCAN -> routed; returns the instruction after the SCAN call."""
    assert sm.step(ev(0.06, kroi)) is None and sm.state == "IDLE"
    assert sm.step(ev(0.06, kroi)) is None and sm.state == "SCAN"
    return sm.step(ev(0.06, kroi, [code] if code else []))


def lane_a(sm):
    parcel(sm, A)
    assert sm.state == "PUSH"


def finish(sm, **landed):
    """Object gone from the zone, kraft gained on the given mats: through HOME and VERIFY."""
    for _ in range(2):
        sm.step(ev(0.0, 0.0, kmats={"blue": 0.0, "orange": 0.0, **landed}))
    states = []
    while sm.state in ("HOME", "VERIFY"):
        sm.step(ev(0.0, 0.0, kmats={"blue": 0.0, "orange": 0.0, **landed}))
        states.append(sm.state)
    return states


def test_lane_a_push_ok_early_end_home_verify_idle_and_counts():
    sm = StateMachine(CFG)
    assert parcel(sm, A) == "push the block onto the blue mat"
    assert sm.step(ev(0.06, 0.08)) == "push the block onto the blue mat" and sm.state == "PUSH"
    assert sm.step(ev(0.0, kmats={"blue": GAIN, "orange": 0.0})) == "push the block onto the blue mat"  # ok 1 of 2
    assert sm.step(ev(0.0, kmats={"blue": GAIN, "orange": 0.0})) == "put both arms back to the home position"
    assert sm.state == "HOME" and sm.calls == 0
    assert finish(sm, blue=GAIN)[-1] == "IDLE"
    assert sm.parcels == 1 and sm.results[0]["outcome"] == "ok" and sm.results[0]["lane"] == "A"
    assert sm.state == "IDLE"  # never DONE with max_parcels null


def test_lane_b_push_fails_twice_then_place_fallback_then_ok():
    sm = StateMachine(CFG)
    parcel(sm, B)
    assert sm.state == "PUSH"
    for attempt in (1, 2):
        for _ in range(CFG["max_calls"]["PUSH"]):
            assert sm.step(ev(0.06, 0.08)) == "push the block onto the orange mat" or sm.state == "HOME"
        assert sm.state == "HOME"
        while sm.state != "PUSH" and sm.state != "PLACE":
            sm.step(ev(0.06, 0.08))
        assert sm.state == ("PUSH" if attempt == 1 else "PLACE")
    assert sm.step(ev(0.06, 0.08)) == "put the block on the orange mat"
    for _ in range(2):
        sm.step(ev(0.0, kmats={"orange": GAIN, "blue": 0.0}))
    assert sm.state == "HOME"
    assert finish(sm, orange=GAIN)[-1] == "IDLE" and sm.results[0]["outcome"] == "ok"


def test_unknown_zip_and_bad_code_reject_to_basket():
    for code, reason in ((NOZIP, "unknown_zip"), (BAD, "bad_code")):
        sm = StateMachine(CFG)
        assert parcel(sm, code) == "put the block into the basket on the right"
        assert sm.reason == reason and sm.lane is None


def test_no_label_after_scan_budget_and_garbage_immediately():
    sm = StateMachine(CFG)
    parcel(sm, None)
    assert sm.state == "SCAN"
    for _ in range(CFG["scan_max_calls"] - 2):
        assert sm.step(ev(0.06, 0.08)) is None
    assert sm.step(ev(0.06, 0.08)) == "put the block into the basket on the right" and sm.reason == "no_label"
    sm = StateMachine(CFG)
    assert parcel(sm, None, kroi=0.0) == "put the object into the basket on the right" and sm.reason == "garbage"


def test_reject_success_then_failure_after_last_try_goes_idle_and_rebases():
    sm = StateMachine(CFG)
    parcel(sm, NOZIP)
    assert finish(sm)[-1] == "IDLE" and sm.results[0]["outcome"] == "ok"
    sm = StateMachine(CFG)
    parcel(sm, NOZIP)
    for _ in range(CFG["tries"]["REJECT"]):  # object never leaves the zone
        while sm.state != "IDLE" and (sm.state != "REJECT" or sm.calls == 0):
            sm.step(ev(0.06, 0.08))
        for _ in range(60):
            if sm.state == "IDLE":
                break
            sm.step(ev(0.06, 0.08))
    assert sm.state == "IDLE" and sm.results[0]["outcome"] == "failed" and sm.rebase
    assert sm.step(ev()) is None and sm.parcels == 1


def test_hand_in_zone_not_stable_no_action_and_one_frame_occlusion_harmless():
    sm = StateMachine(CFG)
    for f in (0.2, 0.35, 0.1, 0.2, 0.0):  # hand moving: presence keeps changing, then leaves
        assert sm.step(ev(f, 0.08)) is None and sm.state == "IDLE"
    parcel(sm, A)
    assert sm.state == "PUSH"
    sm = StateMachine(CFG)
    sm.step(ev(0.06, 0.08)), sm.step(ev(0.06, 0.08))
    assert sm.step(ev(0.06, 0.08, [])) is None and sm.state == "SCAN"  # code occluded: just wait
    assert sm.step(ev(0.06, 0.08, [A])) and sm.state == "PUSH"


def test_max_parcels_makes_done_and_holds():
    sm = StateMachine({**CFG, "max_parcels": 1})
    parcel(sm, NOZIP)
    assert finish(sm)[-1] == "DONE" and sm.step(ev()) is None
