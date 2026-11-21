from pathlib import Path

from labelloop.brain.state_machine import StateMachine
from labelloop.config import load_yaml

BASE = load_yaml(Path(__file__).parent.parent / "configs" / "task_qr_sort.yaml")
CFG = {**BASE, "look": {"enabled": False}, "spin": {"enabled": False}, "handover": True, "tries": {"PUSH": 0, "INSPECT": 3, "HANDOVER": 3, "PLACE": 3, "REJECT": 3, "LIFT2": 3}, "budget": {"case_s": 1e9, "deliver_reserve_s": 0, "wallx_call_s": 11.0, "scripted_call_s": 2.2}}  # the v5 flow (handover, INSPECT, no budget); SPIN / LOOK / arm roles / budget have their own tests below
A, B, NOZIP, BAD = "420/95123/001", "420/90001/002", "420/10001/003", "LL|ZONE-A|001"
GAIN = 0.05  # > bin_gain


def ev(presence=0.0, kroi=0.0, codes=(), bins=None, **kw):
    return {"presence": presence, "kroi": kroi, "codes": set(codes), "bins": {"blue": 0.0, "red": 0.0, "black": 0.0, **(bins or {})}, **kw}


def parcel(sm, code, kroi=0.08, **kw):
    """Drive IDLE -> SCAN -> routed; returns the instruction after the SCAN call."""
    assert sm.step(ev(0.06, kroi, **kw)) is None and sm.state == "IDLE"
    assert sm.step(ev(0.06, kroi, **kw)) is None and sm.state == "SCAN"
    return sm.step(ev(0.06, kroi, [code] if code else [], **kw))


def handover(sm):
    """HANDOVER runs its fixed budget (no success check), then the place step; returns that instruction."""
    assert sm.state == "HANDOVER"
    for _ in range(CFG["max_calls"]["HANDOVER"] - 1):
        assert sm.step(ev(0.06, 0.08)) == "hand the block from the left arm to the right arm" and sm.state == "HANDOVER"
    return sm.step(ev(0.06, 0.08))


def finish(sm, **landed):
    """Object gone from the zone, the given bins changed: through HOME and VERIFY."""
    for _ in range(2):
        sm.step(ev(0.0, 0.0, bins=landed))
    states = []
    while sm.state in ("HOME", "VERIFY"):
        sm.step(ev(0.0, 0.0, bins=landed))
        states.append(sm.state)
    return states


def to_inspect(sm, **kw):
    parcel(sm, None, **kw)
    for _ in range(CFG["scan_max_calls"] - 2):
        assert sm.step(ev(0.06, 0.08, **kw)) is None
    return sm.step(ev(0.06, 0.08, **kw))


def test_lane_a_handover_then_place_ok_early_end_home_verify_idle_and_counts():
    sm = StateMachine(CFG)
    assert parcel(sm, A) == "hand the block from the left arm to the right arm" and sm.plan == ["HANDOVER", "PLACE"]
    assert handover(sm) == "put the block into the blue bin with the right arm" and sm.state == "PLACE"
    assert sm.step(ev(0.0, bins={"blue": GAIN})) == "put the block into the blue bin with the right arm"  # ok 1 of 2
    assert sm.step(ev(0.0, bins={"blue": GAIN})) == "put both arms back to the home position"
    assert sm.state == "HOME" and sm.calls == 0
    assert finish(sm, blue=GAIN)[-1] == "IDLE"
    assert sm.parcels == 1 and sm.results[0]["outcome"] == "ok" and sm.results[0]["lane"] == "A"
    assert sm.state == "IDLE"  # never DONE with max_parcels null


def test_lane_b_place_fails_twice_retries_from_handover_then_ok_on_third_try():
    sm = StateMachine(CFG)
    parcel(sm, B)
    assert handover(sm) == "put the block into the red bin with the right arm" and sm.state == "PLACE"
    for _ in range(CFG["tries"]["PLACE"] - 1):  # object never leaves: budget runs out, HOME, VERIFY, retry from HANDOVER
        for _ in range(CFG["max_calls"]["PLACE"]):
            sm.step(ev(0.06, 0.08))
        assert sm.state == "HOME"
        while sm.state in ("HOME", "VERIFY"):
            sm.step(ev(0.06, 0.08))
        assert sm.state == "HANDOVER"
        handover(sm)
        assert sm.state == "PLACE"
    for _ in range(2):
        sm.step(ev(0.0, bins={"red": GAIN}))
    assert sm.state == "HOME"
    assert finish(sm, red=GAIN)[-1] == "IDLE" and sm.results[0]["outcome"] == "ok"


def test_unknown_zip_and_bad_code_reject_via_handover():
    for code, reason in ((NOZIP, "unknown_zip"), (BAD, "bad_code")):
        sm = StateMachine(CFG)
        assert parcel(sm, code) == "hand the block from the left arm to the right arm" and sm.plan == ["HANDOVER", "REJECT"]
        assert handover(sm) == "put the block into the black bin with the right arm"
        assert sm.reason == reason and sm.lane is None


def test_no_code_goes_to_inspect_variants_per_try_then_no_label_reject():
    sm = StateMachine(CFG)
    assert to_inspect(sm) == CFG["instructions"]["INSPECT"][0].format(obj="block") and sm.state == "INSPECT"
    for t in range(1, CFG["tries"]["INSPECT"]):  # budget spent -> next try, next wording
        for _ in range(CFG["max_calls"]["INSPECT"] - 1):
            assert sm.step(ev(0.06, 0.08)) == CFG["instructions"]["INSPECT"][t - 1].format(obj="block")
        assert sm.step(ev(0.06, 0.08)) == CFG["instructions"]["INSPECT"][t].format(obj="block") and sm.calls == 0
    for _ in range(CFG["max_calls"]["INSPECT"] - 1):
        sm.step(ev(0.06, 0.08))
    assert sm.step(ev(0.06, 0.08)) == "hand the block from the left arm to the right arm"
    assert sm.reason == "no_label" and sm.plan == ["HANDOVER", "REJECT"] and sm.tries["INSPECT"] == CFG["tries"]["INSPECT"]


def test_code_found_mid_inspect_routes_immediately_without_home():
    sm = StateMachine(CFG)
    to_inspect(sm)
    sm.step(ev(0.06, 0.08))
    assert sm.step(ev(0.06, 0.08, [A])) == "hand the block from the left arm to the right arm"
    assert sm.lane == "A" and sm.transitions[-1][:2] == ("INSPECT", "HANDOVER")


def test_big_parcel_both_arms_lane_and_reject_no_handover():
    sm = StateMachine(CFG)
    sm.step(ev(0.4, 0.08)), sm.step(ev(0.4, 0.08))
    assert sm.step(ev(0.4, 0.08, [A])) == "pick up the block with both arms and put it into the blue bin" and sm.state == "LIFT2"
    assert sm.step(ev(0.4, 0.08)) and finish(sm, blue=GAIN)[-1] == "IDLE" and sm.results[0]["outcome"] == "ok"
    sm = StateMachine(CFG)  # reject: mat = black; the "big" override works too
    assert parcel(sm, NOZIP, big=True) == "pick up the block with both arms and put it into the black bin" and sm.plan == ["LIFT2"]
    sm = StateMachine(CFG)
    assert to_inspect(sm, big=True) == "lift the block with both arms and turn it over"


def test_garbage_rejects_via_handover():
    sm = StateMachine(CFG)
    assert parcel(sm, None, kroi=0.0) is None and sm.state == "SCAN"  # garbage waits out the QR confirmation window (2 SCAN calls)
    assert sm.step(ev(0.06, 0.0)) == "hand the object from the left arm to the right arm" and sm.reason == "garbage"
    assert sm.plan == ["HANDOVER", "REJECT"]


def test_already_sorted_codes_are_ignored():
    sm = StateMachine(CFG)
    parcel(sm, A)
    handover(sm)
    assert finish(sm, blue=GAIN)[-1] == "IDLE" and A in sm.sorted
    assert parcel(sm, A) is None and sm.state == "SCAN"  # a labelled parcel lying in the bin can never be re-read
    assert sm.step(ev(0.06, 0.08, [A, B])) == "hand the block from the left arm to the right arm" and sm.code == B


def test_reject_success_then_failure_after_last_try_goes_idle_and_rebases():
    sm = StateMachine(CFG)
    parcel(sm, NOZIP)
    handover(sm)
    assert finish(sm, black=GAIN)[-1] == "IDLE" and sm.results[0]["outcome"] == "ok"
    sm = StateMachine(CFG)
    parcel(sm, NOZIP)
    entered = 0
    for _ in range(300):  # object never leaves the zone
        if sm.state == "IDLE":
            break
        sm.step(ev(0.06, 0.08))
        entered += any(t[1] == "REJECT" for t in sm.transitions)
    assert sm.state == "IDLE" and sm.results[0]["outcome"] == "failed" and sm.rebase and entered == CFG["tries"]["REJECT"]
    assert sm.step(ev()) is None and sm.parcels == 1


def test_hand_in_zone_not_stable_no_action_and_one_frame_occlusion_harmless():
    sm = StateMachine(CFG)
    for f in (0.2, 0.35, 0.1, 0.2, 0.0):  # hand moving: presence keeps changing, then leaves
        assert sm.step(ev(f, 0.08)) is None and sm.state == "IDLE"
    parcel(sm, A)
    assert sm.state == "HANDOVER"
    sm = StateMachine(CFG)
    sm.step(ev(0.06, 0.08)), sm.step(ev(0.06, 0.08))
    assert sm.step(ev(0.06, 0.08, [])) is None and sm.state == "SCAN"  # code occluded: just wait
    assert sm.step(ev(0.06, 0.08, [A])) and sm.state == "HANDOVER"


def test_max_parcels_makes_done_and_holds():
    sm = StateMachine({**CFG, "max_parcels": 1})
    parcel(sm, NOZIP)
    handover(sm)
    assert finish(sm, black=GAIN)[-1] == "DONE" and sm.step(ev()) is None
