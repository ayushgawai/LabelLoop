from pathlib import Path

from labelloop.brain.state_machine import StateMachine
from labelloop.config import load_yaml

CFG = load_yaml(Path(__file__).parent.parent / "configs" / "task_qr_sort.yaml")
A, B = "LL|ZONE-A|001", "LL|ZONE-B|002"
W, H = 1280, 720
BINS = {"blue": (700, 100, 900, 280), "orange": (380, 100, 580, 280)}
KEEP = object()


def ev(front=(), wrist=(), blocks=(), kraft=None, grips=(0.2, 0.2), verdict=None, confirmed=KEEP):
    front = list(front)
    return {"w": W, "h": H, "front": front, "wrist": set(wrist), "bins": BINS, "blocks": list(blocks),
            "kraft": kraft or {"blue": 0.0, "orange": 0.0}, "grips": grips, "verdict": verdict,
            "confirmed": {t for t, _ in front} | set(wrist) if confirmed is KEEP else confirmed}


def sm_in(state, code=A):
    sm = StateMachine(CFG)
    sm.step(ev())
    sm.step(ev())  # leave SURVEY: no evidence -> DONE; force state instead
    sm.state, sm.code, sm.calls, sm.transitions = state, code, 0, []
    return sm


def survey():
    sm = StateMachine(CFG)
    assert sm.step(ev()) is None  # first frame: wait for the QR confirmation window
    return sm


def test_survey_place_wording_and_where_each_third():
    for x, word in ((100, "on the left"), (640, "in the middle"), (1200, "on the right")):
        sm = survey()
        assert sm.step(ev([(A, (x, 450))])) == "put the block %s into the blue bin" % word
    sm = survey()
    assert sm.step(ev([(B, (640, 450))])) == "put the block in the middle into the orange bin"


def test_survey_prefers_target_nearest_centre_and_skips_unconfirmed_and_in_bin():
    sm = survey()
    sm.step(ev([(A, (100, 450)), (B, (500, 400))]))
    assert sm.code == B
    sm = survey()
    sm.step(ev([(A, (640, 450))], blocks=[(640, 450)], confirmed=set()))
    assert sm.state == "FETCH"  # unconfirmed code is not a target yet
    sm = survey()
    sm.step(ev([(A, (800, 200))]))  # code seen inside the blue bin: not a target
    assert sm.state == "DONE"


def test_survey_fetch_for_block_without_code_done_when_nothing():
    sm = survey()
    assert sm.step(ev(blocks=[(250, 450)])) == "pick up the block on the left" and sm.state == "FETCH"
    sm = survey()
    assert sm.step(ev()) == "put both arms back to the home position" and sm.state == "DONE"
    assert sm.step(ev([(A, (640, 450))], blocks=[(1, 1)])) == "put both arms back to the home position"


def test_agent_zero_means_done():
    sm = survey()
    sm.step(ev(blocks=[(250, 450)], verdict={"unsorted_boxes": 0}))
    assert sm.state == "DONE"


def test_place_success_needs_code_gone_two_calls_and_kraft_in_right_bin():
    sm = sm_in("PLACE")
    sm.base = {"blue": 0.001, "orange": 0.0}
    gain = {"blue": 0.005, "orange": 0.0}
    sm.step(ev(kraft=gain))
    assert sm.state == "PLACE"  # gone only once
    sm.step(ev(kraft=gain))
    assert sm.state == "SURVEY" and sm.sorted == {A} and not sm.wrong


def test_place_code_still_on_table_keeps_going_and_updates_where():
    sm = sm_in("PLACE")
    sm.base = {"blue": 0.0, "orange": 0.0}
    for _ in range(3):
        assert sm.step(ev([(A, (1100, 450))], kraft={"blue": 0.005, "orange": 0.0})) == \
            "put the block on the right into the blue bin"
    assert sm.state == "PLACE"


def test_place_wrong_bin_logged():
    sm = sm_in("PLACE")
    sm.base = {"blue": 0.0, "orange": 0.0}
    sm.step(ev(kraft={"blue": 0.0, "orange": 0.004}))
    sm.step(ev(kraft={"blue": 0.0, "orange": 0.004}))
    assert sm.state == "SURVEY" and sm.wrong == {A} and A in sm.sorted
    assert sm.flags == ["wrong_bin:%s:orange" % A]


def test_place_code_seen_inside_bin_counts_without_kraft():
    sm = sm_in("PLACE")
    sm.step(ev([(A, (800, 200))]))
    sm.step(ev([(A, (800, 200))]))
    assert sm.state == "SURVEY" and sm.sorted == {A}


def test_place_budget_then_give_up_after_retries():
    sm = sm_in("PLACE")
    for _ in range(CFG["place_max_calls"]):
        sm.step(ev([(A, (640, 450))]))
    assert sm.state == "SURVEY" and sm.retries == {A: 1} and not sm.gave_up
    sm.step(ev([(A, (640, 450))]))
    assert sm.state == "PLACE"  # retry
    for _ in range(CFG["place_max_calls"]):
        sm.step(ev([(A, (640, 450))]))
    assert sm.gave_up == {A} and sm.state == "SURVEY"
    assert sm.step(ev([(A, (640, 450))], blocks=[(640, 450)])) == "pick up the block in the middle"  # FETCH-able


def hold(sm, grip=0.46):
    sm.step(ev(grips=(0.2, 1.0)))
    sm.step(ev(grips=(0.2, grip)))
    sm.step(ev(grips=(0.2, grip)))


def test_fetch_wrist_code_while_holding_goes_to_place_held():
    sm = sm_in("FETCH", None)
    sm.step(ev(wrist={B}, grips=(0.2, 1.0)))
    assert sm.state == "FETCH"  # not holding yet
    sm.step(ev(wrist={B}, grips=(0.2, 0.46)))
    assert sm.step(ev(wrist={B}, grips=(0.2, 0.46))) == "put the block into the orange bin" and sm.state == "PLACE_HELD"


def test_fetch_holding_arm_can_be_left():
    sm = sm_in("FETCH", None)
    sm.step(ev(grips=(1.0, 0.2)))
    sm.step(ev(wrist={A}, grips=(0.46, 0.2)))
    sm.step(ev(wrist={A}, grips=(0.46, 0.2)))
    assert sm.state == "PLACE_HELD" and sm.code == A


def test_fetch_idle_plateau_is_not_a_hold():
    sm = sm_in("FETCH", None)
    for _ in range(4):
        sm.step(ev(wrist={A}, grips=(0.2, 0.2)))  # never opened: idle 0.2 plateau
    assert sm.state == "FETCH"


def test_fetch_budget_reject_if_holding_else_survey():
    sm = sm_in("FETCH", None)
    hold(sm)
    for _ in range(CFG["fetch_max_calls"]):
        sm.step(ev(grips=(0.2, 0.46)))
    assert sm.state == "REJECT"
    sm = sm_in("FETCH", None)
    for _ in range(CFG["fetch_max_calls"]):
        sm.step(ev())
    assert sm.state == "SURVEY"


def test_reject_on_release_and_budget():
    sm = sm_in("FETCH", None)
    hold(sm)
    for _ in range(CFG["fetch_max_calls"]):
        sm.step(ev(grips=(0.2, 0.46)))
    assert sm.step(ev(grips=(0.2, 0.46))) == "put the block at the left edge of the table"
    sm.step(ev(grips=(0.2, 1.0)))
    assert sm.state == "SURVEY" and sm.rejected_count == 1
    sm = sm_in("REJECT", None)
    for _ in range(CFG["reject_max_calls"]):
        sm.step(ev())
    assert sm.state == "SURVEY" and sm.rejected_count == 0


def test_rejected_block_is_ignored_afterwards():
    sm = sm_in("REJECT", None)
    sm.before = [(640, 450)]
    sm.step(ev(grips=(0.2, 0.46), blocks=[(640, 450)]))
    sm.arms[1].update(held=True, open=True)
    sm.step(ev(grips=(0.2, 1.0), blocks=[(640, 450)]))
    assert sm.state == "SURVEY" and sm.learn
    sm.step(ev(blocks=[(640, 450), (60, 400)]))  # the rejected cube now sits at the left edge
    assert sm.state == "FETCH" and sm.ignored == [(60, 400)]
    sm.state = "SURVEY"
    assert sm.step(ev(blocks=[(60, 400)])) == "put both arms back to the home position"


def test_place_held_release_plus_kraft_in_bin():
    sm = sm_in("PLACE_HELD", B)
    sm.base = {"blue": 0.0, "orange": 0.0}
    sm.step(ev(grips=(0.2, 1.0)))
    sm.arms[1].update(held=True, open=True)
    sm.step(ev(grips=(0.2, 0.46), kraft={"blue": 0.0, "orange": 0.005}))
    assert sm.state == "PLACE_HELD"  # cube over the bin but not released
    sm.step(ev(grips=(0.2, 1.0), kraft={"blue": 0.0, "orange": 0.005}))
    assert sm.state == "SURVEY" and sm.sorted == {B} and not sm.wrong


def test_place_held_wrong_bin_and_budget():
    sm = sm_in("PLACE_HELD", B)
    sm.base = {"blue": 0.0, "orange": 0.0}
    sm.arms[0].update(held=True, open=True)
    sm.step(ev(grips=(1.0, 0.2), kraft={"blue": 0.005, "orange": 0.0}))  # left arm releases
    assert sm.wrong == {B}
    sm = sm_in("PLACE_HELD", B)
    for _ in range(CFG["place_held_max_calls"]):
        sm.step(ev())
    assert sm.state == "SURVEY" and not sm.sorted


def test_gripper_both_arms_and_drop():
    sm = StateMachine(CFG)
    sm.step(ev(grips=(1.0, 0.2)))
    sm.step(ev(grips=(0.46, 0.2)))
    sm.step(ev(grips=(0.47, 0.2)))
    assert [a["held"] for a in sm.arms] == [True, False]
    sm.step(ev(grips=(0.0, 0.2)))  # closed on nothing: dropped
    assert not sm.arms[0]["held"] and not sm.arms[0]["open"]
    sm.step(ev(grips=(1.0, 0.2)))
    sm.step(ev(grips=(0.46, 0.2)))
    sm.step(ev(grips=(0.7, 0.2)))  # not steady, and above hold band
    assert not sm.arms[0]["held"]
