from pathlib import Path

import pytest

from labelloop.brain.instructions import instruction
from labelloop.brain.state_machine import StateMachine
from labelloop.config import load_yaml

CFG = load_yaml(Path(__file__).parent.parent / "configs" / "task_qr_sort.yaml")
A, B = "LL|ZONE-A|001", "LL|ZONE-B|002"


def at(state, code=None):
    sm = StateMachine(CFG)
    sm._enter(state, 0.0)
    sm.memory.current_code = code
    if code:
        sm.memory.seen[code] = "ZONE-A"
    return sm


def go(sm, now=1.0, grip=1.5, codes=(), verdict=None):
    return sm.step({"now": now, "grip": grip, "codes": list(codes), "verdict": verdict})


def test_instruction_variant_and_bin():
    assert instruction("PLACE", CFG, "blue") == "Put the box into the blue bin."
    assert instruction("PLACE", {**CFG, "instruction_variant": 4}, "blue").startswith("Place the box in the blue")


def test_survey_holds_then_looks_and_agent_none_is_fine():
    sm = StateMachine(CFG)
    assert sm.state == "SURVEY" and sm.instruction() is None
    assert go(sm, verdict=None) == instruction("LOOK", CFG) and sm.state == "LOOK"


def test_survey_zero_boxes_is_done_and_stays():
    sm = StateMachine(CFG)
    assert go(sm, verdict={"unsorted_boxes": 0}) == instruction("DONE", CFG)
    go(sm, codes=[(A, "front")])
    assert sm.state == "DONE" and sm.memory.seen[A] == "ZONE-A"


def test_survey_stores_estimate():
    sm = StateMachine(CFG)
    go(sm, verdict={"unsorted_boxes": 3, "bin_side": {"blue": "left"}})
    assert sm.memory.unsorted_estimate == 3 and sm.memory.bin_side == {"blue": "left"}


def test_look_prefers_right_wrist():
    sm = at("LOOK")
    go(sm, codes=[(B, "front"), (A, "right_wrist")])
    assert sm.state == "GRASP" and sm.memory.current_code == A


def test_look_gives_up_after_attempts():
    sm = at("LOOK")
    for i in range(11):
        go(sm)
        assert sm.state == "LOOK"
    go(sm)
    assert sm.state == "GRASP" and sm.memory.current_code is None


def test_codes_in_bins_and_malformed_are_ignored():
    sm = at("LOOK")
    sm.memory.sorted.add(A)
    go(sm, codes=[(A, "right_wrist"), ("hello", "right_wrist"), ("LL|ZONE-C|001", "front")])
    assert sm.state == "LOOK" and sm.memory.seen == {}


def test_present_ignores_front_and_uses_wrist():
    sm = at("PRESENT")
    go(sm, codes=[(B, "front")])
    assert sm.state == "PRESENT"
    go(sm, codes=[(B, "right_wrist")])
    assert sm.state == "PLACE" and instruction("PLACE", CFG, "orange") == sm.instruction()


def test_present_gives_up_to_reject():
    sm = at("PRESENT")
    for _ in range(12):
        go(sm)
    assert sm.state == "REJECT"


def test_grasp_hold_goes_to_place_or_present():
    sm = at("GRASP", A)
    go(sm, grip=0.35), go(sm, grip=0.35)
    assert sm.state == "GRASP"  # never opened first: idle gripper, not a grasp
    go(sm, grip=0.9), go(sm, grip=0.35)
    assert sm.state == "GRASP"
    assert go(sm, grip=0.35) == "Put the box into the blue bin."
    sm = at("GRASP")
    go(sm, grip=0.9), go(sm, grip=0.35), go(sm, grip=0.35)
    assert sm.state == "PRESENT"


def test_grasp_closed_on_nothing_retries_then_survey():
    sm = at("GRASP", A)
    for i in range(3):
        assert sm.state == "GRASP"
        go(sm, grip=1.5)
        go(sm, grip=0.05)
    assert sm.state == "SURVEY" and sm.memory.current_code is None


def test_grasp_closed_without_ever_opening_is_not_a_retry():
    sm = at("GRASP", A)
    for _ in range(5):
        go(sm, grip=0.05)
    assert sm.state == "GRASP" and sm.retries == 0


def test_grasp_agent_done_but_not_holding_counts_retry():
    sm = at("GRASP", A)
    for _ in range(3):
        go(sm, grip=1.5, verdict={"done": True})
    assert sm.state == "SURVEY"


def test_place_release_marks_sorted():
    sm = at("PLACE", A)
    go(sm, grip=0.35)
    assert sm.state == "PLACE"
    go(sm, grip=1.5)
    assert sm.state == "SURVEY" and A in sm.memory.sorted


def test_place_drop_is_not_sorted():
    sm = at("PLACE", A)
    go(sm, grip=0.05)
    assert sm.state == "SURVEY" and A not in sm.memory.sorted


def test_reject_release_counts():
    sm = at("REJECT")
    go(sm, grip=1.5)
    assert sm.state == "SURVEY" and sm.memory.rejected_count == 1


@pytest.mark.parametrize("state,grip,expect", [("LOOK", 1.5, "GRASP"), ("GRASP", 1.5, "SURVEY"),
                                               ("PRESENT", 1.5, "REJECT"), ("PLACE", 0.35, "SURVEY"),
                                               ("REJECT", 0.35, "SURVEY")])
def test_state_timeout(state, grip, expect):
    sm = at(state, A)
    sm.cfg = {**CFG, "state_max_calls": 5}  # below the LOOK/PRESENT attempt budgets
    for _ in range(4):
        go(sm, grip=grip)
    assert sm.state == state
    go(sm, grip=grip)
    assert sm.state == expect and sm.transitions[-1][2] == "timeout"


def test_look_ignores_front_codes_but_remembers_them():
    sm = at("LOOK")
    go(sm, codes=[(A, "front")])
    assert sm.state == "LOOK" and A in sm.memory.seen


def test_reject_drop_goes_to_survey_without_counting():
    sm = at("REJECT")
    go(sm, grip=0.05)
    assert sm.state == "SURVEY" and sm.memory.rejected_count == 0


def test_grasp_needs_steady_reading_not_a_passing_value():
    sm = at("GRASP", A)
    for g in (0.9, 0.2, 0.4, 0.55, 0.3):  # opened, then moving through the band = not holding
        go(sm, grip=g)
    assert sm.state == "GRASP"
    go(sm, grip=0.31)
    assert sm.state == "PLACE"  # 0.30 -> 0.31 is steady
