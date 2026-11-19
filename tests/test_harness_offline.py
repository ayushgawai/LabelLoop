import json
from pathlib import Path

from labelloop.config import load_yaml
from labelloop.eval.runlog import RunLog
from labelloop.harness import Harness
from labelloop.robots.mock import A, B, C, FakeAgent, FakeScene, cube

CFG = load_yaml(Path(__file__).parent.parent / "configs" / "task_qr_sort.yaml")


def run(scene, agent=None, runlog=None, max_calls=120):
    h = Harness(CFG, agent, runlog)
    sent = []
    for _ in range(max_calls):
        if h.done:
            break
        instr = h.step(*scene.observe())
        sent.append((h.sm.state, instr))
        scene.apply(instr)
        h.log_reply(0.01)
    return h, sent


def bins(scene):
    return {c["code"]: c["at"] for c in scene.cubes}


def test_one_cube(tmp_path):
    scene, log = FakeScene([cube(A)]), tmp_path / "sub" / "run.jsonl"
    h, sent = run(scene, runlog=RunLog(str(log)))
    assert h.done and h.sm.sorted == {A} and scene.bin_of == {A: "blue"} and scene.picks == 1
    assert sent[0] == ("SURVEY", None)  # holds one call until the code is confirmed
    assert sent[1] == ("PLACE", "put the block on the left into the blue bin")
    lines = [json.loads(l) for l in log.read_text().splitlines()]
    steps = [l for l in lines if "state" in l]
    assert len(steps) == h.calls and sum("latency_s" in l for l in lines) == h.calls
    assert steps[1]["codes"][0][0] == A and steps[1]["bins"]["blue"] and steps[1]["blocks"] == 1


def test_two_cubes_nearest_centre_first():
    scene = FakeScene([cube(A), cube(B)])
    h, sent = run(scene, FakeAgent(scene))
    assert h.done and scene.bin_of == {A: "blue", B: "orange"} and scene.picks == 2
    assert "put the block in the middle into the orange bin" in [i for _, i in sent]


def test_hidden_label_read_by_wrist_after_pick():
    scene = FakeScene([cube(A), cube(B), cube(C, "side")])
    h, sent = run(scene)
    assert h.done and scene.bin_of == {A: "blue", B: "orange", C: "blue"} and scene.picks == 3
    assert any(s == "FETCH" for s, _ in sent) and any(s == "PLACE_HELD" for s, _ in sent)


def test_decoy_rejected_no_repicks():
    scene = FakeScene()  # A top, B hidden label, plain decoy
    h, sent = run(scene)
    assert h.done and scene.bin_of == {A: "blue", B: "orange"} and h.sm.rejected_count == 1
    assert [c["at"] for c in scene.cubes] == ["blue", "orange", "edge"] and scene.picks == 3
    assert any("left edge" in (i or "") for _, i in sent)


def test_wrong_bin_detected_and_logged(tmp_path):
    log = tmp_path / "run.jsonl"
    scene = FakeScene([cube(A)], wrong_bin=True)
    h, _ = run(scene, runlog=RunLog(str(log)))
    assert h.done and h.sm.wrong == {A} and scene.bin_of == {A: "orange"}
    flags = [f for l in log.read_text().splitlines() for f in json.loads(l).get("flags", [])]
    assert flags == ["wrong_bin:%s:orange" % A]


def test_agent_zero_does_not_block_visible_target_and_reset():
    scene = FakeScene([cube(A)])
    h = Harness(CFG, FakeAgent(scene))
    h.step(*scene.observe())
    h.step(*scene.observe())
    assert h.state_name == "PLACE" and h.subgoal_changed
    h.reset()
    assert h.state_name == "SURVEY" and not h.done and h.calls == 0
