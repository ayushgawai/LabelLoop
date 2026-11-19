import json
from pathlib import Path

from labelloop.config import load_yaml
from labelloop.eval.runlog import RunLog
from labelloop.harness import Harness
from labelloop.robots.mock import A, B, FakeAgent, FakeScene

CFG = load_yaml(Path(__file__).parent.parent / "configs" / "task_qr_sort.yaml")


def run(agent_factory, runlog=None, max_calls=200):
    scene, t = FakeScene(), [0.0]
    h = Harness(CFG, agent_factory(scene), runlog, clock=lambda: t[0])
    sent, grasps = [], 0
    for _ in range(max_calls):
        if h.done:
            break
        images, state = scene.observe()
        instr = h.step(images, state)
        grasps += h.state_name == "GRASP" and h.subgoal_changed
        sent.append((h.sm.state, instr))
        scene.apply(instr)
        h.log_reply(0.01)
        t[0] += 0.5
    return scene, h, sent, grasps


def test_full_episode(tmp_path):
    log = tmp_path / "sub" / "run.jsonl"
    scene, h, sent, grasps = run(FakeAgent, RunLog(str(log)))
    m = h.sm.memory
    assert h.done and h.state_name == "DONE"
    assert m.sorted == {A, B} and m.rejected_count == 1
    assert scene.bin_of == {A: "blue", B: "orange"}
    assert any("blue" in i for _, i in sent if i) and any("orange" in i for _, i in sent if i)
    assert any(s == "PRESENT" for s, _ in sent) and any(s == "REJECT" for s, _ in sent)
    assert grasps == 3  # A, B, C each picked exactly once: bin-visible codes never re-picked
    assert [b["at"] for b in scene.boxes] == ["bin", "bin", "left"]
    lines = [json.loads(l) for l in log.read_text().splitlines()]
    steps = [l for l in lines if "state" in l]
    assert len(steps) == h.calls and steps[0]["state"] == "SURVEY" and "t" in steps[0]
    assert any(l.get("latency_s") == 0.01 for l in lines)


def test_agent_returning_none_still_finishes_or_holds_safely():
    class NoAgent:
        def ask(self, *a):
            return None

    scene, h, sent, _ = run(lambda s: NoAgent(), max_calls=60)
    assert h.calls > 0 and A in h.sm.memory.sorted  # no crash; progress without the agent


def test_no_agent_and_reset():
    h = Harness(CFG)
    scene = FakeScene()
    h.step(*scene.observe())
    assert h.state_name == "LOOK" and h.subgoal_changed
    h.reset()
    assert h.state_name == "SURVEY" and not h.done and h.calls == 0
