import json
from pathlib import Path

from labelloop.config import load_yaml
from labelloop.eval.runlog import RunLog
from labelloop.harness import Harness
from labelloop.robots.mock import A, B, BAD, UNKNOWN, FakeScene, garbage, nolabel, parcel

CFG = load_yaml(Path(__file__).parent.parent / "configs" / "task_qr_sort.yaml")


def run(scene, cfg=CFG, runlog=None, max_calls=500, until=lambda h: h.done):
    h, sent = Harness(cfg, runlog=runlog), []
    for _ in range(max_calls):
        if until(h):
            break
        instr = h.step(*scene.observe())
        sent.append((h.sm.state, instr))
        scene.apply(instr)
        h.log_reply(0.01)
    return h, sent


def test_stream_of_five_routes_each_parcel_and_never_latches_done(tmp_path):
    objs = [parcel(A), parcel(B, fail_places=2), parcel(UNKNOWN), nolabel(), garbage()]
    scene, log = FakeScene(objs), tmp_path / "sub" / "run.jsonl"
    h, sent = run(scene, runlog=RunLog(str(log)), until=lambda h: h.sm.parcels == 5)
    assert [scene.at[id(o)] for o in objs] == ["blue", "orange", "reject", "reject", "reject"]
    assert [r["outcome"] for r in h.sm.results] == ["ok"] * 5 and not h.done
    instrs = [i for _, i in sent if i]
    assert "put the block into the blue bin" in instrs and "put the block into the orange bin" in instrs
    assert "put the object into the black bin" in instrs
    for _ in range(20):  # keeps waiting for the next parcel, silently
        assert h.step(*scene.observe()) is None and not h.done
    lines = [json.loads(l) for l in log.read_text().splitlines()]
    steps = [l for l in lines if "state" in l]
    assert len(steps) == h.calls and sum("latency_s" in l for l in lines) == h.calls - 20
    assert {"codes", "lane", "presence", "kraft_roi", "mats", "kraft_mats", "transitions"} <= set(steps[0])


def test_bad_code_goes_to_reject_and_max_parcels_done():
    scene = FakeScene([parcel(BAD)])
    h, _ = run(scene, {**CFG, "max_parcels": 1})
    assert h.done and scene.at[id(scene.log[0])] == "reject"
    h.reset()
    assert h.state_name == "IDLE" and not h.done and h.calls == 0
