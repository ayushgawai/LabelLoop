import json
from pathlib import Path

from labelloop.config import load_yaml
from labelloop.eval.runlog import RunLog
from labelloop.harness import Harness
from labelloop.robots.mock import A, B, BAD, BIN, UNKNOWN, FakeScene, garbage, nolabel, parcel, qr_image

CFG = load_yaml(Path(__file__).parent.parent / "configs" / "task_qr_sort.yaml")


def run(scene, cfg=CFG, runlog=None, max_calls=1000, until=lambda h: h.done):
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
    objs = [parcel(A), parcel(B, fail_places=2, hidden=2), parcel(UNKNOWN), nolabel(), garbage()]
    scene, log = FakeScene(objs), tmp_path / "sub" / "run.jsonl"
    h, sent = run(scene, runlog=RunLog(str(log)), until=lambda h: h.sm.parcels == 5)
    assert [scene.at[id(o)] for o in objs] == ["blue", "red", "reject", "reject", "reject"]
    assert [r["outcome"] for r in h.sm.results] == ["ok"] * 5 and not h.done
    instrs = [i for _, i in sent if i]
    assert "put the block into the blue bin with the right arm" in instrs and "put the block into the red bin with the right arm" in instrs
    assert "put the object into the black bin with the right arm" in instrs
    assert "hand the object from the left arm to the right arm" in instrs and "pick up the block with the left arm and turn it over" in instrs
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


def test_object_that_leaves_zone_without_reaching_reject_bin_is_not_ok():
    class Dropper(FakeScene):  # REJECT clears the zone but the object lands elsewhere
        def apply(self, instruction):
            o = self.zone
            super().apply(instruction)
            if o and self.at[id(o)] == "reject":
                self.at[id(o)] = "lost"

    scene = Dropper([parcel(BAD)])
    h, sent = run(scene, {**CFG, "max_parcels": 1})
    assert [r["outcome"] for r in h.sm.results] == ["failed"]
    entries = [b for (a, _), (b, _) in zip([(None, 0)] + sent, sent) if b == "REJECT" and a != "REJECT"]
    assert len(entries) == CFG["tries"]["REJECT"]  # retried, then gave up


def states(sent):
    return [s for i, (s, _) in enumerate(sent) if not i or s != sent[i - 1][0]]


def test_hidden_label_found_on_second_inspect_try_routes_to_its_lane():
    scene = FakeScene([parcel(A, hidden=2)])
    h, sent = run(scene, {**CFG, "max_parcels": 1})
    assert scene.at[id(scene.log[0])] == "blue" and h.sm.results[0]["code"] == A
    assert states(sent)[:5] == ["IDLE", "SCAN", "INSPECT", "HANDOVER", "PLACE"]  # the 2nd try (new wording) reveals it
    assert [i for s, i in sent if s == "INSPECT"][-1] == CFG["instructions"]["INSPECT"][1].format(obj="block")


def test_label_never_found_is_no_label_reject_via_handover():
    scene = FakeScene([nolabel()])
    h, sent = run(scene, {**CFG, "max_parcels": 1})
    assert scene.at[id(scene.log[0])] == "reject" and h.sm.results[0]["reason"] == "no_label"
    assert sum(s == "INSPECT" for s, _ in sent) == CFG["tries"]["INSPECT"] * CFG["max_calls"]["INSPECT"]
    assert states(sent)[2:5] == ["INSPECT", "HANDOVER", "REJECT"]


def test_big_parcel_is_lifted_by_both_arms_and_garbage_is_handed_over():
    scene = FakeScene([parcel(A, big=True), garbage()])
    h, sent = run(scene, until=lambda h: h.sm.parcels == 2)
    assert [scene.at[id(o)] for o in scene.log] == ["blue", "reject"]
    assert states(sent)[:9] == ["IDLE", "SCAN", "LIFT2", "HOME", "VERIFY", "IDLE", "SCAN", "HANDOVER", "REJECT"]
    assert "pick up the block with both arms and put it into the blue bin" in [i for _, i in sent]


def test_codes_inside_bins_are_ignored_while_inspecting_but_count_elsewhere():
    h = Harness(CFG)
    scene = FakeScene([])
    imgs, state = scene.observe()
    bx0, by0 = BIN[0], BIN[1]
    imgs["front"][by0 + 20:by0 + 90, bx0 + 20:bx0 + 90] = qr_image(A)  # labelled parcel lying in the reject bin
    imgs["front"][60:130, 30:100] = qr_image(B)  # label in the open (e.g. on the parcel held by the arm)
    h.step(imgs, state), h.step(imgs, state)
    h.sm.state = "INSPECT"
    h.step(imgs, state), h.step(imgs, state)
    assert h.confirmer.counts.keys() == {B}
    assert h.sm.state == "HANDOVER" and h.sm.code == B
