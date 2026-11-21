import json

import numpy as np
import pytest
from pathlib import Path

from labelloop.config import load_yaml
from labelloop.eval.runlog import RunLog
from labelloop import look
from labelloop.harness import Harness
from labelloop.robots.mock import A, B, BAD, BINS, UNKNOWN, FakeScene, garbage, nolabel, parcel, qr_image

BASE = load_yaml(Path(__file__).parent.parent / "configs" / "task_qr_sort.yaml")
CFG = {**BASE, "look": {"enabled": False}, "spin": {"enabled": False}, "handover": True, "tries": {"PUSH": 0, "INSPECT": 3, "HANDOVER": 3, "PLACE": 3, "REJECT": 3, "LIFT2": 3}, "budget": {"case_s": 1e9, "deliver_reserve_s": 0, "wallx_call_s": 11.0, "scripted_call_s": 2.2}}  # the v5 flow
NOLOOK = CFG
LOOKCFG = {**CFG, "look": {**BASE["look"], "enabled": True}}


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
    assert [scene.at[id(o)] for o in objs] == ["blue", "red", "black", "black", "black"]
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
    assert {"codes", "lane", "presence", "kraft_roi", "bin_change", "transitions"} <= set(steps[0])


def test_bad_code_goes_to_reject_and_max_parcels_done():
    scene = FakeScene([parcel(BAD)])
    h, _ = run(scene, {**CFG, "max_parcels": 1})
    assert h.done and scene.at[id(scene.log[0])] == "black"
    h.reset()
    assert h.state_name == "IDLE" and not h.done and h.calls == 0


def test_object_that_leaves_zone_without_reaching_reject_bin_is_not_ok():
    class Dropper(FakeScene):  # REJECT clears the zone but the object lands elsewhere
        def apply(self, instruction):
            o = self.zone
            super().apply(instruction)
            if o and self.at[id(o)] == "black":
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
    h, sent = run(scene, {**NOLOOK, "max_parcels": 1})
    assert scene.at[id(scene.log[0])] == "blue" and h.sm.results[0]["code"] == A
    assert states(sent)[:5] == ["IDLE", "SCAN", "INSPECT", "HANDOVER", "PLACE"]  # the 2nd try (new wording) reveals it
    assert [i for s, i in sent if s == "INSPECT"][-1] == CFG["instructions"]["INSPECT"][1].format(obj="block")


def test_label_never_found_is_no_label_reject_via_handover():
    scene = FakeScene([nolabel()])
    h, sent = run(scene, {**NOLOOK, "max_parcels": 1})
    assert scene.at[id(scene.log[0])] == "black" and h.sm.results[0]["reason"] == "no_label"
    assert sum(s == "INSPECT" for s, _ in sent) == CFG["tries"]["INSPECT"] * CFG["max_calls"]["INSPECT"]
    assert states(sent)[2:5] == ["INSPECT", "HANDOVER", "REJECT"]


def test_big_parcel_is_lifted_by_both_arms_and_garbage_is_handed_over():
    scene = FakeScene([parcel(A, big=True), garbage()])
    h, sent = run(scene, until=lambda h: h.sm.parcels == 2)
    assert [scene.at[id(o)] for o in scene.log] == ["blue", "black"]
    assert states(sent)[:9] == ["IDLE", "SCAN", "LIFT2", "HOME", "VERIFY", "IDLE", "SCAN", "HANDOVER", "REJECT"]
    assert "pick up the block with both arms and put it into the blue bin" in [i for _, i in sent]


def test_codes_inside_bins_are_ignored_while_inspecting_but_count_elsewhere():
    h = Harness(CFG)
    scene = FakeScene([])
    imgs, state = scene.observe()
    bx0, by0 = BINS["black"][:2]
    imgs["front"][by0 + 20:by0 + 90, bx0 + 20:bx0 + 90] = qr_image(A)  # labelled parcel lying in the reject bin
    imgs["front"][60:130, 30:100] = qr_image(B)  # label in the open (e.g. on the parcel held by the arm)
    h.step(imgs, state), h.step(imgs, state)
    h.sm.state = "INSPECT"
    h.step(imgs, state), h.step(imgs, state)
    assert h.confirmer.counts.keys() == {("front", B)}
    assert h.sm.state == "HANDOVER" and h.sm.code == B


FAKE = [("left_ee", np.array([0.1, 0.2, 0.2]), np.array([0, 0.5, 0.0]), {"ik_err": 0.0, "clear": 0.1}),
        ("right_ee", np.array([0.15, 0.1, 0.2]), np.array([0, 0.5, 0.4]), {"ik_err": 0.0, "clear": 0.1}),
        ("left_ee", np.array([0.2, 0.0, 0.2]), np.array([0, 0.5, -0.4]), {"ik_err": 0.0, "clear": 0.1})]


def run_look(scene, **kw):  # like run(), but tells the mock which look-around view the wrist camera is at
    h, sent = Harness(LOOKCFG), []
    while not h.done and len(sent) < 400:
        scene.view = h.look_index
        instr = h.step(*scene.observe())
        sent.append((h.sm.state, instr))
        scene.apply(instr)
        if h.sm.parcels:
            break
    return h, sent


def test_look_finds_label_on_a_wrist_view_and_skips_inspect(monkeypatch):
    monkeypatch.setattr(look, "views", lambda p, cfg, hw=None: FAKE)  # flow only; real IK is covered in test_look.py
    scene = FakeScene([parcel(A, hidden=99, look=2)])
    h, sent = run_look(scene)
    ss = states(sent)
    assert h.sm.results[0]["code"] == A and scene.at[id(scene.log[0])] == "blue" and "INSPECT" not in ss
    assert ss.index("LOOKHOME") > ss.index("LOOK") and ss.index("HANDOVER") > ss.index("LOOKHOME")  # home first, then the route
    assert all(i is None for s, i in sent if s in ("LOOK", "LOOKHOME")) and len(h.views) == 3


def test_look_sees_nothing_then_inspect_then_no_label_reject(monkeypatch):
    monkeypatch.setattr(look, "views", lambda p, cfg, hw=None: FAKE)
    scene = FakeScene([parcel(A, hidden=99)])
    h, sent = run_look(scene)
    ss = states(sent)
    assert ss.index("LOOK") < ss.index("LOOKHOME") < ss.index("INSPECT") and h.sm.results[0]["reason"] == "no_label"
    assert sum(s == "LOOK" for s, _ in sent) == 3 * h.look_span() and scene.at[id(scene.log[0])] == "black"


def test_scripted_reply_drives_the_view_arm_via_a_lift_and_sends_the_other_arm_home(monkeypatch):
    monkeypatch.setattr(look, "views", lambda p, cfg, hw=None: FAKE)
    lc, scene, h = LOOKCFG["look"], FakeScene([parcel(A, hidden=99)]), Harness(LOOKCFG)
    seen = set()
    for _ in range(100):
        imgs, st = scene.observe()
        h.step(imgs, st)
        scene.apply(None)
        if h.sm.state != "LOOK":
            continue
        st["left_ee"]["position"], st["right_ee"]["position"] = [0.3, 0.3, 0.1], [0.4, -0.3, 0.1]
        r = h.scripted_reply("e", st)["actions"]
        arm, pos, rpy, _ = FAKE[h.look_index]
        other = "right_ee" if arm == "left_ee" else "left_ee"
        k = h.sm.calls % h.look_span()
        want = [st[arm]["position"][0], st[arm]["position"][1], look.TABLE_Z + 0.30 - lc["arm_z"]] if k < lc["via_calls"] else pos
        assert np.allclose(r[arm]["position"][-1], want) and np.allclose(r[other]["position"][-1], h.home[other]["position"])
        seen.add((h.look_index, k < lc["via_calls"]))
    assert seen == {(i, v) for i in range(3) for v in (True, False)}


def test_point_test_holds_the_left_fingertips_over_the_parcel_without_wallx(monkeypatch):
    monkeypatch.setattr(look, "views", lambda *a, **k: pytest.fail("no views in point_test"))
    cfg = {**CFG, "look": {**BASE["look"], "enabled": True, "point_test": True}}
    scene, h = FakeScene([parcel(A, hidden=99)]), Harness(cfg, runlog=None)
    for _ in range(60):
        imgs, st = scene.observe()
        h.step(imgs, st)
        scene.apply(None)
        if h.sm.state == "LOOK":
            break
    assert h.sm.state == "LOOK" and h.point and h.look_info["point"][2] == pytest.approx(0.065)
    arm, ee, rpy = h.point
    tip = look.tip_of(arm, ee, rpy, cfg["look"])
    assert arm == "left_ee" and np.allclose(tip[:2], h.look_info["point"][:2], atol=0.01) and tip[2] == pytest.approx(0.165, abs=0.01)
    for _ in range(20):  # holds: never leaves LOOK on its own
        h.step(*scene.observe())
    assert h.sm.state == "LOOK" and h.scripted_reply("e", scene.observe()[1]) is not None
