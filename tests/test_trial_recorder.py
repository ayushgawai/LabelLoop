import base64
import json

from labelloop.robots.trial_recorder import TrialRecorder, reduce_actions

JPEG = b"\xff\xd8\xff\xe0fake-jpeg\xff\xd9"
STATE = {"left_ee": {"position": [0.1, 0.2, 0.3], "rotation_rpy": [0.0, 0.0, 0.0], "gripper": 0.1},
         "right_ee": {"position": [0.1, -0.2, 0.3], "rotation_rpy": [0.0, 0.0, 0.0], "gripper": 0.2}}


def reply(n=32):
    return {"embodiment": "teletwin_desktop", "actions": {
        "left_ee": {"position": [[0.0, 0.1, i * 0.001] for i in range(n)],
                    "rotation_rpy": [[0.0, 0.0, 0.0] for _ in range(n)],
                    "gripper": [0.0] * n},
        "right_ee": {"position": [[0.0, -0.1, i * 0.001] for i in range(n)],
                     "rotation_rpy": [[0.0, 0.0, 0.0] for _ in range(n)],
                     "gripper": [0.1] * n}}}


def test_reduce_actions_evenly_spaced_per_arm():
    a = reduce_actions(reply())
    assert set(a) == {"left_ee", "right_ee"}
    assert len(a["left_ee"]["position"]) == 8
    assert len(a["left_ee"]["gripper"]) == 8 and len(a["right_ee"]["rotation_rpy"]) == 8
    assert a["left_ee"]["position"][0] == [0.0, 0.1, 0.0]
    assert a["left_ee"]["position"][-1] == [0.0, 0.1, 0.031]


def test_reduce_actions_odd_reply_is_none():
    assert reduce_actions(None) is None
    assert reduce_actions({"error": "boom"}) is None
    assert reduce_actions({"actions": {}}) is None
    assert reduce_actions({"actions": {"left_ee": {"position": [[0.0, 0.0, 0.0]]}}}) is None
    assert reduce_actions({"actions": {"left_ee": {"position": [[0.0, 0.0, 0.0]],
                                                   "rotation_rpy": [[0.0, 0.0, 0.0]],
                                                   "gripper": []}}}) is None


def test_recorder_writes_rows_frames_and_summary(tmp_path):
    rec = TrialRecorder(tmp_path)
    tid = rec.start("sort the boxes", profile="real", fast=True)
    d = tmp_path / tid
    assert d.is_dir() and (d / "frames").is_dir()

    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    assert meta["goal"] == "sort the boxes"
    assert meta["profile"] == "real" and meta["fast"] is True
    assert meta["start"] and meta["start_epoch"] > 0

    b64 = base64.b64encode(JPEG).decode("ascii")
    assert rec.frame(0, {"front": b64, "left_wrist": b64}) == ["front", "left_wrist"]
    assert rec.frame(1, {"front": "data:image/jpeg;base64," + b64}) == ["front"]
    assert (d / "frames" / "0_front.jpg").read_bytes() == JPEG
    assert (d / "frames" / "1_front.jpg").read_bytes() == JPEG

    rec.record(t=100.0, index=0, goal="sort the boxes", subgoal="find the label", source="wallx",
               harness={"state": "SCAN", "parcels": 0, "codes": []}, state=STATE,
               latency_ms=70.0, elapsed_ms=95.0, error=None, actions=reduce_actions(reply()))
    rec.record(t=101.0, index=1, goal="sort the boxes", subgoal=None, source="scripted",
               harness={"state": "HOME", "parcels": 0}, state=STATE,
               latency_ms=None, elapsed_ms=40.0, error=None, actions=None)

    lines = (d / "trial.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["source"] == "wallx" and first["harness"]["state"] == "SCAN"
    assert first["state"]["left_ee"]["position"] == [0.1, 0.2, 0.3]
    assert len(first["actions"]["right_ee"]["position"]) == 8
    assert json.loads(lines[1])["actions"] is None

    summary = rec.end()
    assert summary["calls"] == 2 and summary["wallx_calls"] == 1
    assert summary["mean_latency_ms"] == 70.0 and summary["p95_latency_ms"] == 70.0
    assert summary["parcels"] == 0 and summary["last_state"] == STATE
    assert (d / "summary.json").exists()
    assert json.loads((d / "summary.json").read_text(encoding="utf-8"))["calls"] == 2

    assert rec.record(index=2, source="wallx") is None  # ended: nothing more is written
    assert rec.frame(2, {"front": b64}) == []


def test_start_ends_the_previous_trial_in_its_own_folder(tmp_path):
    rec = TrialRecorder(tmp_path)
    a = rec.start("first")
    rec.record(t=1.0, index=0, source="hold", state=STATE)
    b = rec.start("second")
    assert a != b
    assert (tmp_path / a / "summary.json").exists()
    assert not (tmp_path / b / "summary.json").exists()
    rec.end()
