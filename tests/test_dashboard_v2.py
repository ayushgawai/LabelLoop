import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from labelloop import look
from labelloop.dashboard import kinematics as K
from labelloop.dashboard.demo import make_agent_run, make_trial, state_of
from labelloop.dashboard.server import make_server

TID = "20261121_130500"
QS = [[0.3, 1.0, -1.2, 0.2, 0.4, 0.1], [-0.4, 0.8, -0.9, -0.3, 0.5, -0.2], [0.1, 1.5, -1.8, 0.4, -0.6, 0.3], [0.6, 1.2, -0.7, 0.0, 0.9, -0.5]]


@pytest.fixture
def served(tmp_path):
    def start(root=None, **kw):
        httpd = make_server(root or tmp_path, port=0, **kw)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        servers.append(httpd)
        return "http://127.0.0.1:%d" % httpd.server_address[1]
    servers = []
    yield tmp_path, start
    for s in servers:
        s.shutdown()
        s.server_close()


def get_json(url):
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, None


@pytest.mark.parametrize("q", QS)
def test_fk_ik_round_trip(q):
    q = np.array(q)
    p, R = look.fk(q)
    j = K.joints("right_ee", state_of(q, 0.5))  # home-relative state of q, cold IK
    assert j["ok"] and j["err"] < 1e-3
    p2, R2 = look.fk(np.array(j["q"]))
    assert np.abs(p2 - p).max() < 1e-3 and np.abs(R2 - R).max() < 0.02
    assert np.allclose(np.array(j["points"][5]) - j["base"], p2, atol=2e-4)  # the 6th joint origin is the EE
    assert np.allclose(j["points"][6], np.array(j["points"][5]) + R2 @ look.TCP, atol=2e-4)  # then the fingertip
    assert j["gripper"] == pytest.approx(0.5 / K.GRIP_MAX, abs=1e-3)


def test_rest_pose_is_ok_with_joints_2_3_flagged_at_their_limits():
    rest = {"position": [0, 0, 0], "rotation_rpy": [0, 0, 0], "gripper": 0.0}
    for arm, y in (("left_ee", K.FRAME["arm_y"]), ("right_ee", -K.FRAME["arm_y"])):
        j = K.joints(arm, rest)
        assert j["ok"] and np.abs(j["q"]).max() < 1e-3
        assert np.allclose(j["points"][5], [0.1185, y, 0.181 + K.FRAME["arm_z"]], atol=1e-3)
        assert j["near"] == [False, True, True, False, False, False]  # LO[1] = 0, HI[2] = 0: the folded rest pose


def test_unreachable_pose_is_not_ok():
    j = K.joints("left_ee", {"position": [1.2, 0, 0], "rotation_rpy": [0, 0, 0], "gripper": 0})
    assert not j["ok"] and j["err"] > K.IK_POS and len(j["points"]) == 7


def test_joints_route_caches_and_extends_a_live_trial(served):
    root, start = served
    d = make_trial(root, TID, n=40, frames=False)
    url = start() + "/api/trials/%s/joints" % TID
    code, p = get_json(url)
    assert code == 200 and p["rows"] == 40 and len(p["joints"]) == 40 and (d / "joints.json").exists()
    assert all(e[a]["ok"] for e in p["joints"] for a in K.ARMS)
    assert len(p["lo"]) == 6 and p["frame"]["arm_y"] == K.FRAME["arm_y"]
    e = next(e for e in p["joints"] if e["pred"])  # a WALLX row: its predicted chunk as a fingertip path
    assert all(len(pt) == 3 for path in e["pred"].values() for pt in path)
    rows = (d / "trial.jsonl").read_text().splitlines()
    m = (d / "joints.json").stat().st_mtime_ns
    assert get_json(url)[1] == p and (d / "joints.json").stat().st_mtime_ns == m  # cached: not recomputed
    with open(d / "trial.jsonl", "a") as f:  # the live trial grows
        for k in range(5):
            r = json.loads(rows[-1])
            r["index"] = 40 + k
            f.write(json.dumps(r) + "\n")
    code, p2 = get_json(url)
    assert code == 200 and p2["rows"] == 45 and p2["joints"][:40] == p["joints"]


def test_rows_without_state_do_not_break_the_route(served):
    root, start = served
    d = make_trial(root, TID, n=30, frames=False)
    with open(d / "trial.jsonl", "a") as f:
        f.write(json.dumps({"index": 30, "state": None}) + "\n" + json.dumps({"index": 31, "state": {"left_ee": {"position": "bad"}}}) + "\n")
    code, p = get_json(start() + "/api/trials/%s/joints" % TID)
    assert code == 200 and p["rows"] == 32
    assert p["joints"][30]["left_ee"] is None and p["joints"][31]["left_ee"] is None and p["joints"][31]["right_ee"] is None


def story_fixture(tmp):
    """3 recorded trials (ok / ok with a retry + grasp:missed / failed with bin_moved), an agent run that drove the last two, and a
    report-only agent run (no runs/agent folder, as on a laptop that only pulled results/agent)."""
    trials, agent, reports = tmp / "trials", tmp / "agent", tmp / "reports"
    a = make_trial(trials, "20261121_130500", n=30, frames=False)
    b = make_trial(trials, "20261121_132000", lane="A", n=40, retry=True, flags=[(20, "grasp:missed")], frames=False)
    c = make_trial(trials, "20261121_134500", n=30, outcome="failed", flags=[(15, "bin_moved")], frames=False)
    make_agent_run(agent, reports, "20261121_131900", [b, c], {1: "missed grasp", 2: "bin moved"})
    (reports / "REPORT_20261121_124748.md").write_text(
        "# Trial agent report 20261121_124748 (m, 1 robot sessions)\n\n# Early run\n\ntext\n\n## Journal\n"
        '- {"trial": 1, "note": "raw image", "overlay": "none", "wording": {"PLACE": ["put it in the {mat} bin"]}, "verdict": "budget", '
        '"seconds_to_verdict": 64.2, "retries": 1}\n\nArtifacts: runs/agent/20261121_124748\n', encoding="utf-8")
    return trials, agent, reports, (a, b, c)


def test_story_merges_trials_and_agent_runs_in_time_order(served):
    tmp, start = served
    trials, agent, reports, _ = story_fixture(tmp)
    url = start(trials, agent_root=agent, reports_root=reports)
    code, items = get_json(url + "/api/story")
    assert code == 200
    assert [(i["kind"], i["id"]) for i in items] == [("agent", "20261121_124748"), ("trial", "20261121_130500"), ("agent", "20261121_131900"),
                                                     ("trial", "20261121_132000"), ("trial", "20261121_134500")]
    early, t1, run, t2, t3 = items
    assert t1["verdict"] == "ok" and t1["retries"] == 0 and t1["wording"][0] == "use the right arm to put the box into the red bin"
    assert [b["state"] for b in t1["blocks"]] == ["IDLE", "SCAN", "PLACE", "HOME", "DONE"] and t1["blocks"][2]["s"] > 0
    assert t1["overlay"] == "none" and t1["calls"] == 30 and t1["wallx_calls"] > 0 and t1["p95_latency_ms"] >= t1["mean_latency_ms"] > 0
    assert t2["retries"] == 1 and t2["flags"] == [{"i": 20, "flag": "grasp:missed"}] and t2["agent"] == "20261121_131900"
    assert t3["verdict"] == "failed" and t3["flags"] == [{"i": 15, "flag": "bin_moved"}] and t3["agent_trial"] == 2
    assert run["verdict"] == "1/2 ok" and [t["replay"] for t in run["trials"]] == ["20261121_132000", "20261121_134500"]
    assert run["title"] == "Demo run: 2 trials, 1 ok" and run["trials"][1]["flags"][0]["flag"] == "bin_moved"
    assert early["title"] == "Early run" and early["trials"][0]["verdict"] == "budget" and early["trials"][0]["replay"] is None


def test_agent_route_and_bad_ids(served):
    tmp, start = served
    trials, agent, reports, _ = story_fixture(tmp)
    url = start(trials, agent_root=agent, reports_root=reports)
    code, d = get_json(url + "/api/agent/20261121_131900")
    assert code == 200 and "## Journal" in d["report"] and len(d["journal"]) == 2 and len(d["progress"]) == 2 and d["wording"]["PLACE"]
    code, d = get_json(url + "/api/agent/20261121_124748")  # report only
    assert code == 200 and d["journal"][0]["verdict"] == "budget" and d["progress"] == []
    for path in ("/api/agent/20990101_000000", "/api/agent/..%2F..%2Fetc", "/api/agent/x", "/api/story/extra"):
        assert get_json(url + path)[0] == 404, path


def test_story_survives_broken_files(served):
    tmp, start = served
    trials, agent, reports, (a, b, c) = story_fixture(tmp)
    (a / "meta.json").write_text("{not json")
    (b / "summary.json").unlink()  # still recording
    (agent / "20261121_131900" / "journal.json").write_text("[broken")
    (agent / "20261121_131900" / "t1.jsonl").write_text('{"t": 1, "state": "SCAN"}\nnot json\n')
    code, items = get_json(start(trials, agent_root=agent, reports_root=reports) + "/api/story")
    assert code == 200 and len(items) == 5
    by = {i["id"]: i for i in items}
    assert by["20261121_132000"]["live"] and by["20261121_132000"]["verdict"] == "ok"  # outcome known, summary not written yet
    assert len(by["20261121_131900"]["trials"]) == 2  # journal falls back to the report's list


def test_frame_index_overlay_frames_and_rows_free_detail(served):
    root, start = served
    make_trial(root, TID, n=30, frames=True, frame_every=3)
    url = start()
    code, d = get_json(url + "/api/trials/" + TID)
    assert code == 200 and len(d["rows"]) == 30
    f = d["frames"]
    assert f["front"] == list(range(0, 30, 3)) and f["left_wrist"] == f["front"]
    wallx = [r["index"] for r in d["rows"] if r["source"] == "wallx"]
    assert f["front_overlay"] and set(f["front_overlay"]) <= set(wallx)  # the masked image is recorded only when WALLX is called
    code, light = get_json(url + "/api/trials/" + TID + "?rows=0")
    assert code == 200 and "rows" not in light and light["frames"] == f and light["summary"]["calls"] == 30
    k = f["front_overlay"][0]
    with urllib.request.urlopen(url + "/api/trials/%s/frame/%d/front_overlay" % (TID, k + 1), timeout=10) as r:  # nearest earlier
        assert r.status == 200 and r.headers["Content-Type"] == "image/jpeg" and r.read()[:2] == b"\xff\xd8"
