import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from labelloop import look
from labelloop.dashboard import kinematics as K
from labelloop.dashboard.demo import make_trial, state_of
from labelloop.dashboard.server import make_server

TID = "20261121_130500"
QS = [[0.3, 1.0, -1.2, 0.2, 0.4, 0.1], [-0.4, 0.8, -0.9, -0.3, 0.5, -0.2], [0.1, 1.5, -1.8, 0.4, -0.6, 0.3], [0.6, 1.2, -0.7, 0.0, 0.9, -0.5]]


@pytest.fixture
def served(tmp_path):
    def start():
        httpd = make_server(tmp_path, port=0)
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
