import json
import threading
import urllib.error
import urllib.request

import pytest

from labelloop.dashboard.server import make_server

STATE = {"left_ee": {"position": [0.0, 0.0, 0.3], "rotation_rpy": [0.0, 0.0, 0.0], "gripper": 0.1},
         "right_ee": {"position": [0.0, 0.0, 0.3], "rotation_rpy": [0.0, 0.0, 0.0], "gripper": 0.1}}
JPEG = b"\xff\xd8\xff\xe0dashboard\xff\xd9"
TID = "20261121_115000"


def write_trial(root, n=3):
    d = root / TID
    (d / "frames").mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"trial_id": TID, "goal": "sort the boxes",
                                             "start": "2026-11-21T11:50:00", "start_epoch": 1.0}), encoding="utf-8")
    with open(d / "trial.jsonl", "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"t": 1.0 + i, "index": i, "goal": "sort the boxes", "source": "wallx",
                                "state": STATE, "harness": {"state": "SCAN", "codes": []}}) + "\n")
    (d / "frames" / "0_front.jpg").write_bytes(JPEG)
    (d / "frames" / "2_front.jpg").write_bytes(JPEG + b"2")
    (d / "summary.json").write_text(json.dumps({"trial_id": TID, "calls": n, "duration_s": 12.0, "parcels": 1}),
                                    encoding="utf-8")
    return d


@pytest.fixture
def server(tmp_path):
    write_trial(tmp_path)
    httpd = make_server(tmp_path, port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield "http://127.0.0.1:%d" % httpd.server_address[1]
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def get(url):
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_index_page(server):
    code, body = get(server + "/")
    assert code == 200 and b"LabelLoop" in body


def test_trial_list(server):
    code, body = get(server + "/api/trials")
    assert code == 200
    trials = json.loads(body)
    assert len(trials) == 1
    t = trials[0]
    assert t["id"] == TID and t["goal"] == "sort the boxes"
    assert t["calls"] == 3 and t["parcels"] == 1


def test_trial_detail_and_live_tail(server):
    code, body = get(server + "/api/trials/" + TID)
    assert code == 200
    d = json.loads(body)
    assert d["meta"]["goal"] == "sort the boxes"
    assert len(d["rows"]) == 3
    assert d["summary"]["parcels"] == 1

    code, body = get(server + "/api/trials/" + TID + "/rows?after=0")
    assert code == 200
    assert [r["index"] for r in json.loads(body)] == [1, 2]


def test_frame_nearest_earlier(server):
    code, body = get(server + "/api/trials/%s/frame/1/front" % TID)  # no frame 1: nearest earlier is 0
    assert code == 200 and body == JPEG
    code, body = get(server + "/api/trials/%s/frame/9/front" % TID)
    assert code == 200 and body == JPEG + b"2"
    code, _ = get(server + "/api/trials/%s/frame/0/right_wrist" % TID)  # never recorded
    assert code == 404


def test_traversal_and_bad_ids_rejected(server):
    for path in ("/api/trials/../../etc/passwd",
                 "/api/trials/" + TID + "/frame/0/..%2F..%2Fpasswd",
                 "/api/trials/not_a_trial",
                 "/api/trials/" + TID + "/rows/extra",
                 "/api/trials/20260101_000000"):
        code, _ = get(server + path)
        assert code == 404, path
