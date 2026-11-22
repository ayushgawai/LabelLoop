"""Phone QR scanner: package board, the /api/package routes of the dashboard server, the HTTPS listener."""
import json
import shutil
import ssl
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from labelloop.dashboard import package
from labelloop.dashboard.package import PackageBoard, parse
from labelloop.dashboard.server import make_server, make_tls_server

REAL_HEAD = Path(__file__).parent / "data" / "real_head_qr.jpg"  # the real head camera: label 420/95126/001 (~60 px)


@pytest.mark.parametrize("code,lane,zone,zip5", [
    ("420/94043/001", "blue", "1", "94043"),
    ("420/95126/001", "blue", "1", "95126"),
    ("420/90024/002", "red", "2", "90024"),
    ("420/10001/001", "black", "3", "10001"),
    ("hello", "black", "3", None),
])
def test_parse_routes_like_the_relay(code, lane, zone, zip5):
    p = parse(code)
    assert (p["lane"], p["zone"], p["zip5"], p["valid"]) == (lane, zone, zip5, zip5 is not None)
    assert p["where"] == package.WHERE[zone]


def test_board_dedups_keeps_history_and_persists(tmp_path):
    t = [100.0]
    path = tmp_path / "package" / "board.json"
    b = PackageBoard(path, clock=lambda: t[0])
    p, new = b.scan("420/94043/001", "phone camera", "10.0.0.5")
    assert new and p["seen"] == 1 and p["client"] == "10.0.0.5"
    t[0] = 101.0
    p, new = b.scan("420/94043/001")
    assert not new and p["seen"] == 2 and p["last_seen"] == 101.0 and p["scanned"] == 100.0
    p, new = b.scan(" 420/90024/002 ")
    assert new and p["code"] == "420/90024/002" and p["lane"] == "red"
    s = b.state()
    assert s["scans"] == 2 and s["current"]["code"] == "420/90024/002" and [h["code"] for h in s["history"]] == ["420/94043/001"]
    again = PackageBoard(path).state()  # a restarted dashboard keeps the board
    assert again["current"]["code"] == "420/90024/002" and again["scans"] == 2 and again["phone"] is None
    b.clear()
    assert b.state()["current"] is None and b.state()["history"] == [] and PackageBoard(path).state()["current"] is None
    with pytest.raises(ValueError):
        b.scan("   ")


def test_board_phone_liveness():
    t = [0.0]
    b = PackageBoard(clock=lambda: t[0])
    b.frame("10.0.0.5", [])
    assert b.state()["phone"]["live"] and b.state()["phone"]["frames"] == 1
    t[0] = package.FRAME_LIVE_S + 1
    assert not b.state()["phone"]["live"]


def test_decode_jpeg_real_head_frame_and_junk():
    assert package.decode_jpeg(REAL_HEAD.read_bytes()) == ["420/95126/001"]
    assert package.decode_jpeg(b"not an image") == []


@pytest.fixture
def server(tmp_path):
    board = PackageBoard(tmp_path / "board.json")
    httpd = make_server(tmp_path / "trials", port=0, board=board)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield "http://127.0.0.1:%d" % httpd.server_address[1], httpd
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def call(url, body=None, ctype="application/json", method=None, context=None):
    req = urllib.request.Request(url, data=body, method=method or ("POST" if body is not None else "GET"))
    if body is not None:
        req.add_header("Content-Type", ctype)
    try:
        with urllib.request.urlopen(req, timeout=20, context=context) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_scan_page_and_empty_board(server):
    base, _ = server
    status, body = call(base + "/scan")
    assert status == 200 and b"getUserMedia" in body and b"/api/package/frame" in body
    status, body = call(base + "/api/package")
    s = json.loads(body)
    assert status == 200 and s["current"] is None and s["scans"] == 0 and s["phone"] is None
    status, body = call(base + "/")
    assert status == 200 and b'id="pkgCard"' in body and b"/api/package" in body


def test_typed_scan_json_form_and_text(server):
    base, _ = server
    status, body = call(base + "/api/package/scan", json.dumps({"code": "420/94043/001"}).encode())
    r = json.loads(body)
    assert status == 200 and r["new"] and r["package"]["lane"] == "blue" and r["package"]["source"] == "typed"
    status, body = call(base + "/api/package/scan", b"code=420%2F90024%2F002", "application/x-www-form-urlencoded")
    assert status == 200 and json.loads(body)["package"]["lane"] == "red"
    status, body = call(base + "/api/package/scan", b"420/90024/002", "text/plain")
    r = json.loads(body)
    assert status == 200 and not r["new"] and r["package"]["seen"] == 2
    s = json.loads(call(base + "/api/package")[1])
    assert s["current"]["code"] == "420/90024/002" and [h["code"] for h in s["history"]] == ["420/94043/001"]
    assert call(base + "/api/package/scan", json.dumps({"code": ""}).encode())[0] == 400
    assert call(base + "/api/package/scan", b"{not json")[0] == 400
    status, body = call(base + "/api/package/clear", b"")
    assert status == 200 and json.loads(body)["current"] is None


def test_frame_route_decodes_the_phone_jpeg(server):
    base, _ = server
    status, body = call(base + "/api/package/frame", REAL_HEAD.read_bytes(), "image/jpeg")
    r = json.loads(body)
    assert status == 200 and r["codes"] == ["420/95126/001"] and r["new"] and r["package"]["zone"] == "1"
    status, body = call(base + "/api/package/frame", b"\xff\xd8junk\xff\xd9", "image/jpeg")
    r = json.loads(body)
    assert status == 200 and r["codes"] == [] and r["package"] is None
    s = json.loads(call(base + "/api/package")[1])
    assert s["current"]["code"] == "420/95126/001" and s["phone"]["frames"] == 2 and s["phone"]["live"]
    assert s["phone"]["client"] == "127.0.0.1"


def test_post_errors(server, monkeypatch):
    base, _ = server
    assert call(base + "/api/package/nope", b"x")[0] == 404
    assert call(base + "/api/trials", b"x")[0] == 404
    from labelloop.dashboard import server as srv
    monkeypatch.setattr(srv, "MAX_BODY", 10)
    assert call(base + "/api/package/scan", b"420/94043/001")[0] == 413


@pytest.mark.skipif(not shutil.which("openssl"), reason="needs the openssl CLI for a self-signed certificate")
def test_https_listener_shares_the_board(server, tmp_path):
    base, httpd = server
    tls_server = make_tls_server(httpd, "127.0.0.1", 0, cert_dir=tmp_path / "tls")
    thread = threading.Thread(target=tls_server.serve_forever, daemon=True)
    thread.start()
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname, ctx.verify_mode = False, ssl.CERT_NONE  # self-signed, like the phone's "visit this website"
        tbase = "https://127.0.0.1:%d" % tls_server.server_address[1]
        status, body = call(tbase + "/scan", context=ctx)
        assert status == 200 and b"getUserMedia" in body
        status, body = call(tbase + "/api/package/scan", json.dumps({"code": "420/94043/001"}).encode(), context=ctx)
        assert status == 200 and json.loads(body)["new"]
        assert json.loads(call(base + "/api/package")[1])["current"]["code"] == "420/94043/001"  # the laptop's page sees it
        assert (tmp_path / "tls" / "cert.pem").is_file() and (tmp_path / "tls" / "key.pem").is_file()
        for path in ("/", "/index.html", "/api/trials", "/api/story"):  # the LAN-facing listener serves the scanner only
            assert call(tbase + path, context=ctx)[0] == 404, path
        assert call(base + "/")[0] == 200
        with pytest.raises(urllib.error.URLError):  # a client that refuses the certificate gets nothing (and the server lives on)
            call(tbase + "/api/package", context=ssl.create_default_context())
        assert call(tbase + "/api/package", context=ctx)[0] == 200
    finally:
        tls_server.shutdown()
        tls_server.server_close()
        thread.join(timeout=5)
