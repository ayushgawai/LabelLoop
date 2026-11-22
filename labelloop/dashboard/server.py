"""Local trial dashboard server (stdlib only): python -m labelloop.dashboard.server [--root runs/trials] [--agent runs/agent]
[--reports results/agent] [--port 8095].

Reads the folders written by labelloop.robots.trial_recorder and serves the one-file page (index.html) plus a tiny JSON API:
  /                               the dashboard page
  /api/trials                     list (newest first)
  /api/trials/<id>[?rows=0]       {meta, rows, summary, frames: {camera: [row indices]}} (rows=0: without the rows)
  /api/trials/<id>/rows?after=N   rows with index > N (live tail)
  /api/trials/<id>/joints         per-row joint chains of both arms via IK (labelloop.dashboard.kinematics; cached in joints.json)
  /api/trials/<id>/frame/<i>/<c>  JPEG (nearest earlier recorded index if missing)
  POST /api/run {demo}              run that demo's frozen recipe on the REAL robot (GET /api/run: status)
  /api/demos                      one-click demos (demos.json): trial id + step-by-step story
  /api/story                      every trial + trial-agent run in time order (labelloop.dashboard.story)
  /api/agent/<ts>                 one agent run: journal, report text, progress notes

Phone QR scanner (labelloop.dashboard.package; display only, never steers the robot):
  /scan                           the phone's scanner page (camera needs HTTPS on iOS: run with --https-port)
  /api/package                    {current, history, scans, ver, phone}
  POST /api/package/frame         body = one JPEG from the phone; decoded here with labelloop.perception.qr -> {codes, package, new}
  POST /api/package/scan          {"code": "420/94043/001"} (JSON or form): put a code on the board by hand / from an iOS Shortcut
  POST /api/package/clear         empty the board (between demo runs)

  python -m labelloop.dashboard.server --https-port 8443   ->   phone opens https://<this laptop's Wi-Fi IP>:8443/scan
"""
import argparse
import json
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from labelloop.dashboard.package import PackageBoard, decode_jpeg

TRIAL_ID_RE = re.compile(r"^\d{8}_\d{6}$")
CAMERA_RE = re.compile(r"^[a-z_]+$")
INDEX_HTML = Path(__file__).resolve().parent / "index.html"
DEMOS = Path(__file__).resolve().parent / "demos.json"  # one-click demos: trial id + the step-by-step story
REPO = Path(__file__).resolve().parents[2]
RUN = {"proc": None, "demo": None, "log": None, "started": None}  # the one robot run started from the page (only one at a time)
RUN_LOCK = threading.Lock()


def run_status():
    p = RUN["proc"]
    return {"running": bool(p and p.poll() is None), "demo": RUN["demo"], "started": RUN["started"],
            "exit": None if not p or p.poll() is None else p.returncode, "log": RUN["log"]}


def start_run(demo_id):
    """POST /api/run {"demo": id}: runs that demo's FROZEN recipe on the real robot (scripts/run_trial.py: homes, runs, homes, records).
    Only recipes listed in demos.json ("run") can be started, never a path from the request."""
    d = next((x for x in (_read_json(DEMOS) or []) if x.get("id") == demo_id and x.get("run")), None)
    if d is None:
        return 404, {"error": "no runnable demo " + str(demo_id)}
    with RUN_LOCK:
        if run_status()["running"]:
            return 409, {"error": "a run is already going", **run_status()}
        log = REPO / "runs" / "dashboard_runs" / (time.strftime("%Y%m%d_%H%M%S") + "_" + demo_id + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        py = REPO / ".venv/bin/python"
        RUN.update(proc=subprocess.Popen([str(py if py.exists() else sys.executable), "scripts/run_trial.py", "--scenario", d["run"]], cwd=REPO,
                                         stdout=open(log, "w"), stderr=subprocess.STDOUT),
                   demo=demo_id, log=str(log.relative_to(REPO)), started=time.time())
        return 200, run_status()

SCAN_HTML = Path(__file__).resolve().parent / "scan.html"
MAX_BODY = 8 * 1024 * 1024  # one phone frame (a 1920 px JPEG is ~0.5 MB)


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def read_rows(trial_dir):
    """trial.jsonl -> [dict]; a half-written / corrupt line is skipped, never raised."""
    rows = []
    try:
        with open(Path(trial_dir) / "trial.jsonl", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        pass
    return rows


def row_index(row, default=-1):
    v = row.get("index", default) if isinstance(row, dict) else default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def list_trials(root):
    """Every <root>/<YYYYmmdd_HHMMSS>/ folder, newest first."""
    root, out = Path(root), []
    if not root.is_dir():
        return out
    for d in sorted(root.iterdir(), reverse=True):
        if not (d.is_dir() and TRIAL_ID_RE.match(d.name)):
            continue
        meta, summary = _read_json(d / "meta.json") or {}, _read_json(d / "summary.json")
        need = summary is None or any(summary.get(k) is None for k in ("calls", "parcels", "duration_s"))
        rows = read_rows(d) if need else []  # a finished trial's summary has it all: no re-reading long trials on every poll
        parcels = (summary or {}).get("parcels")
        if parcels is None and rows:
            h = rows[-1].get("harness")
            parcels = h.get("parcels") if isinstance(h, dict) else None
        duration = (summary or {}).get("duration_s")
        if duration is None and rows:
            end = float(rows[-1].get("t") or 0)
            start = float(meta.get("start_epoch") or (rows[0].get("t") or end))
            duration = round(max(0.0, end - start), 1)
        out.append({"id": d.name, "goal": meta.get("goal"), "start": meta.get("start"),
                    "calls": (summary or {}).get("calls", len(rows)), "duration": duration,
                    "parcels": parcels, "live": summary is None})
    out.sort(key=lambda r: r["id"], reverse=True)
    return out


def _indexed_frames(trial_dir, camera):
    pairs = []
    for p in (Path(trial_dir) / "frames").glob("*_%s.jpg" % camera):
        head = p.name.split("_", 1)[0]
        try:
            pairs.append((int(head), p))
        except ValueError:
            continue
    return sorted(pairs)


def frame_index(trial_dir):
    """{camera: sorted row indices that have a frame} of a trial (frames/<index>_<camera>.jpg; front_overlay = what WALLX saw)."""
    out = {}
    for p in (Path(trial_dir) / "frames").glob("*.jpg"):
        head, _, cam = p.stem.partition("_")
        if head.isdigit() and CAMERA_RE.match(cam):
            out.setdefault(cam, []).append(int(head))
    return {c: sorted(v) for c, v in sorted(out.items())}


def frame_path(trial_dir, index, camera):
    """The frame of `camera` recorded at the nearest earlier index (else the earliest one); None if the camera has none."""
    pairs = _indexed_frames(trial_dir, camera)
    earlier = [p for n, p in pairs if n <= index]
    if earlier:
        return earlier[-1]
    return pairs[0][1] if pairs else None


class Handler(BaseHTTPRequestHandler):
    root = "runs/trials"
    agent_root = "runs/agent"
    reports_root = "results/agent"
    board = None  # PackageBoard (make_server binds one; shared by the HTTP and HTTPS listeners)
    phone_only = False  # the HTTPS listener (every interface, shared event Wi-Fi): only /scan and /api/package*, no trials / reports
    server_version = "LabelLoopDashboard/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet: the relay owns stdout/stderr
        pass

    # ---- plumbing ----
    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, default=str).encode("utf-8"), "application/json")

    def _file(self, path, ctype):
        try:
            body = Path(path).read_bytes()
        except OSError:
            return self._json({"error": "not found"}, 404)
        self._send(200, body, ctype)

    def _post_run(self):  # POST /api/run (robot button): local dashboard only, never on the phone (HTTPS) server
        try:
            n = min(int(self.headers.get("Content-Length") or 0), 4096)
            body = json.loads(self.rfile.read(n) or b"{}")
            code, out = start_run(str(body.get("demo", "")))
            return self._json(out, code)
        except Exception as e:
            return self._json({"error": type(e).__name__}, 500)

    def do_GET(self):
        url = urlparse(self.path)
        parts = [unquote(p) for p in url.path.split("/") if p]
        try:
            if self.phone_only and not (parts in (["scan"], ["scan.html"]) or parts[:2] == ["api", "package"]):
                return self._json({"error": "not found"}, 404)
            if not parts or parts[0] == "index.html":
                return self._file(INDEX_HTML, "text/html; charset=utf-8")
            if parts == ["scan"] or parts == ["scan.html"]:
                return self._file(SCAN_HTML, "text/html; charset=utf-8")
            if parts == ["api", "package"]:
                return self._json(self._board().state())
            if parts[0] == "api" and len(parts) >= 2 and parts[1] == "trials":
                return self._trials(parts[2:], parse_qs(url.query))
            if parts[0] == "api" and parts[1:] == ["run"]:
                return self._json(run_status())
            if parts[0] == "api" and parts[1:] == ["demos"]:
                return self._json(_read_json(DEMOS) or [])
            if parts[0] == "api" and parts[1:] == ["story"]:
                from labelloop.dashboard import story
                return self._json(story.build(self.root, self.agent_root, self.reports_root))
            if parts[0] == "api" and len(parts) == 3 and parts[1] == "agent" and TRIAL_ID_RE.match(parts[2]):
                from labelloop.dashboard import story
                d = story.agent_detail(parts[2], self.agent_root, self.reports_root)
                return self._json(d) if d else self._json({"error": "no such agent run"}, 404)
            self._json({"error": "not found"}, 404)
        except Exception as e:  # a bad request never takes the dashboard down
            self._json({"error": type(e).__name__}, 500)

    def do_POST(self):
        parts = [unquote(p) for p in urlparse(self.path).path.split("/") if p]
        if parts == ["api", "run"] and not self.phone_only:
            return self._post_run()
        try:
            if parts[:2] != ["api", "package"] or len(parts) != 3 or parts[2] not in ("frame", "scan", "clear"):
                self.close_connection = True  # the body was never read: don't reuse this connection
                return self._json({"error": "not found"}, 404)
            body = self._body()
            if body is None:
                return
            board, client = self._board(), self.client_address[0]
            if parts[2] == "clear":
                board.clear()
                return self._json(board.state())
            if parts[2] == "frame":
                codes = decode_jpeg(body)
                board.frame(client, codes)
                pkg, new = (board.scan(codes[0], "phone camera", client) if codes else (None, False))
                return self._json({"codes": codes, "package": pkg, "new": new})
            code = self._code(body)
            if not code:
                return self._json({"error": "no code"}, 400)
            pkg, new = board.scan(code, "typed", client)
            return self._json({"codes": [code], "package": pkg, "new": new})
        except Exception as e:
            self._json({"error": type(e).__name__}, 500)

    # ---- package board ----
    def _board(self):
        if self.board is None:  # a bare Handler (no make_server): one in-memory board for the class
            type(self).board = PackageBoard()
        return self.board

    def _body(self):
        """The request body, or None after answering 411 / 413."""
        try:
            n = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self.close_connection = True
            self._json({"error": "Content-Length required"}, 411)
            return None
        if n < 0 or n > MAX_BODY:
            self.close_connection = True
            self._json({"error": "body too large"}, 413)
            return None
        return self.rfile.read(n) if n else b""

    def _code(self, body):
        """The code of a /scan body: JSON {"code": ...}, form code=..., or the bare text."""
        text = body.decode("utf-8", "replace").strip()
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype == "application/json" or text.startswith("{"):
            try:
                d = json.loads(text)
            except ValueError:
                return None
            return str(d.get("code") or "").strip() if isinstance(d, dict) else None
        if ctype == "application/x-www-form-urlencoded":
            return (parse_qs(text).get("code") or [""])[0].strip()
        return text

    # ---- routes ----
    def _trials(self, rest, query):
        if not rest:
            return self._json(list_trials(self.root))
        tid = rest[0]
        if not TRIAL_ID_RE.match(tid):  # also rejects every traversal attempt (".", "..", anything with a slash)
            return self._json({"error": "bad trial id"}, 404)
        d = Path(self.root) / tid
        if not d.is_dir():
            return self._json({"error": "no such trial"}, 404)
        if len(rest) == 1:  # ?rows=0: meta + summary + frame index only (the page's periodic refresh of a long trial)
            out = {"meta": _read_json(d / "meta.json") or {}, "summary": _read_json(d / "summary.json"), "frames": frame_index(d)}
            if query.get("rows", ["1"])[0] != "0":
                out["rows"] = read_rows(d)
            return self._json(out)
        if len(rest) == 2 and rest[1] == "rows":
            try:
                after = int(query.get("after", ["-1"])[0])
            except (TypeError, ValueError):
                after = -1
            return self._json([r for r in read_rows(d) if row_index(r) > after])
        if len(rest) == 2 and rest[1] == "joints":
            from labelloop.dashboard import kinematics  # (late: numpy + the arm model only when a page asks for joints)
            return self._json(kinematics.trial_joints(d, read_rows(d)))
        if len(rest) == 4 and rest[1] == "frame":
            camera = rest[3]
            if not CAMERA_RE.match(camera):
                return self._json({"error": "bad camera"}, 404)
            try:
                index = int(rest[2])
            except ValueError:
                return self._json({"error": "bad index"}, 404)
            p = frame_path(d, index, camera)
            if p is None:
                return self._json({"error": "no frame"}, 404)
            if camera == "front" and query.get("track", ["0"])[0] == "1":  # the tracked parcel boxed on the head frame
                from labelloop.dashboard import track
                try:
                    n = int(Path(p).name.split("_")[0])
                    rows = read_rows(d)  # grid trials (a harness 'locate' step logged harness.where): grid + the located cell
                    wh = [(row_index(r), (r.get("harness") or {}).get("where")) for r in rows if (r.get("harness") or {}).get("where")]
                    grid = bool(wh) or any("where" in (r.get("harness") or {}) for r in rows[:3])
                    cell = next((w for i, w in reversed(wh) if i <= n), None)
                    return self._send(200, track.draw(p, track.track(d).get(n), grid, cell), "image/jpeg")
                except Exception:
                    pass
            return self._file(p, "image/jpeg")
        self._json({"error": "not found"}, 404)


def make_server(root="runs/trials", host="127.0.0.1", port=8095, agent_root=None, reports_root=None, board=None):
    """ThreadingHTTPServer bound to (host, port) serving `root` (port 0 = any free port). Agent runs default to <root>/../agent, their
    reports to <root>/../../results/agent (runs/trials -> runs/agent, results/agent). `board` (PackageBoard) defaults to one saved in
    <root>/../package/board.json."""
    root = Path(root)
    agent_root = Path(agent_root) if agent_root else root.parent / "agent"
    reports_root = Path(reports_root) if reports_root else root.parent.parent / "results" / "agent"
    board = board if board is not None else PackageBoard(root.parent / "package" / "board.json")
    handler = type("BoundHandler", (Handler,), {"root": str(root), "agent_root": str(agent_root), "reports_root": str(reports_root),
                                                "board": board})
    return ThreadingHTTPServer((host, port), handler)


def make_tls_server(http_server, host="0.0.0.0", port=8443, cert=None, key=None, cert_dir=None):
    """A second listener over HTTPS for the phone: the scanner page + package API only, on the same package board as `http_server`.
    Without cert/key a self-signed pair is made once in `cert_dir` (default <root>/../package/tls)."""
    from labelloop.dashboard import tls
    handler = type("PhoneHandler", (http_server.RequestHandlerClass,), {"phone_only": True})
    if not (cert and key):
        cert, key = tls.ensure_cert(cert_dir or Path(handler.root).parent / "package" / "tls")
    return tls.TLSServer((host, port), handler, cert, key)


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m labelloop.dashboard.server", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="runs/trials", help="trial recordings (one folder per trial)")
    p.add_argument("--agent", default=None, help="trial-agent runs (default <root>/../agent)")
    p.add_argument("--reports", default=None, help="trial-agent reports REPORT_<ts>.md (default <root>/../../results/agent)")
    p.add_argument("--port", type=int, default=8095)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--https-port", type=int, default=None, help="also serve over HTTPS on this port (the phone scanner: try 8443)")
    p.add_argument("--https-host", default="0.0.0.0", help="HTTPS bind address (default: every interface, so the phone can reach it)")
    p.add_argument("--cert", default=None, help="TLS certificate (PEM); default: a self-signed one made with openssl")
    p.add_argument("--key", default=None, help="TLS private key (PEM)")
    a = p.parse_args(argv)
    server = make_server(a.root, a.host, a.port, a.agent, a.reports)
    print(f"Dashboard at http://{a.host}:{server.server_address[1]} (trials in {a.root})", flush=True)
    servers = [server]
    if a.https_port is not None:
        from labelloop.dashboard.tls import lan_ips
        tls_server = make_tls_server(server, a.https_host, a.https_port, a.cert, a.key)
        servers.append(tls_server)
        port = tls_server.server_address[1]
        for ip in (lan_ips() if a.https_host in ("0.0.0.0", "") else [a.https_host]) or ["<this laptop's IP>"]:
            print(f"Phone scanner at https://{ip}:{port}/scan", flush=True)
        threading.Thread(target=tls_server.serve_forever, name="https", daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for s in servers[1:]:  # the HTTPS thread; the HTTP loop has already returned
            s.shutdown()
        for s in servers:
            s.server_close()


if __name__ == "__main__":
    main()
