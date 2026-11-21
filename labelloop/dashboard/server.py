"""Local trial dashboard server (stdlib only): python -m labelloop.dashboard.server [--root runs/trials] [--port 8095].

Reads the folders written by labelloop.robots.trial_recorder and serves the one-file page (index.html) plus a tiny JSON API:
  /                               the dashboard page
  /api/trials                     list (newest first)
  /api/trials/<id>                {meta, rows, summary}
  /api/trials/<id>/rows?after=N   rows with index > N (live tail)
  /api/trials/<id>/frame/<i>/<c>  JPEG (nearest earlier recorded index if missing)
"""
import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

TRIAL_ID_RE = re.compile(r"^\d{8}_\d{6}$")
CAMERA_RE = re.compile(r"^[a-z_]+$")
INDEX_HTML = Path(__file__).resolve().parent / "index.html"


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
        rows = read_rows(d)
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


def frame_path(trial_dir, index, camera):
    """The frame of `camera` recorded at the nearest earlier index (else the earliest one); None if the camera has none."""
    pairs = _indexed_frames(trial_dir, camera)
    earlier = [p for n, p in pairs if n <= index]
    if earlier:
        return earlier[-1]
    return pairs[0][1] if pairs else None


class Handler(BaseHTTPRequestHandler):
    root = "runs/trials"
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

    def do_GET(self):
        url = urlparse(self.path)
        parts = [unquote(p) for p in url.path.split("/") if p]
        try:
            if not parts or parts[0] == "index.html":
                return self._file(INDEX_HTML, "text/html; charset=utf-8")
            if parts[0] == "api" and len(parts) >= 2 and parts[1] == "trials":
                return self._trials(parts[2:], parse_qs(url.query))
            self._json({"error": "not found"}, 404)
        except Exception as e:  # a bad request never takes the dashboard down
            self._json({"error": type(e).__name__}, 500)

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
        if len(rest) == 1:
            return self._json({"meta": _read_json(d / "meta.json") or {}, "rows": read_rows(d),
                               "summary": _read_json(d / "summary.json")})
        if len(rest) == 2 and rest[1] == "rows":
            try:
                after = int(query.get("after", ["-1"])[0])
            except (TypeError, ValueError):
                after = -1
            return self._json([r for r in read_rows(d) if row_index(r) > after])
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
            return self._file(p, "image/jpeg")
        self._json({"error": "not found"}, 404)


def make_server(root="runs/trials", host="127.0.0.1", port=8095):
    """ThreadingHTTPServer bound to (host, port) serving `root` (port 0 = any free port)."""
    handler = type("BoundHandler", (Handler,), {"root": str(root)})
    return ThreadingHTTPServer((host, port), handler)


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m labelloop.dashboard.server", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="runs/trials", help="trial recordings (one folder per trial)")
    p.add_argument("--port", type=int, default=8095)
    p.add_argument("--host", default="127.0.0.1")
    a = p.parse_args(argv)
    server = make_server(a.root, a.host, a.port)
    print(f"Dashboard at http://{a.host}:{server.server_address[1]} (trials in {a.root})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
