"""Package board: what the phone scanner last read, for the dashboard's package card (display only: it never steers the robot).

The phone (labelloop/dashboard/scan.html, served at /scan) sends camera frames; the dashboard server decodes them with the harness's
own QR reader (labelloop.perception.qr.locate) and calls PackageBoard.scan(code). Re-reading the package already on the board only
bumps its `seen` count; a different code becomes the new current package and the old one moves to the history.
"""
import json
import re
import threading
import time
from pathlib import Path

CODE_RE = re.compile(r"^420/(\d{5})/(\d{3})$")
ZONES = {"blue": "1", "red": "2", "black": "3"}  # = the "zones" map of every scenario steps file (finaldemo1, sc4_wallx, ...)
WHERE = {"1": "back left", "2": "back right", "3": "right edge"}
MAX_CODE = 200
HISTORY = 20
FRAME_LIVE_S = 5.0  # the phone counts as connected while its frames arrive at least this often


def parse(code):
    """A label payload -> {code, valid, zip5, zip3, seq, lane, zone, where}; anything but 420/<ZIP5>/<seq> is the black lane."""
    from labelloop.sequence import bin_for  # (late: one routing rule, the relay's own)
    code = str(code or "").strip()[:MAX_CODE]
    m = CODE_RE.match(code)
    lane = bin_for(code)
    zone = ZONES[lane]
    return {"code": code, "valid": bool(m), "zip5": m.group(1) if m else None, "zip3": m.group(1)[:3] if m else None,
            "seq": m.group(2) if m else None, "lane": lane, "zone": zone, "where": WHERE[zone]}


def decode_jpeg(data):
    """QR / barcode texts in one JPEG/PNG (bytes), nearest the image centre first; [] if it is not an image or has no readable code."""
    import cv2
    import numpy as np
    from labelloop.perception.qr import locate
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return []
    from labelloop.sequence import any_barcodes, norm_code  # also 1D barcodes and bare pincodes (90291 -> 420/90291/000)
    rgb, dist = cv2.cvtColor(img, cv2.COLOR_BGR2RGB), {}
    h, w = rgb.shape[:2]
    for text, (cx, cy) in locate(rgb) + any_barcodes(rgb):
        text = norm_code(text) or text
        if text:
            d = (cx - w / 2) ** 2 + (cy - h / 2) ** 2
            dist[text] = min(d, dist.get(text, d))
    return sorted(dist, key=dist.get)  # nearest the frame centre first: several labels in view -> the one the phone points at


class PackageBoard:
    """Thread-safe current package + history, saved to `path` (JSON) on every change so a restart keeps the board."""

    def __init__(self, path=None, clock=time.time):
        self.path, self.clock, self.lock = Path(path) if path else None, clock, threading.Lock()
        self.current, self.history, self.scans, self.ver, self.phone = None, [], 0, 0, None
        self._load()

    def _load(self):
        if not (self.path and self.path.is_file()):
            return
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if isinstance(d, dict):
            self.current = d.get("current") if isinstance(d.get("current"), dict) else None
            self.history = [h for h in d.get("history") or [] if isinstance(h, dict)][:HISTORY]
            self.scans = int(d.get("scans") or 0)
            self.ver = int(d.get("ver") or 0)

    def _save(self):
        if not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"current": self.current, "history": self.history, "scans": self.scans, "ver": self.ver}),
                           encoding="utf-8")
            tmp.replace(self.path)
        except OSError:
            pass  # display state only: a full disk never breaks a scan

    def scan(self, code, source="phone", client=None):
        """Put `code` on the board. Returns (package, is_new)."""
        p = parse(code)
        if not p["code"]:
            raise ValueError("empty code")
        now = self.clock()
        with self.lock:
            cur = self.current
            if cur and cur.get("code") == p["code"]:
                cur["seen"] = int(cur.get("seen") or 1) + 1
                cur["last_seen"] = now
                self.ver += 1
                self._save()
                return dict(cur), False
            if cur:
                self.history.insert(0, cur)
                del self.history[HISTORY:]
            p.update({"scanned": now, "last_seen": now, "seen": 1, "source": source, "client": client})
            self.current, self.scans, self.ver = p, self.scans + 1, self.ver + 1
            self._save()
            return dict(p), True

    def frame(self, client=None, codes=()):
        """Note a frame from the phone (for the 'phone connected' line); not saved: liveness is only meaningful while running."""
        with self.lock:
            prev = self.phone or {}
            self.phone = {"last_frame": self.clock(), "client": client, "frames": int(prev.get("frames") or 0) + 1,
                          "last_codes": list(codes)}

    def clear(self):
        with self.lock:
            self.current, self.history, self.scans, self.ver = None, [], 0, self.ver + 1
            self._save()

    def state(self):
        now = self.clock()
        with self.lock:
            phone = dict(self.phone) if self.phone else None
            if phone:
                phone["age_s"] = round(max(0.0, now - phone["last_frame"]), 1)
                phone["live"] = phone["age_s"] <= FRAME_LIVE_S
            return {"current": dict(self.current) if self.current else None, "history": [dict(h) for h in self.history],
                    "scans": self.scans, "ver": self.ver, "phone": phone, "now": now}
