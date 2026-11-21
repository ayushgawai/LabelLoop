"""Trial recorder: one folder per real-robot trial with everything needed to replay it (JSONL rows + JPEG frames + summary)."""
import base64
import json
import math
import threading
import time
from pathlib import Path

MAX_ACTION_STEPS = 8


def _plain(value):
    """json.dumps default: numpy arrays/scalars and sets become plain JSON; never raises."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=repr)
    if isinstance(value, bytes):
        return None
    return str(value)


def _even_indices(n, k=MAX_ACTION_STEPS):
    """At most k indices into range(n), evenly spaced (first and last kept when n > k)."""
    k = min(int(n), int(k))
    if k <= 1:
        return [0] if n else []
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})


def reduce_actions(reply, max_steps=MAX_ACTION_STEPS):
    """A WALLX reply's action chunk -> at most `max_steps` evenly spaced steps per arm as plain lists; None if the format is
    unexpected (the relay must never fail on a reply it does not understand)."""
    try:
        acts = reply["actions"]
        out = {}
        for arm, a in acts.items():
            pos, rpy, grip = a["position"], a["rotation_rpy"], a["gripper"]
            n = len(pos)
            if not (n and len(rpy) == n and len(grip) == n):
                raise ValueError("action arrays differ in length")
            idx = _even_indices(n, max_steps)
            out[str(arm)] = {"position": [[float(v) for v in pos[j]] for j in idx],
                             "rotation_rpy": [[float(v) for v in rpy[j]] for j in idx],
                             "gripper": [float(grip[j]) for j in idx]}
        return out or None
    except Exception:  # odd / partial reply: record nothing rather than break the relay
        return None


def _jpeg_bytes(b64):
    if isinstance(b64, (bytes, bytearray)):
        return bytes(b64)
    s = str(b64).strip()
    if s.startswith("data:"):
        s = s.split(",", 1)[-1]
    return base64.b64decode(s)


def _p95(values):
    if not values:
        return None
    v = sorted(values)
    return round(v[min(len(v) - 1, math.ceil(0.95 * len(v)) - 1)], 1)


class TrialRecorder:
    """root/<YYYYmmdd_HHMMSS>/ holds meta.json, trial.jsonl (one line per robot request), frames/<index>_<camera>.jpg, summary.json."""

    def __init__(self, root="runs/trials"):
        self.root = Path(root)
        self.lock = threading.Lock()
        self.dir = None
        self.trial_id = None
        self.goal = None
        self.started = None
        self.started_mono = None
        self.rows = []

    def start(self, goal, **flags):
        """New folder + meta.json (goal, start time and the caller's profile flags); ends the previous trial. Returns the trial id."""
        self.end()
        with self.lock:
            self.root.mkdir(parents=True, exist_ok=True)
            stamp = time.time()
            trial_id = time.strftime("%Y%m%d_%H%M%S", time.localtime(stamp))
            while (self.root / trial_id).exists():  # two starts in one second must not share a folder (one dir per trial)
                stamp += 1
                trial_id = time.strftime("%Y%m%d_%H%M%S", time.localtime(stamp))
            d = self.root / trial_id
            (d / "frames").mkdir(parents=True, exist_ok=True)
            self.dir, self.trial_id, self.goal = d, trial_id, goal
            self.started, self.started_mono, self.rows = time.time(), time.monotonic(), []
            self._write_json(d / "meta.json", {"trial_id": trial_id, "goal": goal,
                                               "start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.started)),
                                               "start_epoch": round(self.started, 3), **flags})
            return trial_id

    def record(self, **kw):
        """Append one JSON line for one robot request (fields are the caller's: t, index, goal, subgoal, source, harness,
        state, latency_ms, elapsed_ms, error, actions)."""
        with self.lock:
            if self.dir is None:
                return None
            row = dict(kw)
            row.setdefault("t", time.time())
            line = json.dumps(row, default=_plain)
            with open(self.dir / "trial.jsonl", "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
            self.rows.append(row)
            return row

    def frame(self, index, images_b64):
        """Raw JPEG bytes of every camera of one request -> frames/<index>_<camera>.jpg. Returns the cameras written."""
        with self.lock:
            d = self.dir
        if d is None or not isinstance(images_b64, dict):
            return []
        written = []
        for cam, b64 in images_b64.items():
            try:
                data = _jpeg_bytes(b64)
                (d / "frames").mkdir(parents=True, exist_ok=True)
                (d / "frames" / ("%d_%s.jpg" % (int(index), cam))).write_bytes(data)
                written.append(cam)
            except Exception:  # a bad frame never fails the relay
                continue
        return written

    def end(self, summary=None):
        """Write summary.json (calls, wallx_calls, mean/p95 latency, parcels, duration, last state; merged with `summary`)."""
        with self.lock:
            d, rows = self.dir, self.rows
            if d is None:
                return None
            duration = time.monotonic() - (self.started_mono or time.monotonic())
            lat = [r["latency_ms"] for r in rows if isinstance(r.get("latency_ms"), (int, float))]
            last = rows[-1] if rows else {}
            harness = last.get("harness") if isinstance(last.get("harness"), dict) else {}
            out = {"trial_id": self.trial_id, "goal": self.goal,
                   "start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.started)) if self.started else None,
                   "end": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "calls": len(rows),
                   "wallx_calls": sum(1 for r in rows if r.get("source") == "wallx"),
                   "mean_latency_ms": round(sum(lat) / len(lat), 1) if lat else None,
                   "p95_latency_ms": _p95(lat),
                   "parcels": harness.get("parcels"),
                   "duration_s": round(duration, 1),
                   "last_state": last.get("state")}
            out.update(summary or {})
            try:
                self._write_json(d / "summary.json", out)
            finally:
                self.dir, self.rows = None, []
            return out

    @staticmethod
    def _write_json(path, obj):
        Path(path).write_text(json.dumps(obj, default=_plain, indent=2) + "\n", encoding="utf-8")
