"""Story mode: every recorded trial (runs/trials/<ts>/) and every trial-agent run (runs/agent/<ts>/ and/or results/agent/REPORT_<ts>.md)
merged in time order, each with the wording used, overlay / stage, the sub-goal timeline (blocks per harness state), verdict, time,
retries and flags. stdlib only. A malformed file never fails the list: that item just has less in it."""
import json
import re
import time
from pathlib import Path

TS_RE = re.compile(r"^\d{8}_\d{6}$")
OUTCOMES = ("ok", "failed")
STAGED = ("STAND", "PICK")
MATCH_S = (-30.0, 10.0)  # a recorded trial belongs to an agent trial if it starts within [first - 30 s, last + 10 s] of its log


def _json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _jsonl(path):
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    v = json.loads(line)
                except ValueError:
                    continue
                if isinstance(v, dict):
                    out.append(v)
    except OSError:
        pass
    return out


def ts_epoch(ts):
    try:
        return time.mktime(time.strptime(ts, "%Y%m%d_%H%M%S"))
    except (TypeError, ValueError):
        return None


def _num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _p95(v):
    v = sorted(v)
    return v[max(0, -(-len(v) * 95 // 100) - 1)] if v else None


def timeline(states):
    """[(state, t)] in row order -> blocks [{state, from, to (row indices), s (seconds until the next block's first row)}]."""
    blocks = []
    for i, (st, t) in enumerate(states):
        if blocks and blocks[-1]["state"] == st:
            blocks[-1]["to"] = i
        else:
            blocks.append({"state": st, "from": i, "to": i, "t0": t})
    for k, b in enumerate(blocks):
        t1 = blocks[k + 1]["t0"] if k + 1 < len(blocks) else states[b["to"]][1]  # (the next block still holds its t0 here)
        t0 = b.pop("t0")
        b["s"] = round(max(0.0, t1 - t0), 1) if t0 is not None and t1 is not None else None
    return blocks


def _transitions(h):
    return [t for t in (h.get("transitions") or []) if isinstance(t, (list, tuple)) and len(t) >= 3]


def judge(harness_rows, live=False):
    """(verdict, outcomes, retries, flags) from harness dicts in row order. verdict: ok (every parcel ok) | failed | mixed | hold (latched
    HOLD, no outcome) | live (still recording) | none."""
    outcomes, retries, flags, hold = [], 0, [], False
    for i, h in enumerate(harness_rows):
        if not isinstance(h, dict):
            continue
        for t in _transitions(h):
            if t[2] in OUTCOMES and t[1] in ("DONE", "IDLE"):
                outcomes.append(t[2])
            retries += t[2] == "retry"
            hold |= t[1] == "HOLD"
        hold |= h.get("state") == "HOLD"
        for f in h.get("flags") or []:
            flags.append({"i": i, "flag": str(f)})
    if outcomes:
        verdict = "ok" if all(o == "ok" for o in outcomes) else "failed" if all(o == "failed" for o in outcomes) else "mixed"
    else:
        verdict = "hold" if hold else "live" if live else "none"
    return verdict, outcomes, retries, flags


def trial_item(d):
    """Story entry of one recorded trial folder."""
    from labelloop.dashboard.server import read_rows  # (late: no import cycle)
    meta, summary, rows = _json(d / "meta.json") or {}, _json(d / "summary.json"), read_rows(d)
    hs = [r.get("harness") if isinstance(r.get("harness"), dict) else {} for r in rows]
    verdict, outcomes, retries, flags = judge(hs, live=summary is None)
    name = {"scripted": "ROBOT", "hold": "DONE"}
    states = [(h.get("state") or name.get(r.get("source"), str(r.get("source") or "—").upper()), _num(r.get("t"))) for h, r in zip(hs, rows)]
    words = []
    for r in rows:
        w = r.get("subgoal")
        if r.get("source") == "wallx" and isinstance(w, str) and w not in words:
            words.append(w)
    lat = [v for v in (_num(r.get("latency_ms")) for r in rows) if v is not None]
    t0 = _num(meta.get("start_epoch")) or ts_epoch(d.name)
    t_end = _num(rows[-1].get("t")) if rows else None
    dur = _num((summary or {}).get("duration_s"))
    if dur is None and t_end is not None and t0 is not None:
        dur = round(max(0.0, t_end - t0), 1)
    overlay = meta.get("overlay")
    if overlay is None:
        overlay = "front_overlay" if next((d / "frames").glob("*_front_overlay.jpg"), None) else "none"
    lanes = sorted({h.get("lane") for h in hs if h.get("lane")})
    codes = []
    for h in hs:
        for c in h.get("confirmed") or []:
            if c not in codes:
                codes.append(c)
    per_state = {}
    for b in timeline(states):
        per_state[b["state"]] = round(per_state.get(b["state"], 0) + (b["s"] or 0), 1)
    return {"kind": "trial", "id": d.name, "epoch": t0, "start": meta.get("start"), "goal": meta.get("goal"), "wording": words,
            "overlay": overlay, "stage": bool(meta.get("stage")) or any(s in STAGED for s, _ in states),
            "blocks": timeline(states), "rows": len(rows), "verdict": verdict, "outcomes": outcomes, "retries": retries, "flags": flags,
            "duration_s": dur, "calls": len(rows), "wallx_calls": sum(1 for r in rows if r.get("source") == "wallx"),
            "mean_latency_ms": round(sum(lat) / len(lat), 1) if lat else None, "p95_latency_ms": _p95(lat), "per_state_s": per_state,
            "lanes": lanes, "codes": codes, "live": summary is None, "replay": d.name, "t_end": t_end}


def _report(path):
    """(title, journal entries parsed from its '## Journal' list, text)."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return None, [], None
    heads = [l[2:].strip() for l in text.splitlines() if l.startswith("# ")]
    journal, inj = [], False
    for l in text.splitlines():
        if l.startswith("## "):
            inj = l.strip().lower() == "## journal"
        elif inj and l.startswith("- {"):
            try:
                v = json.loads(l[2:])
                if isinstance(v, dict):
                    journal.append(v)
            except ValueError:
                pass
    return (heads[1] if len(heads) > 1 else heads[0] if heads else None), journal, text


def agent_runs(agent_root, reports_root):
    """{ts: {"dir": Path | None, "report": Path | None}} of every agent run folder and report."""
    runs = {}
    for root, key in ((agent_root, "dir"), (reports_root, "report")):
        if not root or not Path(root).is_dir():
            continue
        for p in Path(root).iterdir():
            ts = p.name if key == "dir" else (p.stem[len("REPORT_"):] if p.name.startswith("REPORT_") and p.suffix == ".md" else "")
            if TS_RE.match(ts) and (p.is_dir() if key == "dir" else p.is_file()):
                runs.setdefault(ts, {"dir": None, "report": None})[key] = p
    return runs


def agent_item(ts, run, trials):
    """Story entry of one agent run: its trials (journal + t<N>.jsonl logs) matched to recorded trials for replay."""
    title, journal, _ = _report(run["report"]) if run.get("report") else (None, [], None)
    if run.get("dir"):
        j = _json(run["dir"] / "journal.json")
        journal = j if isinstance(j, list) and j else journal
    out = []
    for e in journal:
        if not isinstance(e, dict):
            continue
        n = e.get("trial")
        log = _jsonl(run["dir"] / ("t%s.jsonl" % n)) if run.get("dir") and n is not None else []
        st = [r for r in log if "state" in r]
        verdict, outcomes, retries, flags = judge(st)
        ts_ = [_num(r.get("t")) for r in st if _num(r.get("t")) is not None]
        replay = None
        if ts_:
            lo, hi = ts_[0] + MATCH_S[0], ts_[-1] + MATCH_S[1]
            replay = next((t["id"] for t in trials if t.get("epoch") is not None and lo <= t["epoch"] <= hi), None)
        wording = e.get("wording")
        out.append({"trial": n, "note": e.get("note"), "overlay": e.get("overlay", "none"), "stage": bool(e.get("stage")),
                    "wording": wording if isinstance(wording, dict) else {}, "verdict": e.get("verdict") or verdict,
                    "seconds_to_verdict": _num(e.get("seconds_to_verdict")), "retries": e.get("retries", retries),
                    "blocks": timeline([(r.get("state"), _num(r.get("t"))) for r in st]), "rows": len(st), "flags": flags,
                    "epoch": ts_[0] if ts_ else None, "replay": replay})
    ok = sum(1 for t in out if t["verdict"] == "ok")
    return {"kind": "agent", "id": ts, "epoch": ts_epoch(ts), "title": title, "trials": out, "verdict": "%d/%d ok" % (ok, len(out)),
            "report": bool(run.get("report")), "has_dir": bool(run.get("dir"))}


def build(trials_root, agent_root=None, reports_root=None):
    """The merged story, oldest first. Trials that an agent run drove carry "agent": <ts>."""
    trials = []
    if Path(trials_root).is_dir():
        for d in sorted(Path(trials_root).iterdir()):
            if d.is_dir() and TS_RE.match(d.name):
                try:
                    trials.append(trial_item(d))
                except Exception:  # one broken folder never hides the rest
                    trials.append({"kind": "trial", "id": d.name, "epoch": ts_epoch(d.name), "verdict": "none", "blocks": [], "flags": [],
                                   "wording": [], "broken": True, "replay": d.name})
    agents = [agent_item(ts, r, trials) for ts, r in sorted(agent_runs(agent_root, reports_root).items())]
    for a in agents:
        for t in a["trials"]:
            for tr in trials:
                if t["replay"] == tr["id"]:
                    tr["agent"], tr["agent_trial"] = a["id"], t["trial"]
    return sorted(trials + agents, key=lambda it: (it.get("epoch") or 0, it["kind"] != "agent"))


def agent_detail(ts, agent_root=None, reports_root=None):
    """Journal, report text (markdown as plain text) and the last progress notes of one agent run; None if it does not exist."""
    run = agent_runs(agent_root, reports_root).get(ts)
    if not run:
        return None
    title, journal, text = _report(run["report"]) if run.get("report") else (None, [], None)
    if run.get("dir"):
        j = _json(run["dir"] / "journal.json")
        journal = j if isinstance(j, list) and j else journal
    progress = _jsonl(run["dir"] / "progress.jsonl")[-60:] if run.get("dir") else []
    wording = _json(run["dir"] / "wording.json") if run.get("dir") else None
    return {"id": ts, "title": title, "journal": journal, "report": text, "progress": progress, "wording": wording}
