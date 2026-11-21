#!/usr/bin/env python3
"""Trial-and-error agent for the real robot: an OpenAI model (default gpt-6.2-sol) runs harness trials, tunes the WALLX
instruction wording, resets the table with direct WALLX commands, looks at the head camera, and writes a report.
It gets NO shell: only the fixed tools below. Robot launch command, relay flags and limits are fixed here.

  ROBOT_ASKPASS=/path/to/askpass.sh python3 -I scripts/trial_agent.py [--model gpt-6.2-sol] [--sessions 10] [--minutes 70]

Writes runs/agent/<ts>/ (wording.json, trial logs, images, montages) and results/agent/REPORT_<ts>.md.
"""
import argparse, base64, json, os, re, socket, subprocess, sys, time, urllib.error, urllib.request
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ENV = {k: v.strip().strip('"').strip("'") for k, v in (l.strip().split("=", 1) for l in open(ROOT / ".env") if "=" in l and not l.lstrip().startswith("#"))}
PY = str(ROOT / ".venv/bin/python")
GOAL = "Read each parcel label and sort it: ZIP 940/941/950/951 to the blue bin, 900-902 to the red bin, anything else to the black bin."
KEYS = ("PLACE", "REJECT", "INSPECT", "CENTER", "HOME", "LIFT2", "STAND", "PICK", "PICK_STANDING", "PICK_WIDE", "PLACE_WIDE")
FIELDS = {"mat", "arm", "where", "obj", "fix", "fix_side", "lift", "lift_side", "bin_where"}
CLIENT = ("source /opt/xr/bot/x2robot_client/share/x2robot_client/scripts/session_env.sh >/dev/null 2>&1; timeout -s INT {sec} python3 "
          "/opt/xr/bot/x2robot_client/share/x2robot_client/scripts/start_desktop.py --model-address 192.168.1.157:8001 --instruction {instr} "
          "--robot-mode end_pose --infer-type sync --embodiment teletwin_desktop/gripper-g --image-wire-format base64 --attach --home-on-stop")

SYSTEM = """You are the trial-and-error engineer for team LabelLoop at a robotics hackathon. You drive a REAL dual-arm robot (left + right
6-DOF arms, parallel grippers) over a green table, only through the tools given. Goal: make our harness sort a parcel IN ONE GO
(no retries) and find the instruction wording WALLX (the organisers' vision-language-action model, which generates every arm
motion) understands first time.

Scene (head camera looks down from behind the robot, image top = far side of the table):
- Table ~111 x 68.5 cm usable. Robot base at the near edge. Left arm works the left half, right arm the right half.
- Far edge, left of centre: the BLUE bin = a navy/dark-blue open shoe box. Far edge, right of centre: the RED bin = a red Nike
  shoe box. Right edge, mid-depth: the BLACK bin = a black-and-white box. ALL THREE BINS ARE BOXES TOO: in trial 2 the
  instruction "flip the box over" made WALLX grab the red bin and drag it onto the table. Word every instruction so the target
  is unambiguous: the parcel is the small flat retail box lying on the green table (e.g. a Bhindi Masala food box), never a bin.
- Parcels carry a QR label 420/<ZIP5>/<seq>; ZIP3 940/941/950/951 -> blue, 900-902 -> red, else / no label -> black.
  Test parcels: a flat food carton (420/95126/001 -> BLUE) and a flat black gift box (Miro, 420/90012/... -> RED); the harness
  reads the label itself. At the start of this run the black box lies label-up in the middle of the table (-> red bin).

What we learned so far (do not repeat these):
- Trial 2 (old wording "flip the box over"): WALLX grabbed the RED bin and dragged it onto the table.
- Agent trial 1 (PLACE "Put the small flat ... parcel on the green table into the open blue bin."): the label was read and routed
  correctly, but WALLX reached across with the RIGHT arm, grabbed the BLUE BIN and tipped it; the parcel (front left, by the left
  gripper) was never touched. WALLX is strongly drawn to the shoe-box bins. So every template MUST name the acting arm ({arm}) and
  the parcel's position ({where}), describe the parcel as the small flat package lying on the green table, and mention the bin
  only as the destination. Try e.g. "use the {arm} arm to pick up the small flat package lying at the {where} of the green table,
  then drop it into the {mat} bin". Splitting into two sub-goals is not available; one instruction per action.
- Fixed since then (Claude, 13:30): a parcel already on the table at the start was never read in SCAN (its label was filtered out),
  so the harness wasted a CENTER slide first; and small dense QR codes (the black box) are now read via the printed ZIP (OCR,
  confirmed over 2 frames). Expect IDLE -> SCAN -> PLACE within ~3 calls now; if you see CENTER, the label was not read.
- Agent run 2 (13:16): trial 1 raw image + long arm/position PLACE wording: WALLX worked on the parcel but could not lift the wide
  black box; trial 2 (overlay "parcel", same wording): WALLX lifted the BLACK bin into the red bin. The harness now latches HOLD by
  itself when a non-target bin moves (flag "bin_moved"); that ends the trial, so reset the scene (human) before the next one.
- Agent run 3 (13:33): the organisers' verbatim PLACE phrase, raw image: WALLX tipped the RED (target) bin. Every wording so far
  ends with WALLX grabbing a bin. NEW HARD RULE (run_trial stage=true): the harness splits delivery into WALLX sub-goals and edits
  the head image WALLX sees: wide parcel -> STAND (all bins painted out; one arm steadies, the other tips the box onto its 38 mm
  edge; success = footprint shrinks 40 %) -> PICK "pick up the box" / "pick up the standing box from the top with the {arm} arm"
  (all bins painted out) -> PLACE (only the target bin visible). Wording keys STAND, PICK, PICK_STANDING, PLACE. TEST stage=true
  FIRST (default wording), then tune STAND / PICK_STANDING wording. Report per sub-goal which worked.
- Agent run 4 (13:56): STAND failed 3/3 (the box only slid / rotated); "Use both arms to squeeze the short ends ... and roll it" LIFTED
  the Miro box clean off the table (first real pick of a wide parcel!). The bin_moved stop that followed was a FALSE alarm (the lifted
  box hid the blue bin); fixed: the bin check is off once the parcel has left its table spot. Now (stage=true): wide parcels skip
  STAND and get PICK_WIDE "use both arms to squeeze the short ends of the box and lift it up" then PLACE_WIDE "put the box into the
  {mat} bin with both arms" (only the target bin visible). PICK / PLACE get up to 35 calls (~50 s) per try: slow is fine, finish it.
  Use run_trial seconds 200-300.
- Agent run 5 (14:24): two-arm squeeze + retry: the RIGHT arm lifted the box, then carried it to the BLACK bin at the right edge
  (not the red target) and bumped the blue bin. New: {where} is frozen once routed; {bin_where} = the target bin's place in words
  ("at the back right" for red, "at the back left" blue, "at the right edge" black). Try PLACE_WIDE e.g. "keep holding the box and
  put it into the {mat} bin {bin_where}".
- OLDER TEST PLAN, in this order, from a matched start: (1) the organisers' own phrase verbatim, PLACE "put the box into the {mat} bin",
  raw image (WALLX was likely fine-tuned on it; it was never tested with the label already read); (2) the same with overlay "mask"
  (non-target bins painted out); (3) the smaller food carton instead of the wide black box; (4) grasp phrases.
- PARCEL SIZES (mm, L x W x H) vs the 97.5 mm gripper opening: Miro black box 216 x 165 x 38, Bhindi food carton 165 x 102 x 38,
  IQBAR 267 x 102 x 64, perfume box 152 x 102 x 51 -> ALL too wide for ONE gripper lying flat (that is why single-arm pickups slid
  and tilted them). Use TWO-ARM wording for these, e.g. PLACE "pick up the box with both arms and put it into the {mat} bin" or
  "use both arms to lift the flat package at the {where} of the green table by its two short ends and put it into the {mat} bin".
  Wallet 114 x 89 x 19 fits one gripper (grasp across the 89 mm side). The team chooses the judged props, so also report which
  parcel sizes / bin types worked best (a recommendation for the props we bring).
- WORDING MUST BE GENERIC: the judged cases use unknown parcels, so never use a product name (no "Bhindi Masala").

Visual prompt (run_trial "overlay"): WALLX never reads grid numbers in text, but our relay can draw on the head image it sees:
  none = raw image; parcel = magenta box round the parcel, green box round the destination bin, yellow arrow parcel -> bin;
  mask = parcel + the NON-target bins painted out with table green; grid = parcel + a numbered grid over the free table.
WALLX was trained on raw images, so overlays may help or confuse it: test them (one change at a time) and report which works.
Grasp style is yours to test in wording: "from the top", "by the short sides", "with both arms" (only for big parcels).

How a trial works: run_trial starts our relay + harness, then a robot session. The harness reads the label from the head camera,
picks a sub-goal and sends WALLX one short instruction at a time from the wording templates (set_wording). Templates may use
{mat} (bin colour: blue/red/black), {where} (parcel position on the table: e.g. "front left", "middle", "back right"),
{arm} (left/right: the arm nearest the parcel or the lane's arm), {obj}. Default event wording:
  PLACE  "put the box into the {mat} bin"     REJECT "put the box into the black bin"     INSPECT "flip the box over"
  CENTER "slide the box to the middle of the table with the {arm} arm"   HOME "put both arms back to the home position"
The harness verifies success itself (bin region changed + parcel gone) and retries (2 tries per action).
You get a summary (states, transitions, codes read, verdict, timing) and a montage of head-camera frames for each trial.

Table reset: after a success the parcel is in a bin. Use wallx_direct to have WALLX put it back, e.g. "take the small box out of the
blue bin and put it on the green table in front of the left arm", then snapshot to check. Vary the start position across trials
(left, middle, right, near an edge) once the basic case works.

Rules:
- SAFETY FIRST. If a bin has moved off its place, anything fell off the table, an arm looks stuck or collided, or you cannot reset
  the table: STOP and call finish explaining what the human must fix. Never ask WALLX to move, lift or touch a bin except to drop
  the parcel into it or take the parcel out of it.
- Change one thing at a time; keep what works. Prefer short, concrete instructions naming the parcel and the destination bin by
  colour and position ("the blue bin at the back left").
- You have a limited number of robot sessions (each run_trial / wallx_direct / snapshot is one). Budget them.
- End with finish: a concise report: per-trial table (wording, start position, result, time, retries), the best wording found,
  failure modes seen, and concrete recommendations for the harness."""

TOOLS = [
    {"type": "function", "name": "set_wording", "description": "Set the WALLX instruction templates for the next trials (only the keys given change).",
     "parameters": {"type": "object", "properties": {k: {"type": "string"} for k in KEYS}, "additionalProperties": False}},
    {"type": "function", "name": "run_trial", "description": "Run one full harness trial on the robot (relay + robot session). Returns a summary and a head-camera montage.",
     "parameters": {"type": "object", "properties": {"seconds": {"type": "integer", "description": "session length, 60-300"},
                                                     "note": {"type": "string", "description": "what you are testing"},
                                                     "overlay": {"type": "string", "enum": ["none", "parcel", "mask", "grid"], "description": "visual prompt on the image WALLX sees"},
                                                     "stage": {"type": "boolean", "description": "staged delivery with hard bin hiding (see rules)"}},
                    "required": ["seconds", "note", "overlay", "stage"], "additionalProperties": False}},
    {"type": "function", "name": "wallx_direct", "description": "Send ONE raw instruction straight to WALLX (no harness) for a short robot session, e.g. to reset the table. Returns before/after frames.",
     "parameters": {"type": "object", "properties": {"instruction": {"type": "string"}, "seconds": {"type": "integer", "description": "20-90"}},
                    "required": ["instruction", "seconds"], "additionalProperties": False}},
    {"type": "function", "name": "snapshot", "description": "Arms hold still; returns the current head-camera image of the table.",
     "parameters": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"type": "function", "name": "finish", "description": "End the session with the final report (markdown).",
     "parameters": {"type": "object", "properties": {"report": {"type": "string"}}, "required": ["report"], "additionalProperties": False}},
]


class Agent:
    def __init__(self, a):
        self.a, self.ts = a, time.strftime("%Y%m%d_%H%M%S")
        self.dir = ROOT / "runs/agent" / self.ts
        self.dir.mkdir(parents=True, exist_ok=True)
        self.wording = {}
        (self.dir / "wording.json").write_text("{}")
        self.sessions, self.n, self.t0, self.journal = 0, 0, time.time(), []

    # ---- robot plumbing (fixed commands only) ----
    def relay(self, *flags, log=None):
        args = [PY, "-m", "labelloop.robots.event_server", "--host", "0.0.0.0", "--port", "8001", "--profile", "real",
                "--save-images", str(self.dir / "images"), "--wording", str(self.dir / "wording.json"), *flags]
        if log:
            args += ["--log", str(log)]
        p = subprocess.Popen(args, cwd=ROOT, stdout=open(self.dir / "relay.out", "a"), stderr=subprocess.STDOUT)
        for _ in range(40):
            try:
                socket.create_connection(("127.0.0.1", 8001), 0.5).close()
                return p
            except OSError:
                time.sleep(0.5)
        p.kill()
        raise RuntimeError("relay did not start")

    def robot(self, instr, sec):
        cmd = CLIENT.format(sec=int(sec), instr="\\\"%s\\\"" % instr.replace('"', "").replace("\\", "").replace("$", "").replace("`", ""))
        env = {**os.environ, "SSH_ASKPASS": os.environ["ROBOT_ASKPASS"], "SSH_ASKPASS_REQUIRE": "force", "DISPLAY": ":0"}
        r = subprocess.run(["ssh", "-o", "ConnectTimeout=8", "xr@192.168.1.21", 'docker exec desktop_client-robot_client-1 bash -c "%s"' % cmd],
                           env=env, capture_output=True, text=True, timeout=sec + 60)
        return [l for l in (r.stdout + r.stderr).splitlines() if re.search(r"START|STOP|done|rror", l) and not re.search(r"key|token|secret|password", l, re.I)][-6:]

    def session(self, flags, instr, sec, log=None):
        self.sessions += 1
        before = set(self.frames())
        p = self.relay(*flags, log=log)
        try:
            out = self.robot(instr, sec)
        finally:
            p.terminate()
            p.wait(10)
        return out, [f for f in self.frames() if f not in before]

    def frames(self):
        return sorted((self.dir / "images").glob("*/*_front.jpg"), key=lambda f: (f.parent.name, int(f.name.split("_")[0])))

    def montage(self, files, name, k=9):
        files = [files[i] for i in np.linspace(0, len(files) - 1, min(k, len(files))).astype(int)] if files else []
        ims = []
        for f in files:
            im = cv2.resize(cv2.imread(str(f)), (480, 270))
            cv2.putText(im, f.name.split("_")[0], (8, 28), 0, 0.9, (0, 0, 255), 2)
            ims.append(im)
        if not ims:
            return None
        while len(ims) % 3:
            ims.append(np.zeros_like(ims[0]))
        img = np.vstack([np.hstack(ims[i:i + 3]) for i in range(0, len(ims), 3)])
        cv2.imwrite(str(self.dir / name), img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return base64.b64encode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])[1]).decode()

    # ---- tools ----
    def set_wording(self, **kw):
        for k, v in kw.items():
            bad = set(re.findall(r"{(\w*)}", v)) - FIELDS
            if k not in KEYS or not v.strip() or len(v) > 160 or bad or (k == "PLACE" and "{mat}" not in v):
                return f"rejected {k}: max 160 chars, fields only {sorted(FIELDS)}, PLACE must contain {{mat}}", None
            self.wording[k] = [v.strip()]
        (self.dir / "wording.json").write_text(json.dumps(self.wording, indent=1))
        return "wording now: " + json.dumps(self.wording), None

    def run_trial(self, seconds, note, overlay="none", stage=False):
        self.n += 1
        log = self.dir / f"t{self.n}.jsonl"
        out, files = self.session((["--overlay", overlay] if overlay != "none" else []) + (["--stage"] if stage else []), GOAL, max(60, min(300, seconds)), log)
        s = summarize(log)
        s.update(trial=self.n, note=note, overlay=overlay, stage=stage, wording=self.wording, client=out)
        self.journal.append({k: s[k] for k in ("trial", "note", "overlay", "wording", "verdict", "seconds_to_verdict", "retries")})
        return json.dumps(s), self.montage(files, f"t{self.n}.jpg")

    def wallx_direct(self, instruction, seconds):
        out, files = self.session(["--passthrough"], instruction[:200], max(20, min(90, seconds)))
        return json.dumps({"client": out, "frames": len(files)}), self.montage(files[:1] + files[-2:], f"direct{self.sessions}.jpg", 3)

    def snapshot(self):
        out, files = self.session(["--hold"], "hold still", 15)
        return json.dumps({"client": out}), self.montage(files[-1:], f"snap{self.sessions}.jpg", 1)

    def save(self, name, args, out):  # progress survives a crash / exhausted credits: journal + live notes after every tool call
        with open(self.dir / "progress.jsonl", "a") as f:
            f.write(json.dumps({"t": time.strftime("%H:%M:%S"), "tool": name, "args": args, "out": out[:4000]}) + "\n")
        (self.dir / "journal.json").write_text(json.dumps(self.journal, indent=1))

    # ---- loop ----
    def ask(self, inp, prev):
        body = {"model": self.a.model, "instructions": SYSTEM, "input": inp, "tools": TOOLS, "reasoning": {"effort": "medium"}}
        if prev:
            body["previous_response_id"] = prev
        req = urllib.request.Request("https://api.openai.com/v1/responses", json.dumps(body).encode(),
                                     {"Authorization": "Bearer " + ENV["OPENAI_API_KEY"], "content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"OpenAI {e.code}: {e.read()[:500]!r}") from None

    def pending(self, rid):  # function calls the previous run left unanswered (its last finish): the thread needs their outputs first
        req = urllib.request.Request(f"https://api.openai.com/v1/responses/{rid}", headers={"Authorization": "Bearer " + ENV["OPENAI_API_KEY"]})
        with urllib.request.urlopen(req, timeout=60) as r:
            out = json.load(r).get("output", [])
        return [{"type": "function_call_output", "call_id": o["call_id"], "output": "ok (that run ended)"} for o in out if o.get("type") == "function_call"]

    def run(self):
        memo = ROOT / "runs/agent/last_response_id"  # same thread across runs (Responses API keeps the conversation server-side)
        prev = memo.read_text().strip() if memo.exists() and not self.a.fresh else None
        past = "" if prev else "\n\n".join(f.read_text()[:6000] for f in sorted((ROOT / "results/agent").glob("REPORT_*.md"))[-4:])
        inp, report = [{"role": "user", "content": [{"type": "input_text", "text":
               f"New run. You have {self.a.sessions} robot sessions and {self.a.minutes} minutes. The human just reset the table: "
               "bins in place, the test parcel somewhere on the table. Take a snapshot first. Learn from the earlier runs"
               + (" in this thread." if prev else ":\n\n" + past)}]}], None
        if self.a.video:  # reference footage from the human: frames as images (the API takes no video)
            inp[0]["content"].append({"type": "input_text", "text": (
                "REFERENCE VIDEO from the human (frames in time order, two 3x3 grids): X Square's own robot (the WALLX family) sorting "
                "parcels at speed. Study exactly how it grasps flat parcels (which arm, from where, side vs top, two arms, flips) and how "
                "it drops them into bins, then translate that into our sub-goal wording (PICK_WIDE / PLACE_WIDE / PICK / PLACE / STAND) "
                "and test it. The human says picking up or flipping a parcel should be this easy; get our robot to at least do it, slowly is fine.")})
            for img in video_grids(self.a.video, self.dir):
                inp[0]["content"].append({"type": "input_image", "image_url": "data:image/jpeg;base64," + img})
        if prev:
            inp = self.pending(prev) + inp
        while report is None:
            left = self.a.sessions - self.sessions
            r = self.ask(inp, prev)
            prev, inp = r["id"], []
            memo.write_text(prev)
            calls = [o for o in r.get("output", []) if o.get("type") == "function_call"]
            text = "".join(c.get("text", "") for o in r.get("output", []) if o.get("type") == "message" for c in o.get("content", []))
            if text:
                print("model:", text[:400], flush=True)
            if not calls:
                inp = [{"role": "user", "content": [{"type": "input_text", "text": "Continue with a tool call (call finish when done)."}]}]
                continue
            for c in calls:
                args = json.loads(c.get("arguments") or "{}")
                print(f"[{time.strftime('%H:%M:%S')}] {c['name']} {json.dumps(args)[:200]}", flush=True)
                img = None
                if c["name"] == "finish":
                    report = args["report"]
                    out = "ok"
                elif c["name"] != "set_wording" and (left <= 0 or time.time() - self.t0 > self.a.minutes * 60):
                    out = "LIMIT REACHED: no more robot sessions. Call finish now with your report."
                else:
                    try:
                        out, img = getattr(self, c["name"])(**args)
                    except Exception as e:
                        out = f"tool error: {type(e).__name__}: {e}"[:500]
                print("   ->", out[:300], flush=True)
                self.save(c["name"], args, out)
                inp.append({"type": "function_call_output", "call_id": c["call_id"], "output": out})
                if img:
                    inp.append({"role": "user", "content": [{"type": "input_text", "text": f"Head-camera frames from {c['name']}:"},
                                                            {"type": "input_image", "image_url": "data:image/jpeg;base64," + img}]})
        rep = ROOT / "results/agent" / f"REPORT_{self.ts}.md"
        rep.parent.mkdir(parents=True, exist_ok=True)
        rep.write_text(f"# Trial agent report {self.ts} ({self.a.model}, {self.sessions} robot sessions)\n\n{report}\n\n## Journal\n"
                       + "\n".join("- " + json.dumps(j) for j in self.journal) + f"\n\nArtifacts: {self.dir.relative_to(ROOT)}\n")
        print("REPORT", rep, flush=True)


def video_grids(path, out_dir, n=18, per=9):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in np.linspace(0, total - 1, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            f = cv2.resize(f, (360, int(360 * f.shape[0] / f.shape[1])))
            cv2.putText(f, "%.0fs" % (i / (cap.get(cv2.CAP_PROP_FPS) or 25)), (8, 30), 0, 0.9, (0, 0, 255), 2)
            frames.append(f)
    grids = []
    for k in range(0, len(frames), per):
        g = frames[k:k + per]
        while len(g) % 3:
            g.append(np.zeros_like(g[0]))
        img = np.vstack([np.hstack(g[i:i + 3]) for i in range(0, len(g), 3)])
        cv2.imwrite(str(Path(out_dir) / f"video_{k // per}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        grids.append(base64.b64encode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])[1]).decode())
    return grids


def summarize(log):
    rows = [json.loads(l) for l in open(log)] if Path(log).exists() else []
    st = [r for r in rows if "state" in r]
    lat = [r["latency_s"] for r in rows if "latency_s" in r]
    tr = [(r["call"], *t) for r in st for t in r["transitions"]]
    flags = [f for r in st for f in r["flags"]]
    t0 = st[0]["t"] if st else 0
    verdict = next((("ok" if t[2] == "DONE" and t[3] == "ok" else t[3]) for t in tr if t[2] == "DONE"), "no verdict")
    done_t = next((round(r["t"] - t0, 1) for r in st for t in r["transitions"] if t[1] == "DONE"), None)
    return {"verdict": verdict, "seconds_to_verdict": done_t, "calls": len(st), "states": dict(Counter(r["state"] for r in st)),
            "transitions": [f"call {c}: {a}->{b} ({w})" for c, a, b, w in tr][:40], "flags": flags[:20],
            "codes_read": sorted({c for r in st for c in r.get("confirmed", [])}), "instructions_sent": sorted({r["instruction"] for r in st if r.get("instruction")}),
            "retries": sum(1 for t in tr if t[3] == "retry"), "wallx_latency_ms": round(1000 * sum(lat) / len(lat)) if lat else None}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-6.2-sol")
    ap.add_argument("--sessions", type=int, default=12)
    ap.add_argument("--minutes", type=int, default=75)
    ap.add_argument("--video", default="", help="reference video: frames are shown to the model at the start")
    ap.add_argument("--fresh", action="store_true", help="start a new thread (default: continue the last one; earlier reports are given to a new thread)")
    assert os.environ.get("ROBOT_ASKPASS"), "set ROBOT_ASKPASS"
    Agent(ap.parse_args()).run()
