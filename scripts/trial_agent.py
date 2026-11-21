#!/usr/bin/env python3
"""Trial-and-error agent for the real robot: an OpenAI model (default gpt-6.2-sol) runs harness trials, tunes the WALLX
instruction wording, resets the table with direct WALLX commands, looks at the head camera, and writes a report.
It gets NO shell: only the fixed tools below. Robot launch command, relay flags and limits are fixed here.

  ROBOT_ASKPASS=/path/to/askpass.sh python3 -I scripts/trial_agent.py [--model gpt-6.2-sol] [--sessions 10] [--minutes 70]

Writes runs/agent/<ts>/ (wording.json, trial logs, images, montages) and results/agent/REPORT_<ts>.md.
"""
import argparse, base64, json, os, re, socket, subprocess, sys, time, urllib.request
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ENV = {k: v.strip().strip('"').strip("'") for k, v in (l.strip().split("=", 1) for l in open(ROOT / ".env") if "=" in l and not l.lstrip().startswith("#"))}
PY = str(ROOT / ".venv/bin/python")
GOAL = "Read each parcel label and sort it: ZIP 940/941/950/951 to the blue bin, 900-902 to the red bin, anything else to the black bin."
KEYS = ("PLACE", "REJECT", "INSPECT", "CENTER", "HOME", "LIFT2")
FIELDS = {"mat", "arm", "where", "obj"}
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
  Current test parcel: a flat food carton, label 420/95126/001 -> BLUE bin (for your eyes only; keep wording generic).

What we learned so far (do not repeat these):
- Trial 2 (old wording "flip the box over"): WALLX grabbed the RED bin and dragged it onto the table.
- Agent trial 1 (PLACE "Put the small flat ... parcel on the green table into the open blue bin."): the label was read and routed
  correctly, but WALLX reached across with the RIGHT arm, grabbed the BLUE BIN and tipped it; the parcel (front left, by the left
  gripper) was never touched. WALLX is strongly drawn to the shoe-box bins. So every template MUST name the acting arm ({arm}) and
  the parcel's position ({where}), describe the parcel as the small flat package lying on the green table, and mention the bin
  only as the destination. Try e.g. "use the {arm} arm to pick up the small flat package lying at the {where} of the green table,
  then drop it into the {mat} bin". Splitting into two sub-goals is not available; one instruction per action.
- WORDING MUST BE GENERIC: the judged cases use unknown parcels, so never use a product name (no "Bhindi Masala").

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
                                                     "note": {"type": "string", "description": "what you are testing"}}, "required": ["seconds", "note"], "additionalProperties": False}},
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

    def run_trial(self, seconds, note):
        self.n += 1
        log = self.dir / f"t{self.n}.jsonl"
        out, files = self.session([], GOAL, max(60, min(300, seconds)), log)
        s = summarize(log)
        s.update(trial=self.n, note=note, wording=self.wording, client=out)
        self.journal.append({k: s[k] for k in ("trial", "note", "wording", "verdict", "seconds_to_verdict", "retries")})
        return json.dumps(s), self.montage(files, f"t{self.n}.jpg")

    def wallx_direct(self, instruction, seconds):
        out, files = self.session(["--passthrough"], instruction[:200], max(20, min(90, seconds)))
        return json.dumps({"client": out, "frames": len(files)}), self.montage(files[:1] + files[-2:], f"direct{self.sessions}.jpg", 3)

    def snapshot(self):
        out, files = self.session(["--hold"], "hold still", 15)
        return json.dumps({"client": out}), self.montage(files[-1:], f"snap{self.sessions}.jpg", 1)

    # ---- loop ----
    def ask(self, inp, prev):
        body = {"model": self.a.model, "instructions": SYSTEM, "input": inp, "tools": TOOLS, "reasoning": {"effort": "medium"}}
        if prev:
            body["previous_response_id"] = prev
        req = urllib.request.Request("https://api.openai.com/v1/responses", json.dumps(body).encode(),
                                     {"Authorization": "Bearer " + ENV["OPENAI_API_KEY"], "content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.load(r)

    def run(self):
        inp, prev, report = [{"role": "user", "content": [{"type": "input_text", "text":
               f"Start. You have {self.a.sessions} robot sessions and {self.a.minutes} minutes. The table was just reset by the human: "
               "bins in place, the Bhindi box somewhere on the table. Take a snapshot first."}]}], None, None
        while report is None:
            left = self.a.sessions - self.sessions
            r = self.ask(inp, prev)
            prev, inp = r["id"], []
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
                inp.append({"type": "function_call_output", "call_id": c["call_id"], "output": out})
                if img:
                    inp.append({"role": "user", "content": [{"type": "input_text", "text": f"Head-camera frames from {c['name']}:"},
                                                            {"type": "input_image", "image_url": "data:image/jpeg;base64," + img}]})
        rep = ROOT / "results/agent" / f"REPORT_{self.ts}.md"
        rep.parent.mkdir(parents=True, exist_ok=True)
        rep.write_text(f"# Trial agent report {self.ts} ({self.a.model}, {self.sessions} robot sessions)\n\n{report}\n\n## Journal\n"
                       + "\n".join("- " + json.dumps(j) for j in self.journal) + f"\n\nArtifacts: {self.dir.relative_to(ROOT)}\n")
        print("REPORT", rep, flush=True)


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
    assert os.environ.get("ROBOT_ASKPASS"), "set ROBOT_ASKPASS"
    Agent(ap.parse_args()).run()
