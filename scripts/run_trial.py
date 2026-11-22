#!/usr/bin/env python3
"""One robust real-robot trial: home the arms, snapshot, relay + robot session, stop, home again, summary + montage.

  .venv/bin/python scripts/run_trial.py NAME [--seconds 240] [--relay "--stage --overlay none"] [--wording '{"PICK": "..."}']
                                        [--instruction TEXT (with --relay --passthrough)] [--home-only] [--snapshot-only]

Everything lands in runs/manual/<NAME>/ (relay.out, trial.jsonl, images/, montage.jpg, summary.json); one line is appended to
runs/manual/NOTES.md. The SSH password comes from the askpass helper (ROBOT_ASKPASS, default ~/.config/labelloop/robot_askpass.sh).
"""
import argparse, json, os, re, socket, subprocess, sys, threading, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from trial_agent import GOAL, summarize  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

PY = str(ROOT / ".venv/bin/python") if (ROOT / ".venv/bin/python").exists() else str(ROOT / ".venv/Scripts/python.exe")  # Windows venv
ASKPASS = os.environ.get("ROBOT_ASKPASS", str(Path.home() / ".config/labelloop/robot_askpass.sh"))
CLIENT = ("source /opt/xr/bot/x2robot_client/share/x2robot_client/scripts/session_env.sh >/dev/null 2>&1; timeout -s INT {sec} python3 "
          "/opt/xr/bot/x2robot_client/share/x2robot_client/scripts/start_desktop.py --model-address " + os.environ.get("LL_RELAY_ADDR", "192.168.1.157:8001") + " --instruction \\\"{instr}\\\" "
          "--robot-mode end_pose --infer-type sync --embodiment teletwin_desktop/gripper-g --image-wire-format base64 --attach{home}")


def free_port():
    for _ in range(20):
        try:
            socket.create_connection(("127.0.0.1", 8001), 0.3).close()
        except OSError:
            return
        if os.name == "nt":  # no pkill on Windows: stop the stale relay yourself
            raise SystemExit("port 8001 busy: stop the running relay first")
        subprocess.run(["pkill", "-f", "labelloop.robots.event_server"], capture_output=True)  # a stale relay holds the port
        time.sleep(0.5)
    raise SystemExit("port 8001 still busy")


def relay(flags, out_dir, log=None):
    free_port()
    args = [PY, "-m", "labelloop.robots.event_server", "--host", "0.0.0.0", "--port", "8001", "--profile", "real",
            "--save-images", str(out_dir / "images"), *flags] + (["--log", str(log)] if log else [])
    p = subprocess.Popen(args, cwd=ROOT, stdout=open(out_dir / "relay.out", "a"), stderr=subprocess.STDOUT)
    for _ in range(60):
        try:
            socket.create_connection(("127.0.0.1", 8001), 0.3).close()
            return p
        except OSError:
            if p.poll() is not None:
                raise SystemExit("relay exited: see %s" % (out_dir / "relay.out"))
            time.sleep(0.3)
    p.kill()
    raise SystemExit("relay did not start")


def robot(instr, sec, home=True, retries=2):
    cmd = CLIENT.format(sec=int(sec), instr=re.sub(r'["\\$`]', "", instr), home=" --home-on-stop" if home else "")
    env = {**os.environ, "SSH_ASKPASS": ASKPASS, "SSH_ASKPASS_REQUIRE": "force", "DISPLAY": ":0"}
    for k in range(retries + 1):
        r = subprocess.run(["ssh", "-o", "ConnectTimeout=8", "xr@192.168.1.21", 'docker exec desktop_client-robot_client-1 bash -c "%s"' % cmd],
                           env=env, capture_output=True, text=True, timeout=sec + 90)
        lines = [l for l in (r.stdout + r.stderr).splitlines() if re.search(r"START|STOP|done|rror", l) and not re.search(r"key|token|secret|passw", l, re.I)]
        if any("START" in l for l in lines):
            return lines[-6:]
        print("  robot session did not start (try %d): %s" % (k + 1, lines[-3:]), flush=True)
        time.sleep(3)
    raise SystemExit("robot session failed to start")


def seq_done(rows, need=2):
    """A sequence run is over once its last `need` rows are post-sequence holds (no instruction, source hold)."""
    tail = rows[-need:]
    return len(tail) == need and all(r.get("source") == "hold" and (r.get("harness") or {}).get("instruction") is None
                                     and ((r.get("harness") or {}).get("step") or 0) >= 1 for r in tail)


def stop_when_done(t0, stop):
    """Watch the trial this session records (runs/trials/<ts>/ newer than t0); when the sequence is done, end the robot session the same
    way the time limit does (one SIGINT to the client's python, not to `timeout`: a second Ctrl-C skips --home-on-stop)."""
    while not stop.wait(1.0):
        recs = [d for d in (ROOT / "runs/trials").glob("*/") if d.stat().st_mtime >= t0 and (d / "trial.jsonl").exists()]
        if not recs:
            continue
        try:
            rows = [json.loads(l) for l in (max(recs, key=lambda d: d.name) / "trial.jsonl").read_text().splitlines()[-3:] if l.strip()]
        except (OSError, ValueError):
            continue
        if seq_done(rows):
            env = {**os.environ, "SSH_ASKPASS": ASKPASS, "SSH_ASKPASS_REQUIRE": "force", "DISPLAY": ":0"}
            subprocess.run(["ssh", "-o", "ConnectTimeout=8", "xr@192.168.1.21",
                            "docker exec desktop_client-robot_client-1 pkill -INT -f '^python3 .*start_desktop.py'"], env=env, capture_output=True)
            print("sequence done: robot session stopped early (arms home)", flush=True)
            return


def session(out_dir, flags, instr, sec, log=None, home=True):
    p = relay(flags, out_dir, log)
    stop, w = threading.Event(), None
    if "--sequence" in flags:
        w = threading.Thread(target=stop_when_done, args=(time.time() - 1, stop), daemon=True)
        w.start()
    try:
        return robot(instr, sec, home)
    finally:
        stop.set()
        p.terminate()
        try:
            p.wait(10)
        except subprocess.TimeoutExpired:
            p.kill()


def home(out_dir):  # hold session (arms still), then --home-on-stop drives both arms to their rest pose
    return session(out_dir, ["--hold", "--no-record"], "hold still", 8)


def fronts(out_dir):
    return sorted((out_dir / "images").glob("*/*_front.jpg"), key=lambda f: (f.parent.name, int(f.name.split("_")[0])))


def montage(files, path, k=12):
    if not files:
        return
    files = [files[i] for i in np.linspace(0, len(files) - 1, min(k, len(files))).astype(int)]
    ims = []
    for f in files:
        im = cv2.resize(cv2.imread(str(f)), (480, 270))
        cv2.putText(im, f.parent.name[-6:] + "/" + f.name.split("_")[0], (8, 26), 0, 0.7, (0, 0, 255), 2)
        ims.append(im)
    while len(ims) % 3:
        ims.append(np.zeros_like(ims[0]))
    cv2.imwrite(str(path), np.vstack([np.hstack(ims[i:i + 3]) for i in range(0, len(ims), 3)]), [cv2.IMWRITE_JPEG_QUALITY, 70])


def demo_video(t0, path, fps=4):
    """demo.mp4 from every head frame of the trial recording that started after t0 (runs/trials/<ts>/frames), ~5x real time."""
    rec = [d for d in sorted((ROOT / "runs/trials").glob("*/")) if d.stat().st_mtime >= t0 and (d / "frames").exists()]
    for d in reversed(rec):
        frames = sorted((d / "frames").glob("*_front.jpg"), key=lambda f: int(f.name.split("_")[0]))
        if len(frames) > 5:
            first = cv2.imread(str(frames[0]))
            vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, first.shape[1::-1])
            for f in frames:
                vw.write(cv2.imread(str(f)))
            vw.release()
            return d
    return None


def main():
    a = argparse.ArgumentParser()
    a.add_argument("name", nargs="?", default="")
    a.add_argument("--scenario", default="", help="scenarios/<file>.json: frozen recipe (relay flags, wording, seconds, setup); runs go to runs/scenarios/<name>/run_N")
    a.add_argument("--seconds", type=int, default=240)
    a.add_argument("--relay", default="--stage", help="extra relay flags, e.g. '--stage --overlay mask --assume-code 420/90024/001'")
    a.add_argument("--wording", default="", help="JSON of instruction templates for this trial")
    a.add_argument("--instruction", default=GOAL, help="robot client instruction (the WALLX command itself with --passthrough)")
    a.add_argument("--home-only", action="store_true")
    a.add_argument("--snapshot-only", action="store_true")
    o = a.parse_args()
    sc = None
    if o.scenario:
        sc = json.loads(Path(o.scenario).read_text())
        o.relay, o.seconds = sc["relay"], sc.get("seconds", o.seconds)
        o.wording = json.dumps(sc["wording"]) if sc.get("wording") else ""
        o.instruction = sc.get("instruction", o.instruction)
        base = ROOT / "runs/scenarios" / sc["name"]
        o.name = "run_%d" % (len(list(base.glob("run_*"))) + 1)
        d = base / o.name
        print("scenario %s: %s\nsetup: %s" % (sc["name"], sc.get("goal", ""), sc.get("setup", "")), flush=True)
    else:
        d = ROOT / "runs/manual" / o.name
    d.mkdir(parents=True, exist_ok=True)
    quick = bool(sc and sc.get("quick"))  # recipe "quick": no separate homing sessions before / after (the sequence starts from
    if not quick:                         # rest and ends with its own home steps; the client homes on stop anyway)
        print("homing arms ...", home(d), flush=True)
    if o.home_only:
        return
    n0 = len(fronts(d))
    if o.snapshot_only:
        montage(fronts(d)[n0 - 1:], d / "snapshot.jpg", 1)
        return print("snapshot:", d / "snapshot.jpg")
    flags = o.relay.split()
    if o.wording:
        (d / "wording.json").write_text(json.dumps(json.loads(o.wording), indent=1))
        flags += ["--wording", str(d / "wording.json")]
    t0 = time.time()
    print("trial ...", session(d, flags, o.instruction, o.seconds, d / "trial.jsonl"), flush=True)
    files = fronts(d)[n0:]
    if not quick:
        print("homing arms ...", home(d), flush=True)
    s = summarize(d / "trial.jsonl")
    s.update(name=o.name, relay=o.relay, wording=o.wording, wall_s=round(time.time() - t0))
    (d / "summary.json").write_text(json.dumps(s, indent=1))
    montage(files, d / "montage.jpg")
    rec = demo_video(t0, d / "demo.mp4")
    s["recording"] = str(rec.relative_to(ROOT)) if rec else None
    (d / "summary.json").write_text(json.dumps(s, indent=1))
    if sc:
        res = ROOT / "scenarios/RESULTS.md"
        if not res.exists():
            res.write_text("# Scenario results (one line per run; recipe = scenarios/<name>.json, artifacts = runs/scenarios/<name>/run_N)\n\n")
        verdict = input("verdict for %s %s (ok / fail / partial + note, Enter = harness verdict %s): " % (sc["name"], o.name, s["verdict"])) if sys.stdin.isatty() else ""
        with open(res, "a") as f:
            f.write(f"- {time.strftime('%Y-%m-%d %H:%M')} | {sc['name']} {o.name} | harness {s['verdict']} in {s['seconds_to_verdict']} s | "
                    f"observed: {verdict or '-'} | states {s['states']} | flags {s['flags'][:3]}\n")
    with open(ROOT / "runs/manual/NOTES.md", "a") as f:
        f.write(f"- {time.strftime('%Y-%m-%d %H:%M')} | {o.name} | relay `{o.relay}` | wording {o.wording or 'default'} | verdict {s['verdict']} "
                f"in {s['seconds_to_verdict']} s | states {s['states']} | flags {s['flags'][:4]}\n")
    print(json.dumps({k: s[k] for k in ("verdict", "seconds_to_verdict", "states", "transitions", "flags", "codes_read", "instructions_sent")}, indent=1))
    print("montage:", d / "montage.jpg")


if __name__ == "__main__":
    main()
