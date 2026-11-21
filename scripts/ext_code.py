#!/usr/bin/env python3
"""External coding loop: an outside model (OpusMax Claude / OpenAI / Meta) edits repo files, we run the tests, feed failures back.

  python3 scripts/ext_code.py TASK.txt FILE [FILE ...] [--read RFILE ...] [--provider anthropic|openai|meta] [--rounds 3] [--test "pytest args"]

FILEs may be edited (full contents are sent and returned); RFILEs are context only. docs/AGENT_CONTEXT.md is always sent.
Keys come from .env (never printed). The model answers with full replacement files:  <<<FILE path>>> ...content... <<<END>>>
"""
import argparse, json, re, subprocess, sys, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = {k: v.strip().strip('"').strip("'") for k, v in (l.strip().split("=", 1) for l in open(ROOT / ".env") if "=" in l and not l.lstrip().startswith("#"))}
SYSTEM = ("You are a senior Python engineer editing a robotics repo. Make the minimal correct change. Return ONLY the files you change, each as\n"
          "<<<FILE relative/path.py>>>\n<the COMPLETE new file content>\n<<<END>>>\nNo diffs, no prose outside the blocks except one short summary line at the end. Do NOT use any tools (no web search): everything you need is in this message; answer directly.")


def call(provider, prompt):
    if provider == "anthropic":
        base = ENV["ANTHROPIC_OPUS_MAX_BASE_URL"].rstrip("/"); url = base + ("/messages" if base.endswith("/v1") else "/v1/messages")
        body = {"model": ENV.get("EXT_ANTHROPIC_MODEL", "claude-sonnet-5-5"), "max_tokens": 32000, "thinking": {"type": "disabled"}, "system": SYSTEM, "messages": [{"role": "user", "content": prompt}]}
        body["stream"] = True  # streamed: a long full-file answer otherwise hits the proxy's 504 gateway timeout
        req = urllib.request.Request(url, json.dumps(body).encode(), {"x-api-key": ENV["ANTHROPIC_OPUS_MAX_API_KEY"], "anthropic-version": "2023-06-01",
                                                                      "content-type": "application/json", "accept": "text/event-stream"})
        out = []
        with urllib.request.urlopen(req, timeout=1800) as r:
            for line in r:
                line = line.decode("utf-8", "replace").strip()
                if line.startswith("data:") and '"text_delta"' in line:
                    out.append(json.loads(line[5:])["delta"].get("text", ""))
        return "".join(out)
    if provider == "openai":
        r = post("https://api.openai.com/v1/responses", {"Authorization": "Bearer " + ENV["OPENAI_API_KEY"]}, {"model": ENV.get("EXT_OPENAI_MODEL", "gpt-5.4-mini"), "instructions": SYSTEM, "input": prompt})
        return r.get("output_text") or "".join(c.get("text", "") for o in r.get("output", []) for c in o.get("content", []) if isinstance(c, dict))
    if provider == "meta":  # OpenAI-compatible chat endpoint from .env (AGENT_CHAT_URL / AGENT_MODEL / META_MODEL_API_KEY)
        r = post(ENV["AGENT_CHAT_URL"], {"Authorization": "Bearer " + ENV.get("META_MODEL_API_KEY") or ENV["AGENT_API_KEY"]},
                 {"model": ENV["AGENT_MODEL"], "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]})
        return r["choices"][0]["message"]["content"]
    raise SystemExit(f"unknown provider {provider}")


def post(url, headers, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {**headers, "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=1200) as r:
        return json.load(r)


def bundle(paths):
    return "\n\n".join(f"===== FILE: {p} =====\n{(ROOT / p).read_text()}" for p in paths if (ROOT / p).exists())


def main():
    a = argparse.ArgumentParser(); a.add_argument("task"); a.add_argument("files", nargs="+"); a.add_argument("--read", nargs="*", default=[])
    a.add_argument("--provider", default="anthropic"); a.add_argument("--rounds", type=int, default=3); a.add_argument("--test", default="-q -x")
    o = a.parse_args()
    for f in o.files + o.read:
        assert ".env" not in f and not f.startswith("instructions"), f
    task = Path(o.task).read_text(); feedback = ""
    for rnd in range(1, o.rounds + 1):
        prompt = (f"{bundle(['docs/AGENT_CONTEXT.md'])}\n\n## TASK\n{task}\n\n## EDITABLE FILES (current)\n{bundle(o.files)}\n\n"
                  f"## READ-ONLY CONTEXT\n{bundle(o.read)}\n{feedback}")
        out = call(o.provider, prompt)
        blocks = re.findall(r"<<<FILE (.+?)>>>\n(.*?)\n<<<END>>>", out, re.S)
        if not blocks:
            print(f"round {rnd}: no file blocks returned:\n{out[:800]}"); feedback = "\n\n## PREVIOUS ANSWER HAD NO <<<FILE>>> BLOCKS. Return full files in the required format."; continue
        for path, content in blocks:
            path = path.strip()
            if path not in o.files and not path.startswith("tests/"):
                print(f"  skipped non-editable {path}"); continue
            (ROOT / path).parent.mkdir(parents=True, exist_ok=True); (ROOT / path).write_text(content.rstrip("\n") + "\n"); print(f"  wrote {path}")
        t = subprocess.run([str(ROOT / ".venv/bin/python"), "-m", "pytest", *o.test.split()], cwd=ROOT, capture_output=True, text=True, timeout=1500)
        tail = (t.stdout + t.stderr).strip().splitlines()[-60:]
        print(f"round {rnd}: {tail[-1] if tail else '?'}")
        if t.returncode == 0:
            print(out.strip().splitlines()[-1][:300]); return 0
        feedback = "\n\n## TEST FAILURES AFTER YOUR LAST CHANGE (fix them; files above are the current state)\n" + "\n".join(tail)
    return 1


if __name__ == "__main__":
    sys.exit(main())
