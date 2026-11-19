"""Run one LabelLoop QR-sort case in X2Real (Linux box, X2Real conda env active; policy server already up)."""
import argparse
import os
import shlex
import subprocess
from pathlib import Path

BOARD = "labelloop_tabletop"  # see sim/README.md step 3
SAVE = "outputs/x2real_eval/labelloop/benchmark"  # board save_path in boards.yaml


def env_name(case):
    return f"artixon_arm6a_G_labelloop_qr_sort_case{case}_id_env"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", type=int, required=True, choices=range(1, 7))
    ap.add_argument("--runs", type=int, required=True, help="episodes")
    ap.add_argument("--viz", choices=["none", "kit"], default="none")
    ap.add_argument("--server-port", type=int, default=8001)
    ap.add_argument("--x2real", default=os.environ.get("X2REAL_ROOT"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not a.x2real:
        ap.error("set --x2real or X2REAL_ROOT")
    env = env_name(a.case)
    cmd = ["python", "manaenv/scripts/x2real_eval.py", "--board", BOARD, "--env", env,
           "--num-episodes", str(a.runs), "--num-parallel-envs", "1",
           "--policy-config", "policy/x2real_ps@v1.0.0.yaml",
           "--server-address", "127.0.0.1", "--server-port", str(a.server_port),
           "--enable_cameras", "--viz", a.viz]
    print(f"cd {a.x2real} && {shlex.join(cmd)}")
    root = Path(a.x2real) / SAVE / env
    print(f"report: {root}/evaluation_report.jsonl (+ {Path(a.x2real) / SAVE}/benchmark_report.json)")
    print(f"videos: {root}/videos/combined/episode_*.mp4")
    if not a.dry_run:
        raise SystemExit(subprocess.run(cmd, cwd=a.x2real).returncode)


if __name__ == "__main__":
    main()
