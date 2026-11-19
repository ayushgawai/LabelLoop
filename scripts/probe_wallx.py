#!/usr/bin/env python3
"""Level-2 WALLX link test: handshake, latency, response shape, bad-request handling.

  python scripts/probe_wallx.py [--n 3|50] [--images DIR] [--reset-history]
Reads WALLX_ENDPOINT / WALLX_API_KEY / WALLX_HEADERS_FILE from env or .env. Never prints keys.
"""
import argparse
import base64
import sys
from pathlib import Path
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from labelloop.wallx_client import WallxClient, WallxError, build_payload, encode_jpeg_b64, pack  # noqa: E402

CAMS = ("front", "left_wrist", "right_wrist")


def load_dotenv(path):
    if path.exists():
        import os
        for line in path.read_text().splitlines():
            k, _, v = line.partition("=")
            if k.strip() and not k.startswith("#") and v.strip():
                os.environ.setdefault(k.strip(), v.strip())


def check_shape(reply):
    for arm in ("left_ee", "right_ee"):
        a = reply["actions"][arm]
        assert np.asarray(a["position"]).shape == (32, 3), f"{arm}.position {np.asarray(a['position']).shape}"
        assert np.asarray(a["rotation_rpy"]).shape == (32, 3), f"{arm}.rotation_rpy"
        assert np.asarray(a["gripper"]).shape == (32,), f"{arm}.gripper"


def try_call(client, name, payload, results):
    try:
        client.infer(payload)
        out = "UNEXPECTED success (no error)"
    except WallxError as e:  # session_closed = WALLX closed 1011 on us; client already reconnected
        out = f"closed: {e.message}" if e.code == "session_closed" else f"error reply: {e.code}: {e.message}"
    except Exception as e:
        out = f"FAIL {type(e).__name__}: {e}"
        try:
            client.connect()
        except Exception:
            pass
    results.append((name, out))
    print(f"  {name}: {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--images", help="dir with front.jpg left_wrist.jpg right_wrist.jpg")
    ap.add_argument("--robot", default=str(ROOT / "configs/ex001_6r.yaml"))
    ap.add_argument("--jpeg-quality", type=int, default=90)
    ap.add_argument("--reset-history", action="store_true")
    a = ap.parse_args()
    load_dotenv(ROOT / ".env")
    robot = yaml.safe_load(open(a.robot))
    if a.images:
        imgs = {c: base64.b64encode((Path(a.images) / f"{c}.jpg").read_bytes()).decode() for c in CAMS}
    else:
        rng = np.random.default_rng(0)
        imgs = {c: encode_jpeg_b64(rng.integers(0, 255, (480, 640, 3), dtype=np.uint8), a.jpeg_quality) for c in CAMS}
    state = {arm: {**robot["home_pose"][arm], "gripper": 0.1} for arm in ("left_ee", "right_ee")}
    emb = robot["embodiment"]
    good = lambda: build_payload(emb, "pick up the box", dict(imgs), state)  # noqa: E731

    client = WallxClient.from_env()
    client.connect()
    print("handshake:", client.metadata)
    lat = []
    for i in range(a.n):
        p, kb = good(), len(pack(good())) / 1024
        t = time.monotonic()
        reply = client.infer(p, timeout=120)
        lat.append(time.monotonic() - t)
        check_shape(reply)
        print(f"call {i}: send->recv {lat[-1] * 1000:.0f} ms, request {kb:.0f} KB, shape OK")
    if len(lat) > 1:
        print(f"p50 {np.percentile(lat, 50) * 1000:.0f} ms  p95 {np.percentile(lat, 95) * 1000:.0f} ms "
              f"(first call includes warm-up)")

    print("bad requests:")
    results = []
    p = good(); p["embodiment"] = "not_a_robot"
    try_call(client, "bad embodiment", p, results)
    p = good(); del p["observation"]["images"]["right_wrist"]
    try_call(client, "missing camera", p, results)
    p = good(); p["observation"]["images"]["front"] = base64.b64encode(b"not a jpeg").decode()
    try_call(client, "non-JPEG image", p, results)
    try:
        check_shape(client.infer(good(), timeout=120))
        results.append(("good request after bad ones", "OK"))
    except Exception as e:
        results.append(("good request after bad ones", f"FAIL {type(e).__name__}: {e}"))

    if a.reset_history:
        p = good(); p["reset_history"] = True
        try:
            check_shape(client.infer(p))
            print("reset_history: ACCEPTED (valid reply)")
        except WallxError as e:
            print(f"reset_history: REJECTED {e.code}: {e.message}")
        except Exception as e:
            print(f"reset_history: FAIL {type(e).__name__}: {e}")
    client.close()
    print("summary:")
    for name, out in results:
        print(f"  {name}: {out}")


if __name__ == "__main__":
    main()
