"""Event-day relay: robot client (WALLX public_v2) -> harness picks instruction -> real WALLX. Replies pass through."""
import argparse
import asyncio
import base64
import json
import time
from pathlib import Path

import numpy as np
from websockets.asyncio.server import serve

from labelloop.wallx_client import (WallxClient, WallxError, decode_jpeg_b64, hold_response, pack, unpack)

MAX_FRAME_BYTES = 32 * 1024 * 1024
EMBODIMENTS = ("quanta_x1_pro", "quanta_x1_pro/gripper-g", "quanta_x1_pro/gripper-h", "teletwin_desktop", "teletwin_desktop/gripper-g", "teletwin_desktop/gripper-h")  # teletwin = the event robots


def validate_request(request):  # organizer rules
    if not isinstance(request, dict) or request.get("normalization") not in (None, "provided"):  # the event client omits it (deprecated)
        raise ValueError(f"Expected a WALLX observation; got keys {sorted(request) if isinstance(request, dict) else type(request)}")
    if request.get("embodiment") not in EMBODIMENTS:
        raise ValueError("Unsupported embodiment")
    if any(k in request for k in ("use_rtc", "prev_action_chunk", "inference_delay")):
        raise ValueError("This harness uses synchronous requests without RTC")
    if not (isinstance(request.get("instruction"), str) and request["instruction"].strip()):
        raise ValueError("Expected a non-empty string instruction")
    obs = request["observation"]
    for name in ("front", "left_wrist", "right_wrist"):
        if not isinstance(obs["images"][name], str) or not obs["images"][name]:
            raise ValueError("Three JPEG Base64 images required")
    for arm in ("left_ee", "right_ee"):
        s = obs["state"][arm]
        for field in ("position", "rotation_rpy"):
            a = np.asarray(s[field], dtype=float)
            if a.shape != (3,) or not np.isfinite(a).all():
                raise ValueError("Finite 3D position/RPY required")
        g = np.asarray(s["gripper"], dtype=float)
        if g.shape != () or not np.isfinite(g):
            raise ValueError("Finite scalar gripper required")


def save_request(root, ep, n, req):
    """Raw JPEG bytes of the three cameras + a sidecar JSON (state, instruction; never keys/headers) -> root/ep/<n>_<camera>.jpg / <n>.json."""
    d = Path(root) / ep
    d.mkdir(parents=True, exist_ok=True)
    for cam, b64 in req["observation"]["images"].items():
        (d / f"{n}_{cam}.jpg").write_bytes(base64.b64decode(b64))
    (d / f"{n}.json").write_text(json.dumps({"instruction": req["instruction"], "state": _plain_state(req["observation"]["state"])}))


def _plain_state(state):
    return {arm: {"position": np.asarray(state[arm]["position"], float).tolist(),
                  "rotation_rpy": np.asarray(state[arm]["rotation_rpy"], float).tolist(),
                  "gripper": float(state[arm]["gripper"])} for arm in ("left_ee", "right_ee")}


class Relay:
    """make_harness() -> Harness per downstream connection, or None for passthrough."""

    def __init__(self, endpoint, headers, make_harness=None, make_client=None, save_dir=None, hold=False):
        self.endpoint, self.headers, self.make_harness, self.make_client, self.save_dir, self.hold = endpoint, headers, make_harness, make_client, save_dir, hold

    async def handle_hold(self, downstream, cap=30):
        """Camera-capture mode: every request gets a hold reply (arms stay put), no WALLX, no harness; the first `cap` requests are saved."""
        await downstream.send(pack({"model": "hold"}))
        ep, k = time.strftime("%Y%m%d_%H%M%S"), 0
        async for frame in downstream:
            req = unpack(frame)
            validate_request(req)
            if self.save_dir and k < cap:
                save_request(self.save_dir, ep, k, req)
            k += 1
            await downstream.send(pack(hold_response(req["embodiment"], _plain_state(req["observation"]["state"]))))

    async def handle(self, downstream):
        if self.hold:
            return await self.handle_hold(downstream)
        return await self.handle_wallx(downstream)

    async def handle_wallx(self, downstream):
        wallx = self.make_client() if self.make_client else WallxClient(self.endpoint, self.headers)
        harness, goal, n, fresh, ep, k = None, None, 0, True, None, 0
        try:
            harness = self.make_harness() if self.make_harness else None
            await asyncio.to_thread(wallx.connect)
            await downstream.send(pack(wallx.metadata))
            async for frame in downstream:
                t0 = time.monotonic()
                req = unpack(frame)
                validate_request(req)
                if req["instruction"] != goal:
                    goal = req["instruction"]
                    ep, k = time.strftime("%Y%m%d_%H%M%S"), 0  # a new episode: image log restarts
                    print(json.dumps({"event": "goal", "goal": goal}), flush=True)
                    if harness:
                        harness.reset()
                if self.save_dir and k % 10 == 0:  # the 1st request of an episode, then every 10th (the instruction logged is the robot's)
                    save_request(self.save_dir, ep, k, req)
                k += 1
                state = _plain_state(req["observation"]["state"])
                instruction = req["instruction"]
                if harness:
                    images = {k: decode_jpeg_b64(v) for k, v in req["observation"]["images"].items()}
                    instruction = await asyncio.to_thread(harness.step, images, state)
                scripted = harness.scripted_reply(req["embodiment"], state) if harness else None
                if scripted is not None:
                    await downstream.send(pack(scripted))
                elif instruction is None:  # harness says hold still: don't spend a WALLX call
                    await downstream.send(pack(hold_response(req["embodiment"], state)))
                else:
                    if harness and (fresh or harness.subgoal_changed):
                        req["reset_history"] = True
                    fresh = False if harness else fresh
                    req["instruction"] = instruction  # whole original request forwarded, extras preserved
                    t1, err = time.monotonic(), None
                    try:
                        reply = await asyncio.to_thread(wallx.infer, req)
                    except WallxError as e:
                        if e.reply is None:  # session closed by WALLX: nothing to pass through
                            raise
                        reply, err = e.reply, str(e)  # error replies also pass through unchanged
                    if harness:
                        harness.log_reply(time.monotonic() - t1, err)
                    await downstream.send(pack(reply))
                print(json.dumps({"event": "chunk", "index": n, "instruction": instruction,
                                  "elapsed_ms": round((time.monotonic() - t0) * 1000, 1)}), flush=True)
                n += 1
        except Exception as e:  # never print keys/images
            print(f"Relay failed: {type(e).__name__}. Stop the session in WebUI.", flush=True)
            await downstream.close(code=1011, reason="Harness failure; stop in WebUI")
        finally:
            wallx.close()


def start(relay, host, port):
    return serve(relay.handle, host, port, ping_interval=None, compression=None, max_size=MAX_FRAME_BYTES)


async def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--passthrough", action="store_true", help="forward with the client's instruction, no harness")
    p.add_argument("--task", default="configs/task_qr_sort.yaml")
    p.add_argument("--log", default="runs/event.jsonl")
    p.add_argument("--profile", choices=("sim", "real"), default="real", help="camera / robot model (task yaml profiles)")
    p.add_argument("--real-safe", action=argparse.BooleanOptionalAction, default=True, help="hold + scripted HOME only; rejects / unread / big parcels latch HOLD")
    p.add_argument("--fast", action=argparse.BooleanOptionalAction, default=True, help="event_fast (with --profile real): WALLX-only, no scripted motion, retries skip HOME, real-time budget")
    p.add_argument("--hold", action="store_true", help="camera-capture mode: hold reply to every request, no WALLX / harness, save the first 30 requests' images")
    p.add_argument("--save-images", default="runs/event_images", help="dir for the 1st + every 10th request's camera JPEGs and state ('' = off)")
    a = p.parse_args()
    from labelloop.config import load_env, load_yaml
    load_env()
    make_harness = None
    if not a.passthrough and not a.hold:
        from labelloop.brain.agent import VisionAgent
        from labelloop.eval.runlog import RunLog
        from labelloop.harness import Harness
        task, agent, runlog = {**load_yaml(a.task), "profile": a.profile, "real_safe": a.real_safe, "event_fast": a.fast and a.profile == "real"}, VisionAgent.from_env(), RunLog(a.log)
        make_harness = lambda: Harness(task, agent, runlog)  # noqa: E731
    async with start(Relay(None, None, make_harness, WallxClient.from_env, a.save_images or None, a.hold), a.host, a.port):  # backend from configs/wallx.yaml
        print(f"Relay listening at ws://{a.host}:{a.port}", flush=True)
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Relay exited. Confirm the robot session is stopped in WebUI.")
