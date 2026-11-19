"""Event-day relay: robot client (WALLX public_v2) -> harness picks instruction -> real WALLX. Replies pass through."""
import argparse
import asyncio
import json
import time

import numpy as np
from websockets.asyncio.server import serve

from labelloop.wallx_client import (WallxClient, WallxError, decode_jpeg_b64, endpoint_headers_from_env,
                                    hold_response, pack, unpack)

MAX_FRAME_BYTES = 32 * 1024 * 1024
EMBODIMENTS = ("quanta_x1_pro", "quanta_x1_pro/gripper-g", "quanta_x1_pro/gripper-h")


def validate_request(request):  # organizer rules
    if not isinstance(request, dict) or request.get("normalization") != "provided":
        raise ValueError("Expected WALLX observation with normalization=provided")
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


def _plain_state(state):
    return {arm: {"position": np.asarray(state[arm]["position"], float).tolist(),
                  "rotation_rpy": np.asarray(state[arm]["rotation_rpy"], float).tolist(),
                  "gripper": float(state[arm]["gripper"])} for arm in ("left_ee", "right_ee")}


class Relay:
    """make_harness() -> Harness per downstream connection, or None for passthrough."""

    def __init__(self, endpoint, headers, make_harness=None):
        self.endpoint, self.headers, self.make_harness = endpoint, headers, make_harness

    async def handle(self, downstream):
        wallx, harness, goal, n, fresh = WallxClient(self.endpoint, self.headers), None, None, 0, True
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
                    print(json.dumps({"event": "goal", "goal": goal}), flush=True)
                    if harness:
                        harness.reset()
                state = _plain_state(req["observation"]["state"])
                instruction = req["instruction"]
                if harness:
                    images = {k: decode_jpeg_b64(v) for k, v in req["observation"]["images"].items()}
                    instruction = await asyncio.to_thread(harness.step, images, state)
                if instruction is None:  # harness says hold still: don't spend a WALLX call
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
    a = p.parse_args()
    from labelloop.config import load_env, load_yaml
    load_env()
    make_harness = None
    if not a.passthrough:
        from labelloop.brain.agent import VisionAgent
        from labelloop.eval.runlog import RunLog
        from labelloop.harness import Harness
        task, agent, runlog = load_yaml(a.task), VisionAgent.from_env(), RunLog(a.log)
        make_harness = lambda: Harness(task, agent, runlog)  # noqa: E731
    async with start(Relay(*endpoint_headers_from_env(), make_harness), a.host, a.port):
        print(f"Relay listening at ws://{a.host}:{a.port}", flush=True)
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Relay exited. Confirm the robot session is stopped in WebUI.")
