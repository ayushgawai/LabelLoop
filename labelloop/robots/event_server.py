"""Event-day relay: robot client (WALLX public_v2) -> harness picks instruction -> real WALLX. Replies pass through."""
import argparse
import asyncio
import base64
import json
import time
from pathlib import Path

import cv2
import numpy as np
from websockets.asyncio.server import serve

from labelloop.perception import overlay
from labelloop.robots.trial_recorder import TrialRecorder, reduce_actions
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
    """make_harness() -> Harness per downstream connection, or None for passthrough. `recorder` (TrialRecorder or None) stores
    every trial: one line + the camera frames per request, a new trial per goal, ended when the connection ends."""

    def __init__(self, endpoint, headers, make_harness=None, make_client=None, save_dir=None, hold=False, recorder=None, record_meta=None, overlay_mode="none"):
        self.endpoint, self.headers, self.make_harness, self.make_client, self.save_dir, self.hold = endpoint, headers, make_harness, make_client, save_dir, hold
        self.recorder, self.record_meta = recorder, dict(record_meta or {})
        self.overlay_mode = overlay_mode
        self.lock_arm = ()  # arms whose WALLX actions are replaced by a hold (--lock-arm)

    def _overlay_front(self, harness, req, frame_index):
        mask_bins = getattr(harness, "mask_bins", None)
        if not harness or (self.overlay_mode == "none" and mask_bins is None):
            return req
        try:
            img = decode_jpeg_b64(req["observation"]["images"]["front"])
            parcel, target, bins, roi = harness.prompt_boxes(img.shape)
            if mask_bins is not None and mask_bins == bins.keys():
                target = None
            out = overlay.draw(img, self.overlay_mode, parcel=parcel, target=target, bins=bins, roi=roi, mask_bins=mask_bins)
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ok:
                req = dict(req)
                obs = dict(req["observation"])
                images = dict(obs["images"])
                images["front"] = base64.b64encode(buf.tobytes()).decode("ascii")
                obs["images"] = images
                req["observation"] = obs
                if self.recorder is not None:
                    try:
                        self.recorder.extra_frame(frame_index, "front_overlay", buf.tobytes())
                    except Exception:
                        pass
                return req
        except Exception:
            pass
        return req if mask_bins is None else None  # required stage mask failed: hold instead of forwarding the raw bins

    @staticmethod
    def _infer_retry(wallx, req, tries=2):
        """wallx.infer; when WALLX drops the connection (real event: ConnectionClosed mid-session), reconnect and resend instead of ending the session."""
        for k in range(tries + 1):
            try:
                return wallx.infer(req)
            except WallxError as e:
                if e.reply is not None or k == tries:
                    raise
            except Exception:  # websockets ConnectionClosed / OSError on the upstream socket
                if k == tries:
                    raise
            print(json.dumps({"event": "wallx_reconnect", "try": k + 1}), flush=True)
            try:
                wallx.close()
            except Exception:
                pass
            time.sleep(0.5)
            wallx.connect()
            req = {**req, "reset_history": True}

    async def handle_hold(self, downstream, cap=30):
        """Camera-capture mode: every request gets a hold reply (arms stay put), no WALLX, no harness; the first `cap` requests are saved."""
        await downstream.send(pack({"model": "hold"}))
        ep, k, i, goal = time.strftime("%Y%m%d_%H%M%S"), 0, 0, None
        try:
            async for frame in downstream:
                t0 = time.monotonic()
                req = unpack(frame)
                validate_request(req)
                if self.save_dir and k < cap:
                    save_request(self.save_dir, ep, k, req)
                k += 1
                state = _plain_state(req["observation"]["state"])
                reply = hold_response(req["embodiment"], state)
                if self.recorder is not None:  # hold sessions are recorded too (source "hold")
                    if req["instruction"] != goal:
                        goal, i = req["instruction"], 0
                        self.recorder.start(goal, **self.record_meta)
                    self.recorder.frame(i, req["observation"]["images"])
                    self.recorder.record(t=time.time(), index=i, goal=goal, subgoal=None, source="hold", harness=None,
                                         state=state, latency_ms=None, elapsed_ms=round((time.monotonic() - t0) * 1000, 1),
                                         error=None, actions=reduce_actions(reply))
                    i += 1
                await downstream.send(pack(reply))
        finally:
            if self.recorder is not None:
                self.recorder.end()

    async def handle(self, downstream):
        if self.hold:
            return await self.handle_hold(downstream)
        return await self.handle_wallx(downstream)

    async def handle_wallx(self, downstream):
        wallx = self.make_client() if self.make_client else WallxClient(self.endpoint, self.headers)
        harness, goal, n, fresh, ep, k = None, None, 0, True, None, 0
        trial, i = self.recorder is not None, 0  # i = row index within the current trial
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
                    if trial:  # a new goal starts a new trial (the previous one is ended by start())
                        i = 0
                        self.recorder.start(goal, **self.record_meta)
                if self.save_dir and k % 10 == 0:  # the 1st request of an episode, then every 10th (the instruction logged is the robot's)
                    save_request(self.save_dir, ep, k, req)
                k += 1
                state = _plain_state(req["observation"]["state"])
                instruction = req["instruction"]
                if harness:
                    images = {k: decode_jpeg_b64(v) for k, v in req["observation"]["images"].items()}
                    instruction = await asyncio.to_thread(harness.step, images, state)
                scripted = harness.scripted_reply(req["embodiment"], state) if harness else None
                forward = self._overlay_front(harness, req, i) if scripted is None and instruction is not None else req
                reply, source, latency_ms, err = None, None, None, None
                if scripted is not None:
                    reply, source = scripted, "scripted"
                    await downstream.send(pack(reply))
                elif instruction is None or forward is None:
                    if forward is None:
                        fresh, err = True, "stage_mask_failed"
                    reply, source = hold_response(req["embodiment"], state), "hold"
                    await downstream.send(pack(reply))
                else:
                    source = "wallx"
                    req = forward
                    if harness and (fresh or harness.subgoal_changed):
                        req["reset_history"] = True
                    fresh = False if harness else fresh
                    req["instruction"] = instruction  # whole original request forwarded, extras preserved
                    t1, err = time.monotonic(), None
                    try:
                        reply = await asyncio.to_thread(self._infer_retry, wallx, req)
                    except WallxError as e:
                        if e.reply is None:  # session closed by WALLX and the reconnect failed too: nothing to pass through
                            raise
                        reply, err = e.reply, str(e)  # error replies also pass through unchanged
                    latency_ms = (time.monotonic() - t1) * 1000
                    if harness:
                        harness.log_reply(latency_ms / 1000, err)
                    locked = set(self.lock_arm) | set(harness.lock_arms() if harness and hasattr(harness, "lock_arms") else ())
                    if locked and isinstance(reply, dict) and isinstance(reply.get("actions"), dict):  # hard rule: the idle arm holds still
                        held = hold_response(req["embodiment"], state)["actions"]
                        for arm in locked:
                            if arm + "_ee" in reply["actions"]:
                                reply["actions"][arm + "_ee"] = held[arm + "_ee"]
                    await downstream.send(pack(reply))
                elapsed_ms = (time.monotonic() - t0) * 1000
                if trial:
                    self.recorder.frame(i, req["observation"]["images"])
                    self.recorder.record(t=time.time(), index=i, goal=goal,
                                         subgoal=instruction if source == "wallx" else None, source=source,
                                         harness=getattr(harness, "last_log", None) if harness else None, state=state,
                                         latency_ms=None if latency_ms is None else round(latency_ms, 1),
                                         elapsed_ms=round(elapsed_ms, 1), error=err, actions=reduce_actions(reply))
                    i += 1
                print(json.dumps({"event": "chunk", "index": n, "instruction": instruction,
                                  "elapsed_ms": round(elapsed_ms, 1)}), flush=True)
                n += 1
        except Exception as e:  # never print keys/images
            print(f"Relay failed: {type(e).__name__}. Stop the session in WebUI.", flush=True)
            await downstream.close(code=1011, reason="Harness failure; stop in WebUI")
        finally:
            if self.recorder is not None:
                self.recorder.end()
            wallx.close()


def start(relay, host, port):
    return serve(relay.handle, host, port, ping_interval=None, compression=None, max_size=MAX_FRAME_BYTES)


async def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--wording", default="", help="JSON file overriding WALLX instruction templates, e.g. {\"PLACE\": \"put the small box at the {where} into the {mat} bin\"}")
    p.add_argument("--assume-after-search", action="store_true", help="DEMO: use --assume-code only after reading + turning the parcel (INSPECT) failed")
    p.add_argument("--lock-idle", action="store_true", help="harness locks the arm the parcel's lane does not use while WALLX acts")
    p.add_argument("--lock-arm", default="", help="comma list of arms (left,right) whose WALLX motion is replaced by holding still (hard rule: the idle arm cannot grab a bin)")
    p.add_argument("--cfg", default="", help="JSON merged into the task config (recipes), e.g. '{\"tries\": {\"STAND\": 2}}'")
    p.add_argument("--assume-code", default="", help="PRACTICE ONLY: treat the object on the table as carrying this code (manipulation trials without label reading)")
    p.add_argument("--stage", action=argparse.BooleanOptionalAction, default=False, help="event_fast staged delivery with per-stage bin masking (hard rule: WALLX cannot see the bins it must not grab)")
    p.add_argument("--passthrough", action="store_true", help="forward with the client's instruction, no harness")
    p.add_argument("--task", default="configs/task_qr_sort.yaml")
    p.add_argument("--log", default="runs/event.jsonl")
    p.add_argument("--profile", choices=("sim", "real"), default="real", help="camera / robot model (task yaml profiles)")
    p.add_argument("--real-safe", action=argparse.BooleanOptionalAction, default=True, help="hold + scripted HOME only; rejects / unread / big parcels latch HOLD")
    p.add_argument("--fast", action=argparse.BooleanOptionalAction, default=True, help="event_fast (with --profile real): WALLX-only, no scripted motion, retries skip HOME, real-time budget")
    p.add_argument("--hold", action="store_true", help="camera-capture mode: hold reply to every request, no WALLX / harness, save the first 30 requests' images")
    p.add_argument("--save-images", default="runs/event_images", help="dir for the 1st + every 10th request's camera JPEGs and state ('' = off)")
    p.add_argument("--record", default="runs/trials", help="dir for trial recordings, one folder per trial ('' = off; see labelloop.dashboard.server)")
    p.add_argument("--no-record", dest="record", action="store_const", const="", help="disable trial recording")
    p.add_argument("--overlay", choices=("none", "parcel", "mask", "grid"), default="none")
    a = p.parse_args()
    from labelloop.config import load_env, load_yaml
    load_env()
    make_harness = None
    if not a.passthrough and not a.hold:
        from labelloop.brain.agent import VisionAgent
        from labelloop.eval.runlog import RunLog
        from labelloop.harness import Harness
        task, agent, runlog = {**load_yaml(a.task), "profile": a.profile, "real_safe": a.real_safe, "event_fast": a.fast and a.profile == "real"}, VisionAgent.from_env(), RunLog(a.log)
        task["assume_code"] = a.assume_code or None  # practice only
        task["lock_idle_arm"] = a.lock_idle
        task["assume_after_search"] = a.assume_after_search
        for k, v in json.loads(a.cfg or "{}").items():  # recipe overrides, e.g. {"tries": {"STAND": 2}}
            task[k] = {**task[k], **v} if isinstance(v, dict) and isinstance(task.get(k), dict) else v
        task["event_fast_stage"] = a.stage  # STAND (wide) -> PICK (all bins hidden) -> PLACE (only the target bin visible)
        if a.wording:  # JSON {state: template or [templates]} overriding the WALLX instruction wording (scripts/trial_agent.py)
            task["wording"] = json.loads(Path(a.wording).read_text() or "{}")
        make_harness = lambda: Harness(task, agent, runlog)  # noqa: E731
    recorder = TrialRecorder(a.record) if a.record else None
    meta = {"profile": a.profile, "real_safe": a.real_safe, "fast": a.fast, "hold": a.hold, "passthrough": a.passthrough}
    relay = Relay(None, None, make_harness, WallxClient.from_env, a.save_images or None, a.hold, recorder, meta, a.overlay)  # backend from configs/wallx.yaml
    relay.lock_arm = tuple(x for x in a.lock_arm.split(",") if x in ("left", "right"))
    async with start(relay, a.host, a.port):
        print(f"Relay listening at ws://{a.host}:{a.port}", flush=True)
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Relay exited. Confirm the robot session is stopped in WebUI.")
