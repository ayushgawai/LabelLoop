"""Policy Space bridge for this sweep. The only motion source is LAN WALLX."""
import json
import os
from pathlib import Path
import time

import cv2
from policy_space.base import PolicyModel

from labelloop.config import load_yaml
from labelloop.robots.x2real_server import CAMERAS, X2RealLink
from labelloop.wallx_client import WallxClient

WORK = Path(os.environ["LABELLOOP_SWEEP_WORK"])
RESULTS = WORK / "results"
STATUS = RESULTS / "STATUS.md"
MODEL = {"backend": "lan", "model": "wall-x", "version": "2610-dualhead-v2"}
SHORT = "Put the IQBAR box into zone 2."
STRATEGIES = {
    1: ("Short one-shot", [(SHORT, 20)]),
    2: ("Object description", [("Put the dark-brown IQBAR carton, about 20 by 10 by 5 centimeters, standing on its long edge near the center of the table, into zone 2.", 20)]),
    3: ("Detailed one-shot", [("Approach the upright IQBAR box in the middle of the table from above with the claw open. Place the jaws on either side of its thin width, across the broad faces, instead of gripping the two short ends. Lower the open jaws past the top edge before closing firmly around the box. Lift the box clear of the table, carry it to the taped square at the back right, lower it inside zone 2, and release it.", 20)]),
    4: ("Atomic sequence", [
        ("Open the claw and move it directly above the IQBAR box.", 3),
        ("Lower the open claw around the IQBAR box, with one jaw on each broad face.", 3),
        ("Close the claw around the sides of the IQBAR box.", 3),
        ("Lift the IQBAR box straight up without opening the claw.", 3),
        ("Keep holding the IQBAR box and carry it above the taped square at the back right.", 3),
        ("Lower the IQBAR box inside zone 2 and set it down.", 3),
        ("Open the claw to release the IQBAR box and lift the empty claw away.", 3),
    ]),
    5: ("Pick then place", [("Pick up the IQBAR box.", 8), (SHORT, 12)]),
    6: ("Named arm and grip", [("Use the right arm to grip the IQBAR box from above, with the jaws across the two broad sides. Lift it and put it into zone 2.", 20)]),
    7: ("State and negation", [("The IQBAR box is not in zone 2 yet. Pick up the box from the middle of the table, then put it down fully inside the taped square at the back right.", 20)]),
    8: ("Container grammar", [("Put the IQBAR box into the tray.", 20)]),
    9: ("Three fresh-history rounds", [(SHORT, 8)] * 3),
    10: ("Displacement with history reset", [(SHORT, 20)]),
}


def publish(path, payload):
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    for attempt in range(20):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.005)


def load(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, PermissionError):
        return {}


def ranking(row):
    score = row["scores"]
    telemetry = load(RESULTS / row["telemetry_path"])
    closest = min((g["gap"] for sample in telemetry if sample["call"] for g in sample["grippers"]),
                  default=1.0)
    return (bool(score["placed_inside_zone_2"] and score["lifted"]),
            bool(score["lifted"]), bool(score["grasped"]),
            bool(score["placed_inside_zone_2"]), bool(score["followed_literally"]),
            -(row["calls_to_success"] or 99), -closest)


class Sweep:
    def __init__(self):
        RESULTS.mkdir(parents=True, exist_ok=True)
        self.results = [load(path) for path in sorted(RESULTS.glob("trial_*.json"))]
        self.trial = len(self.results)
        if [r.get("trial") for r in self.results] != list(range(1, self.trial + 1)):
            raise RuntimeError("Trial checkpoints must be contiguous")
        settings = load(WORK / "extension.json")
        self.extension = settings.get("trials", [])
        self.repeats = settings.get("repeat_strategy_ids", [])
        self.done = True
        self.runlog = None
        if not STATUS.exists():
            STATUS.write_text("# WALLX simulation sweep\n\n", encoding="utf-8")

    def reset(self):
        if self.trial and not self.done:
            self.finish("Episode ended before the policy requested termination", valid=False)
        self.trial += 1
        if 11 <= self.trial <= 14 and not self.repeats:
            selected = sorted((r for r in self.results if r["trial"] <= 10 and r["valid"] and r["strategy_id"] != 10),
                              key=ranking, reverse=True)[:2]
            if len(selected) < 2:
                selected = [{"strategy_id": 1}, {"strategy_id": 8}]
            self.repeats = [r["strategy_id"] for r in selected] * 2
        if self.trial <= 14:
            self.strategy = self.trial if self.trial <= 10 else self.repeats[self.trial - 11]
            self.name, self.plan = STRATEGIES[self.strategy]
            self.tray = self.strategy == 8
            self.perturb = self.strategy == 10
            self.reset_after_teleport = self.perturb
        else:
            spec = self.extension[self.trial - 15]
            self.strategy = spec["strategy_id"]
            self.name = spec["strategy"]
            self.plan = [(spec["prompt"], spec["max_calls"])]
            self.tray = spec.get("tray", False)
            self.perturb = spec["perturb"]
            self.reset_after_teleport = spec["reset_after_teleport"]
        self.call = self.obs = self.stage = self.stage_calls = self.errors = 0
        self.phase_first = self.opened = self.lifted = self.placed = self.done = False
        self.arm_open = {arm: False for arm in ("left", "right")}
        self.arm_grips = {}
        self.peak_opening = {}
        self.hold_samples = {arm: 0 for arm in ("left", "right")}
        self.open_call = None
        self.attempts = {}
        self.grasp_call = self.lift_call = self.place_call = None
        self.grasp_arm = self.grip_axis = self.jaw_axis = self.teleport = None
        self.frames, self.calls, self.path_samples = [], [], []
        self.seen_steps = set()
        self.evidence = RESULTS / f"trial_{self.trial:02d}"
        self.evidence.mkdir(exist_ok=True)
        self.pending = True
        self.last_request = 0.0
        self.signal()
        print(f"sweep trial {self.trial:02d} / {self.name}", flush=True)

    def signal(self):
        publish(WORK / "active.json", {"trial": self.trial, "call": self.call,
                                       "tray": self.tray, "perturb": self.perturb,
                                       "done": self.done})

    def record_observation(self, images, state, telemetry):
        self.obs += 1
        frame = self.evidence / f"head_{self.obs:03d}.jpg"
        if not cv2.imwrite(str(frame), cv2.cvtColor(images["front"], cv2.COLOR_RGB2BGR)):
            raise OSError(f"Could not save {frame.name}")
        self.frames.append(frame.relative_to(RESULTS).as_posix())
        if telemetry.get("trial") != self.trial:
            return
        if telemetry.get("teleport") and self.teleport is None:
            self.teleport = telemetry["teleport"]
            self.phase_first = self.reset_after_teleport
        plane = -0.232 + (0.006 if self.tray else 0)
        for sample in telemetry.get("samples", []):
            if sample["step"] in self.seen_steps:
                continue
            self.seen_steps.add(sample["step"])
            self.path_samples.append(sample)
            call = sample["call"]
            for arm, jaw in zip(("left", "right"), sample["grippers"]):
                grip = jaw["opening"]
                initial = self.arm_grips.setdefault(arm, grip)
                self.peak_opening[arm] = max(grip, self.peak_opening.get(arm, grip))
                if call and grip >= max(0.40, initial + 0.10):
                    self.arm_open[arm] = self.opened = True
                    if self.open_call is None:
                        self.open_call = call
                closing_near_box = (self.arm_open[arm] and jaw["gap"] < 0.030 and grip < 1.22
                                    and self.peak_opening[arm] - grip >= 0.11)
                if closing_near_box and arm not in self.attempts:
                    self.attempts[arm] = {"call": call, "direction": jaw["axis"],
                                          "jaw_axis": jaw["jaw_axis"], "gap_m": jaw["gap"]}
                holding = closing_near_box and sample["bottom"] - plane >= 0.015
                self.hold_samples[arm] = self.hold_samples[arm] + 1 if holding else 0
                if self.hold_samples[arm] >= 2 and not self.lifted:
                    self.lift_call = call
                    self.grasp_call = self.attempts[arm]["call"]
                    self.grasp_arm = arm
                    self.grip_axis = jaw["axis"]
                    self.jaw_axis = jaw["jaw_axis"]
                    self.lifted = True
        if not self.placed and telemetry["rest"] >= 3:
            self.placed, self.place_call = True, self.call

    def step(self, images, state):
        if self.done:
            return None
        telemetry = load(WORK / "telemetry.json")
        self.record_observation(images, state, telemetry)
        if self.pending:
            # A hold chunk lets box and tray settle, and confirms this is our environment.
            if telemetry.get("trial") != self.trial:
                if self.obs >= 10:
                    self.finish("Simulator telemetry was unavailable", valid=False)
                return None
            self.pending = False
        if self.placed:
            self.finish("Box settled inside target", valid=self.errors < self.call)
            return None
        prompt, budget = self.plan[self.stage]
        if self.stage_calls >= budget or (self.strategy == 5 and self.stage == 0 and self.lifted):
            if self.stage + 1 == len(self.plan):
                self.finish("WALLX call budget reached", valid=self.errors < self.call)
                return None
            self.stage += 1
            self.stage_calls = 0
            self.phase_first = True
            prompt, _ = self.plan[self.stage]
        if self.errors >= 3 and self.errors == self.call:
            self.finish("No successful WALLX replies", valid=False)
            return None
        self.subgoal_changed = self.phase_first or self.call == 0
        self.phase_first = False
        self.call += 1
        self.stage_calls += 1
        self.signal()
        time.sleep(max(0.0, 0.65 - (time.monotonic() - self.last_request)))
        self.last_request = time.monotonic()
        self.calls.append({"call": self.call, "step": self.stage + 1, "prompt": prompt,
                           "reset_history": self.subgoal_changed, "frame": self.frames[-1],
                           "measured_gripper": {a: state[f"{a}_ee"]["gripper"] for a in ("left", "right")},
                           "measured_position": {a: state[f"{a}_ee"]["position"] for a in ("left", "right")}})
        return prompt

    def log_reply(self, latency_s, error=None):
        row = self.calls[-1]
        row["latency_s"] = round(latency_s, 4)
        # The backend may include request data in its error text. Record type only.
        row["error_type"] = error.split(":")[0] if error else None
        self.errors += bool(error)
        with (self.evidence / "calls.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        if self.call == 1 or self.call % 8 == 0 or error:
            print(f"trial {self.trial:02d} call {self.call} latency {latency_s:.2f}s error {row['error_type']}", flush=True)

    def finish(self, note, valid):
        self.done = True
        grasped = self.grasp_call is not None
        success = self.lifted and self.placed
        top = self.grip_axis is not None and self.grip_axis[2] < -0.55
        sides = self.jaw_axis is not None and abs(self.jaw_axis[0]) > 0.65
        literal = success
        if self.strategy == 6:
            literal = success and self.grasp_arm == "right" and top and sides
        elif self.strategy == 3:
            literal = None if success and top and sides else False
        elif self.strategy == 4:
            literal = None if success else False
        elif self.strategy == 5:
            literal = success and self.calls[self.lift_call - 1]["step"] == 1
        elif self.perturb:
            literal = success and self.teleport is not None and self.lift_call > self.teleport["call"]
        notes = [note, "Grasp flag requires a measured lift with a nearby closing claw. Review frames for holds at table height and literal step order."]
        if self.perturb:
            history = "reset history" if self.reset_after_teleport else "continued history"
            notes.append("No approaching pass reached the teleport trigger." if not self.teleport else
                         f"Box relocated 12 cm during call {self.teleport['call']}, then WALLX {history} on the next request.")
        if self.strategy == 9:
            notes.append("A fresh history starts each round. The box and arms stay in their current positions between rounds.")
        row = {
            "trial": self.trial, "strategy_id": self.strategy, "strategy": self.name, "valid": valid,
            "repeat": 10 < self.trial <= 14, "model": MODEL,
            "tray_in_zone_2": self.tray, "perturbation_requested": self.perturb,
            "reset_after_teleport": self.reset_after_teleport,
            "prompts": [{"text": p, "max_calls": n} for p, n in self.plan],
            "calls": self.call, "wallx_error_calls": self.errors,
            "calls_to_success": self.place_call if success else None,
            "scores": {"claw_opened": self.opened, "grasped": grasped, "lifted": self.lifted,
                       "placed_inside_zone_2": self.placed, "followed_literally": literal},
            "open_call": self.open_call, "grasp_call": self.grasp_call, "lift_call": self.lift_call, "place_call": self.place_call,
            "grasp_arm": self.grasp_arm, "gripper_forward_at_lift": self.grip_axis,
            "jaw_axis_at_lift": self.jaw_axis, "close_approaches": self.attempts,
            "teleport": self.teleport, "notes": notes, "frame_paths": self.frames,
            "calls_path": f"{self.evidence.name}/calls.jsonl",
            "telemetry_path": f"{self.evidence.name}/telemetry.json",
        }
        publish(self.evidence / "telemetry.json", self.path_samples)
        publish(RESULTS / f"trial_{self.trial:02d}.json", row)
        with STATUS.open("a", encoding="utf-8") as f:
            f.write(f"- Trial {self.trial:02d} | {self.name} | {self.call} calls | "
                    f"open {'Y' if self.opened else 'N'} | grasp {'Y' if grasped else 'N'} | "
                    f"lift {'Y' if self.lifted else 'N'} | zone 2 {'Y' if self.placed else 'N'} | "
                    f"{'scored' if valid else 'invalid'}\n")
            f.flush()
            os.fsync(f.fileno())
        self.results.append(row)
        self.signal()
        print(f"sweep saved trial {self.trial:02d}", flush=True)


class Model(PolicyModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.sweep = Sweep()
        self.client = WallxClient("ws://127.0.0.1:33678", timeout=15, open_timeout=8)
        self.client.connect()
        handshake = {k: self.client.metadata.get(k) for k in ("model", "model_version")}
        if tuple(handshake.values()) != (MODEL["model"], MODEL["version"]):
            raise RuntimeError("LAN server model identity differs from the sweep configuration")
        print(f"sweep model {handshake}", flush=True)
        config = load_yaml(str(Path(os.environ["LABELLOOP_SWEEP_REPO"]) / "configs" / "ex001_6r.yaml"))
        self.link = X2RealLink(self.sweep, self.client, config)
        self.observation = None

    def initialize_episode(self, episode_info=None):
        self.observation = None
        self.link.reset(episode_info)

    def ingest_observation(self, obs):
        self.observation = obs

    def infer_actions(self):
        if self.observation is None:
            raise RuntimeError("No X2Real observation")
        if not self.sweep.trial or not hasattr(self.sweep, "call"):
            self.link.reset()
        return self.link.step(self.observation)

    def close(self):
        self.client.close()
