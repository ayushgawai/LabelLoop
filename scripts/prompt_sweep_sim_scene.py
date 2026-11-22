"""X2Real scene telemetry and one-time IQBAR displacement for the prompt sweep."""
import json
import math
import os
from pathlib import Path
from time import sleep

import torch
from manaenv.assets.objects.labelloop_qr_sort import _Rigid, register_asset
from manaenv.assets.resolver import resolve_runtime_asset_reference

WORK = Path(os.environ["LABELLOOP_SWEEP_WORK"])
ZONE = (0.326, -0.490)
ZONE_INSIDE = 0.131
SURFACE_Z = -0.232
BOX_HALF = (0.025, 0.100, 0.050)
ARMS = ("left", "right")
for name in ("iqbar", "mat", "tray"):
    asset_name = "sol_prompt_sweep_" + name
    path = resolve_runtime_asset_reference(f"assets/labelloop/sol_prompt_sweep/{name}.usda")
    register_asset(type(asset_name, (_Rigid,), {"name": asset_name, "usd_path": path, "tags": ["object"]}))


def register_assets():
    """The scene factory calls this hook after importing the custom module."""


def write_json(path, payload):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    for _ in range(3):
        try:
            temp.replace(path)
            return
        except PermissionError:
            sleep(0.005)
    # A reader can temporarily pin the destination on Windows. The next
    # sample retries with the accumulated window; never abort the episode.



def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, PermissionError):
        return {}


def root_pose(env, name):
    data = env.scene[name].data
    return torch.cat([data.root_pos_w.torch, data.root_quat_w.torch], dim=-1).clone()


def move(env, name, pose):
    ids = torch.arange(env.num_envs, device=env.device)
    body = env.scene[name]
    body.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
    body.write_root_velocity_to_sim_index(root_velocity=torch.zeros((env.num_envs, 6), device=env.device), env_ids=ids)


def sample(env, call):
    data = env.scene["iqbar"].data
    p = (data.root_pos_w.torch - env.scene.env_origins)[0].tolist()
    q = data.root_quat_w.torch[0]
    x, y, z, w = q.tolist()
    axes = (
        (1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)),
        (2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)),
        (2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)),
    )
    extent = [sum(abs(a) * h for a, h in zip(row, BOX_HALF)) for row in axes]
    robot = env.scene["robot"]
    # Gripper-base link local +X points toward the jaws. Use a conservative
    # near-box threshold and require observed box motion before scoring a hold.
    if not hasattr(env, "_sol_sweep_grippers"):
        env._sol_sweep_grippers = [
            (robot.find_bodies(f"{arm}_arm_gripper_base_link")[0][0],
             robot.find_joints(f"{arm}_arm_gripper")[0][0]) for arm in ARMS
        ]
    tips = []
    for i, joint in env._sol_sweep_grippers:
        bp = (robot.data.body_link_pos_w.torch[0, i] - env.scene.env_origins[0]).tolist()
        b, c, d, a = robot.data.body_link_quat_w.torch[0, i].tolist()
        forward = (1 - 2 * (c*c + d*d), 2 * (b*c + a*d), 2 * (b*d - a*c))
        tip = [v + 0.205 * axis for v, axis in zip(bp, forward)]
        gap = math.sqrt(sum(max(0.0, abs(v - centre) - e)**2 for v, centre, e in zip(tip, p, extent)))
        lateral = (2 * (b*c - a*d), 1 - 2 * (b*b + d*d), 2 * (c*d + a*b))
        grip = float(robot.data.joint_pos.torch[0, joint]) * 1.526 / 1.89
        tips.append({"tip": tip, "axis": forward, "jaw_axis": lateral, "gap": gap, "opening": grip})
    bottom = p[2] - extent[2]
    speed = float(data.root_lin_vel_w.torch[0].norm())
    return {"call": call, "position": p, "extent": extent, "bottom": bottom, "speed": speed, "grippers": tips}


def judge(env):
    """Publish measured box state at 5 Hz and stop after the policy commits its score."""
    step = int(env.episode_length_buf[0])
    zero = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if step % 6 or step == 0:
        return zero
    active = read_json(WORK / "active.json")
    if not active:
        return zero
    trial, call = active["trial"], active["call"]
    record = getattr(env, "_sol_sweep_record", None)
    if not record or trial != record["trial"] or step < record["step"]:
        record = {"trial": trial, "step": step, "teleport": None, "rest": 0, "samples": []}
        env._sol_sweep_record = record
        record["approach_gap"] = 0.065
        record["shift_xy"] = [0.0, -0.12]
        record["closing_gap"] = 0.075
        record["minimum_call"] = 2
        record["approaching_call"] = 8
        if active["perturb"] and trial >= 15:
            extension = read_json(WORK / "extension.json").get("trials", [])
            if trial - 15 < len(extension):
                spec = extension[trial - 15]
                record["approach_gap"] = spec.get("approach_gap_m", 0.065)
                record["shift_xy"] = spec.get("shift_xy_m", [0.0, -0.12])
                record["closing_gap"] = spec.get("closing_gap_m", 0.075)
                record["minimum_call"] = spec.get("minimum_call", 2)
                record["approaching_call"] = spec.get("approaching_call", 8)
        if active["tray"]:
            pose = root_pose(env, "tray")
            pose[0, :3] = env.scene.env_origins[0] + torch.tensor([*ZONE, SURFACE_Z + 0.001], device=env.device)
            move(env, "tray", pose)
    record["step"] = step
    row = sample(env, call)
    # Trigger once on an open-claw approach near the stationary box. Hold off
    # while the first model chunk is only opening near the home pose.
    open_near = any(t["gap"] < record["approach_gap"] and t["opening"] >= 0.40 for t in row["grippers"])
    previous = record["samples"][-1] if record["samples"] else None
    approaching = (trial >= 15 and call >= record["approaching_call"] and previous
                   and any(g["gap"] < record["closing_gap"] and old["gap"] - g["gap"] > 0.003
                           for g, old in zip(row["grippers"], previous["grippers"])))
    if (active["perturb"] and record["teleport"] is None and call >= record["minimum_call"]
            and row["bottom"] <= SURFACE_Z + 0.009 and row["speed"] < 0.12
            and (open_near or approaching)):
        pose = root_pose(env, "iqbar")
        before = row["position"]
        dx, dy = record["shift_xy"]
        pose[0, :2] += torch.tensor([dx, dy], device=env.device)
        move(env, "iqbar", pose)
        record["teleport"] = {"call": call, "step": step, "before": before,
                              "after": [before[0] + dx, before[1] + dy, before[2]],
                              "shift_xy_m": [dx, dy], "approach_threshold_m": record["approach_gap"],
                              "trigger": "open_near" if open_near else "near_and_approaching",
                              "nearest_gripper_gap_m": min(t["gap"] for t in row["grippers"])}
        record["rest"] = 0
    px, py, _ = row["position"]
    ex, ey, _ = row["extent"]
    plane = SURFACE_Z + (0.006 if active["tray"] else 0.0)
    in_zone = (abs(px - ZONE[0]) + ex <= ZONE_INSIDE and abs(py - ZONE[1]) + ey <= ZONE_INSIDE
               and abs(row["bottom"] - plane) < 0.010 and row["speed"] < 0.035
               and (all(t["gap"] > 0.035 or t["opening"] > 0.9 for t in row["grippers"])))
    record["rest"] = record["rest"] + 1 if in_zone else 0
    row["step"] = step
    record["samples"].append(row)
    del record["samples"][:-48]
    write_json(WORK / "telemetry.json", record)
    if active.get("done"):
        return ~zero
    return zero
