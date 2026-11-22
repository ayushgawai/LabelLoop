"""Build a new X2Real suite and assets for the scenario 5 prompt sweep."""
import argparse
import os
import math
from pathlib import Path
import shutil
import sys

import yaml

REPO = Path(os.environ.get("LABELLOOP_SWEEP_REPO", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO / "sim" / "qr_sort_task"))
import build_task as base
from make_usd import bin_body, box_mesh, cube, material, omni, srgb_to_linear, usda

SCENE = "sol_prompt_sweep_scene"
TASK = "sol_prompt_sweep_task"
ENV = "artixon_arm6a_G_sol_prompt_sweep_id_env"
MOD = "manaenv.assets.objects.prompt_sweep_sim_scene"
ZONES = ((0.326, -0.060), (0.326, -0.490), (0.009, -0.490))
ZONE_HALF = 0.137
START = (0.130, -0.295)
BOX_HALF = (0.025, 0.100, 0.050)
TAPE_WIDTH = 0.012
TRAY_SIZE = (0.260, 0.260, 0.025)
TRAY_FLOOR = 0.005


def placement(name, x, y, z):
    return {"object": name, "placement": {"type": "pose", "poses": [
        {"id": "fixed", "position": [x, y, z], "orientation": [0, 0, 0, 1]}
    ]}}


def tape_geometry():
    z = base.TABLE_MAT["size"][2] + 0.00025
    sides = []
    digits = (
        [(-0.030, -0.035), (0.012, -0.056), (0.012, 0.058), (-0.035, 0.058), (0.045, 0.058)],
        [(-0.044, -0.045), (-0.008, -0.057), (0.038, -0.034), (0.036, -0.020),
         (-0.043, 0.046), (-0.043, 0.058), (0.050, 0.058)],
        [(-0.044, -0.054), (0.040, -0.054), (0.047, -0.029), (0.011, -0.002), (0.047, 0.025),
         (0.037, 0.055), (-0.044, 0.055)],
    )
    for number, (x, y) in enumerate(ZONES, 1):
        x, y = x - base.TABLE_C[0], y - base.TABLE_C[1]
        segments = [(x + dx, y + dy, sx, sy) for dx, dy, sx, sy in [
            (-ZONE_HALF, 0, TAPE_WIDTH, 2 * ZONE_HALF + TAPE_WIDTH),
            (ZONE_HALF, 0, TAPE_WIDTH, 2 * ZONE_HALF + TAPE_WIDTH),
            (0, -ZONE_HALF, 2 * ZONE_HALF - TAPE_WIDTH, TAPE_WIDTH),
            (0, ZONE_HALF, 2 * ZONE_HALF - TAPE_WIDTH, TAPE_WIDTH),
        ]]
        for i, (xx, yy, sx, sy) in enumerate(segments):
            sides.append(cube(f"tape_{number}_{i}", (sx, sy, 0.0005), (xx, yy, z), "tape")
                         .replace('"PhysicsCollisionAPI", ', ""))
        for i, ((u0, v0), (u1, v1)) in enumerate(zip(digits[number-1], digits[number-1][1:])):
            dx, dy = v0 - v1, u0 - u1
            centre = (x - (v0 + v1) / 2, y - (u0 + u1) / 2, z)
            angle = math.degrees(math.atan2(dy, dx))
            strip = cube(f"digit_{number}_{i}", (math.hypot(dx, dy) + 0.005, 0.009, 0.0005), centre, "tape")
            strip = strip.replace('"PhysicsCollisionAPI", ', "")
            strip = strip.replace('uniform token[] xformOpOrder', f'double xformOp:rotateZ = {angle}\n        uniform token[] xformOpOrder')
            sides.append(strip.replace('"xformOp:translate", ', '"xformOp:translate", "xformOp:rotateZ", '))
    return "".join(sides)


def write_assets(path, texture_path):
    path.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(texture_path, path / "iqbar_texture.jpg")
    texture = "./iqbar_texture.jpg"
    face = (0, 1/7, 1, 5/7)
    top = (0, 5/7, 1, 1)
    side = (0, 0, 1, 1/7)
    uv = {"pz": top, "nz": side, "px": face, "nx": face, "py": side, "ny": side}
    (path / "iqbar.usda").write_text(usda("box", box_mesh("pic", BOX_HALF, uv),
                                         material("pic", omni(texture)), mass=0.35), encoding="utf-8")
    x, y, h = base.TABLE_MAT["size"]
    body = cube("mat", (x, y, h), (0, 0, h / 2), "mat") + tape_geometry()
    mat = material("mat", omni(color=srgb_to_linear(base.TABLE_MAT["color"])))
    tape = material("tape", omni(color=srgb_to_linear((237, 227, 195))))
    (path / "mat.usda").write_text(usda("mat", body, mat + tape, kinematic=True), encoding="utf-8")
    looks = material("out", omni(color=srgb_to_linear((146, 55, 38))))
    looks += material("in", omni(color=srgb_to_linear((185, 139, 97))))
    (path / "tray.usda").write_text(usda("tray", bin_body(TRAY_SIZE), looks, kinematic=True), encoding="utf-8")


def write_suite(definition, work, episodes):
    # Keep the installed catalog untouched. This is a separate, directly launched suite.
    suite = yaml.safe_load((definition / "labelloop.yaml").read_text(encoding="utf-8"))
    scene = base.scene(31, {"late": {}})
    scene["definition"] = {"scene_type": SCENE, "metadata": {"name": SCENE}}
    scene["background"]["asset_module"] = MOD
    scene["objects"] = [
        base.obj("table_mat", "sol_prompt_sweep_mat", [], kinematic=True)
        | {"spawn_static_friction": 1.0, "spawn_dynamic_friction": 0.8},
        base.obj("iqbar", "sol_prompt_sweep_iqbar", ["pickable"], spawn_mass=0.35,
                 spawn_static_friction=1.0, spawn_dynamic_friction=1.0),
        base.obj("tray", "sol_prompt_sweep_tray", ["container"], kinematic=True),
    ]
    scene["layout"]["nodes"] = {
        "table_mat": placement("table_mat", *base.TABLE_C, base.TABLE_Z),
        "iqbar": placement("iqbar", *START, base.SURFACE_Z + BOX_HALF[2] + 0.002),
        "tray": placement("tray", -3.0, -3.0, -2.5),
    }
    stage = base.STAGE_BASE["direct_task_generator"]
    components = stage["component_sections"]
    task = {"direct_task_generator": {
        "definition": stage["definition"] | {
            "task_type": TASK, "episode_length_s": 95.0, "max_steps_per_episode": 2850,
        },
        "component_sections": components | {
            "scene_constraint_component": {"constraints": [{"mode": 0, "tags": SCENE}]},
            "instruction_component": {"topic": "pick_and_place", "instruction": "Move the IQBAR box into zone 2."},
            "termination_component": {"terms": components["termination_component"]["terms"] | {
                "success": {"_func": f"{MOD}.judge"},
            }},
        },
    }}
    assembly = {
        "name": ENV, "scene_generator_name": SCENE, "robot_generator_name": "EX001-6R-G",
        "task_generator_name": TASK, "num_envs": 1,
        "metadata": {"x2real": {"embodiment": "artixon_arm6a", "capabilities": ["benchmark"],
                                "robot_generator_identity": "EX001-6R-G", "board": "labelloop_tabletop"}}
    }
    rollout = {
        "env_ref": ENV, "num_episodes": episodes, "num_parallel_envs": 1, "episode_length_s": 95, "max_steps": 2850,
        "show_evaluation_overlay": False,
        "planner": {"PolicyPlanner": {"policy_cfg": {"include": "policy/x2real_ps@v1.0.0.yaml"}}},
    }
    suite.update(scene_orders=[scene], task_orders=[task], assembly_orders=[{"assembly_generator": assembly}],
                 rollout=[rollout])
    suite["output"][0].update(save_path=(work / "sim_output").as_posix(), record_video=False)
    base.dump(definition / "sol_prompt_sweep.yaml", suite)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--definition", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--texture", type=Path, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    args = parser.parse_args()
    write_assets(args.assets / "labelloop" / "sol_prompt_sweep", args.texture)
    write_suite(args.definition, args.work, args.episodes)


if __name__ == "__main__":
    main()
