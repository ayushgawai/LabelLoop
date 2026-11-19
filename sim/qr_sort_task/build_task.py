"""Emit LabelLoop scene/task/suite YAML into an X2Real checkout (one scene+DAG per case 1-6).

  python sim/qr_sort_task/build_task.py --x2real $X2REAL_ROOT

Writes <x2real>/manaenv/defination/{labelloop.yaml, labelloop/scene/caseN.yaml, labelloop/task/{stage_base,caseN_dag}.yaml,
x2real/task/labelloop_leaderboard.csv} and appends board `labelloop_tabletop` to x2real/task/boards.yaml (once).
Mirrors pick_by_attribute / classify_objects_color / image_puzzle YAML. NOT RUN against Isaac Sim.
"""
import argparse
from pathlib import Path

import yaml

MOD = "manaenv.assets.objects.labelloop_qr_sort"
SCENE = lambda n: f"labelloop_qr_sort_case{n}_scene"
ENV = lambda n: f"artixon_arm6a_G_labelloop_qr_sort_case{n}_id_env"
TIMEOUT_S = 90
SIDE, DOWN = [0.7071, 0.0, 0.0, 0.7071], [1.0, 0.0, 0.0, 0.0]  # label sideways / down (box quats XYZW, roll about x)
SURF_TO_ENV = (0.0, -0.27)  # tabletop-local xy -> env xy (read off classify_objects_color poses; VERIFY in Kit)
TABLE_Z = -0.235
BIN_X = -0.105  # centre of the fruit scene's basket x range
# cube region copied from fruits_to_basket scene (object_start_region): x 0.05..0.28, y -0.38..0.38, yaws +-0.1
# boxes: asset suffix; tilt: label orientation; late: boxes parked at start, put on the table by a human at t
CASES = {
    1: dict(boxes=["a001"]),
    2: dict(boxes=["a001", "b001"]),
    3: dict(boxes=["a001", "b001", "a002"]),
    4: dict(boxes=["a001", "b001", "a002"], tilt={"a001": SIDE, "b001": DOWN, "a002": SIDE}, bins_moved=True),
    5: dict(boxes=["a001", "b001"], late={"a002": 12.0}, move={"a001": (8.0, [0.0, 0.08, 0.0])}),
    6: dict(boxes=["a001", "b001"], decoys=[1, 2]),  # ponytail: no table mat (its footprint blocked box placement)
}
BG = {"id": "booth", "asset": {"type": "asset_id", "id": "green_booth_mesh"}, "asset_module": MOD,
      "pose": {"position": [-0.452, -0.271, -0.985], "orientation": [0.0, 0.0, -0.704, 0.709]},
      "required_surfaces": ["tabletop"]}


def obj(oid, asset, tags, **kw):
    return {"id": oid, "asset": {"type": "asset_id", "id": asset}, "semantic_tags": tags, **kw}


def scene(n, c):
    boxes, late, decoys = c["boxes"], c.get("late", {}), c.get("decoys", [])
    mass = [0.07, 0.09]
    objs = [obj(f"box_{b}", f"labelloop_box_{b}", ["pickable", "qr_box"], spawn_mass=mass, spawn_static_friction=1.0,
                spawn_dynamic_friction=1.0, **({"orientation": c["tilt"][b]} if b in c.get("tilt", {}) else {}))
            for b in boxes + list(late)]
    objs += [obj(f"decoy_{i}", f"labelloop_decoy_{i}", ["pickable", "decoy"], spawn_mass=mass, spawn_static_friction=1.0,
                 spawn_dynamic_friction=1.0) for i in decoys]
    objs += [obj("bin_blue", "labelloop_bin_blue", ["container", "bin_blue"], kinematic=True),
             obj("bin_orange", "labelloop_bin_orange", ["container", "bin_orange"], kinematic=True)]
    movable = [f"box_{b}" for b in boxes] + [f"decoy_{i}" for i in decoys]
    # bins sit where the fruit scene's basket does (x -0.14..-0.07, y +-0.08), split left/right of centre
    far = {"blue": [BIN_X, 0.26], "orange": [BIN_X, -0.26]} if c.get("bins_moved") else {"blue": [BIN_X, 0.15], "orange": [BIN_X, -0.15]}
    nodes = {o: {"object": o} for o in movable}
    for col, (x, y) in far.items():
        nodes[f"bin_{col}"] = {"object": f"bin_{col}", "placement": {"type": "surface", "surface": "tabletop",
                               "ranges": {"x": [x - 0.01, x + 0.01], "y": [y - 0.01, y + 0.01]}, "orientation": [0, 0, 0, 1]}}
    for i, b in enumerate(late):  # parked far below the table until the human event fires
        nodes[f"box_{b}"] = {"object": f"box_{b}", "placement": {"type": "pose", "poses": [
            {"id": "parked", "position": [0.5, 3.0 + i, -5.0], "orientation": [0, 0, 0, 1]}]}}
    return {
        "definition": {"scene_type": SCENE(n), "metadata": {"name": f"labelloop_qr_sort_case{n}", "source": "labelloop",
                       "semantic_goal": "Sort QR-labelled boxes into the bin named by the code (A=blue, B=orange)"}},
        "render_settings": {"gen_method": 0, "tags": "green_booth"},
        "background": BG, "objects": objs,
        "layout": {"surfaces": {"tabletop": {"type": "ref", "ref": "background.tabletop"}}, "nodes": nodes,
                   "groups": {"box_start_region": {"type": "region", "nodes": movable, "surface": "tabletop",
                              "ranges": {"x": [0.05, 0.28], "y": [-0.38, 0.38]}, "yaws": [-0.1, 0.1],
                              "sample_method": "stratified", "spacing_mode": "footprint_gap", "min_spacing": 0.03,
                              "jitter": 0.35, "max_attempts": 20000}}},
        "variants": {"materialization": [], "assignments": []},
    }


def dag(n, c):
    assign = {f"box_{b}": ("bin_blue" if b[0] == "a" else "bin_orange") for b in c["boxes"] + list(c.get("late", {}))}
    assign.update({f"decoy_{i}": None for i in c.get("decoys", [])})
    events = {}
    for b, t in c.get("late", {}).items():  # human adds a box: teleport it onto the table
        x, y = 0.2 + SURF_TO_ENV[0], 0.15 + SURF_TO_ENV[1]
        events[f"human_adds_{b}"] = {"_func": f"{MOD}.disturb", "mode": "interval", "interval_range_s": [t, t],
                                     "is_global_time": True, "object_name": f"box_{b}", "to_xyz": [round(x, 3), round(y, 3), TABLE_Z + 0.05]}
    for b, (t, delta) in c.get("move", {}).items():  # human nudges a box (repeats every tick: long interval)
        events[f"human_moves_{b}"] = {"_func": f"{MOD}.disturb", "mode": "interval", "interval_range_s": [t, 1e6],
                                      "is_global_time": True, "object_name": f"box_{b}", "delta_xyz": delta}
    sections = {"instruction_component": {"instruction": "Read each box's QR label and put the box in the bin the code names "
                "(ZONE-A blue, ZONE-B orange); put unlabeled boxes aside; then return the arms home.", "topic": "qr_sort"}}
    if events:
        sections["event_component"] = {"terms": events}
    tags = [{"mode": 0, "tags": SCENE(n)}]
    return {"graph_task_generator": {"definition": {
        "task_type": f"labelloop_qr_sort_case{n}",
        "layout": {"runtime_sample": {"id": "runtime", "targets": {
            "placements": [{"id": "box_start_region", "target": {"type": "group", "id": "box_start_region"}, "sample": {"type": "inherit"}},
                           *({"id": k, "target": {"type": "node", "id": k}, "sample": {"type": "inherit"}} for k in ("bin_blue", "bin_orange"))]}}},
        "component_sections": sections},
        "nodes": [
            {"id": "sort", "include": "labelloop/task/stage_base.yaml", "score": 8,
             "direct_task_generator": {"definition": {"task_type": f"labelloop_qr_sort_case{n}_sort"},
                "component_sections": {"scene_constraint_component": {"constraints": tags},
                    "termination_component": {"terms": {"success": {"_func": f"{MOD}.boxes_sorted", "assignments": assign,
                        "bins": ["bin_blue", "bin_orange"], "xy": 0.07, "z_min": -0.02, "z_max": 0.15,
                        "vel_max": 0.5, "consecutive_steps": 5}}}}}},
            {"id": "return", "include": "labelloop/task/stage_base.yaml", "score": 2,
             "direct_task_generator": {"definition": {"task_type": f"labelloop_qr_sort_case{n}_return"},
                "component_sections": {"scene_constraint_component": {"constraints": tags},
                    "termination_component": {"terms": {"success": {"_func": "articulation_joints_near_default",
                        "asset_cfg": "robot", "atol": 0.4, "consecutive_steps": 10}}},
                    "planner_skills_component": {"skills": [{"type": "home"}]}}}}],
        "edges": [{"src": "sort", "dst": "return"}]}}


STAGE_BASE = {"direct_task_generator": {"definition": {"gen_method": 0, "episode_length_s": 180.0, "num_episodes": 10,
    "max_steps_per_episode": 2700}, "component_sections": {
    "termination_component": {"terms": {"mdp.time_out": {"time_out": True}}},
    "event_component": {"terms": {"mdp.reset_scene_to_default": {"mode": "reset", "reset_joint_targets": True}}},
    "metric_component": {"terms": {"success_rate": {}}},
    "viewer_component": {"terms": {"look_at_position": {"lookat_position": [0.204651, -0.268246, -0.240423], "offset": [-1.0, 0.0, 1.5]}}}}}}


def dump(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc, sort_keys=False, width=120))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--x2real", required=True)
    a = ap.parse_args()
    d = Path(a.x2real) / "manaenv" / "defination"
    dump(d / "labelloop/task/stage_base.yaml", STAGE_BASE)
    policy = {"PolicyPlanner": {"policy_cfg": {"include": "policy/x2real_ps@v1.0.0.yaml"}}}
    for n, c in CASES.items():
        dump(d / f"labelloop/scene/case{n}.yaml", scene(n, c))
        dump(d / f"labelloop/task/case{n}_dag.yaml", dag(n, c))
    # x2real_eval reads the suite raw (no include expansion), so copy x2real.yaml and replace its four lists.
    suite = yaml.safe_load((d / "x2real.yaml").read_text())
    suite.update({"scene_orders": [{"include": f"labelloop/scene/case{n}.yaml"} for n in CASES],
             "task_orders": [{"include": f"labelloop/task/case{n}_dag.yaml"} for n in CASES],
             "assembly_orders": [{"assembly_generator": {
                 "name": ENV(n), "scene_generator_name": SCENE(n), "robot_generator_name": "EX001-6R-G",
                 "task_generator_name": f"labelloop_qr_sort_case{n}", "num_envs": 1,
                 "metadata": {"x2real": {"embodiment": "artixon_arm6a", "capabilities": ["benchmark"],
                                         "robot_generator_identity": "EX001-6R-G", "board": "labelloop_tabletop"}}}} for n in CASES],
             "rollout": [{"env_ref": ENV(n), "num_episodes": 40, "episode_length_s": TIMEOUT_S, "max_steps": TIMEOUT_S * 30,
                          "num_parallel_envs": 4, "show_evaluation_overlay": True, "planner": policy} for n in CASES]})
    dump(d / "labelloop.yaml", suite)
    rows = ["subcategory,task,x2real_package,n_id,id_envs,id_timeout_s"]
    rows += [f"labelloop,qr_sort_case{n},labelloop/qr_sort_case{n},1,{ENV(n)},{TIMEOUT_S}" for n in CASES]
    csv = d / "x2real/task/labelloop_leaderboard.csv"
    csv.write_text("\n".join(rows) + "\n")
    boards = d / "x2real/task/boards.yaml"
    if "labelloop_tabletop" not in boards.read_text():
        with boards.open("a") as f:
            f.write("""  labelloop_tabletop:
    robot: artixon_arm6a
    grippers:
    - G
    num_episodes: 40
    num_parallel_envs: 4
    leaderboard_csv: x2real/task/labelloop_leaderboard.csv
    suite: labelloop.yaml
    scoring: id_only
    capabilities:
    - benchmark
    case_count: 6
    save_path: ./outputs/x2real_eval/labelloop/benchmark
""")
    print(f"wrote 6 cases under {d}")


if __name__ == "__main__":
    main()
