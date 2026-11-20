"""Emit LabelLoop scene/task/suite YAML into an X2Real checkout (one scene+DAG per case 1-12).

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
# MEASURED (probe episode, env positions read from the sim): the tabletop-local frame is rotated 180 degrees about the table centre,
# env = (TABLE_C.x - local.x, TABLE_C.y - local.y); e.g. blue mat local (-0.08,-0.20) -> env (0.29,-0.07), orange (-0.08,+0.20) -> env (0.28,-0.47).
# Head image: env +x = FAR, env +y = image LEFT (image x ~ 650 at env y -0.27, ~1000 px/m near the mats). Robot root env (-0.322,-0.259).
TABLE_C = (0.2047, -0.2682)
to_env = lambda lx, ly: (round(TABLE_C[0] - lx, 3), round(TABLE_C[1] - ly, 3))
SURF_TO_ENV = (0.0, -0.27)  # WRONG map (kept only so case 5's teleport spot stays numerically identical: env (0.2, -0.12))
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
# lane sort (cases 7-12). Tabletop is 0.685 x 1.11 m; induction zone = where a human slides a parcel in.
MAT_XY = {"lane_blue": (-0.08, -0.20), "lane_orange": (-0.08, 0.20)}
BASKET_XY, BASKET_Z = (0.12, 0.68), -0.425  # tabletop-local xy (inner edge at y=0.555 = table edge, image-right); base z env (rim -0.275 = 4 cm below tabletop)
BIN_REJECT_XY = (0.125, 0.30)  # case 13/14 dark bin ON the table, tabletop-local = env (0.08, -0.57): image-right of the induction zone, left of the right gripper home pose
# induction zone, ENV frame (parcels are teleported with env coordinates): lands at head-image ~ x 0.35..0.65, y 0.55..0.80, mid-table in front of the arms
INDUCT_ENV = {"x": [-0.08, 0.06], "y": [-0.42, -0.12]}
# case 15-17: the lanes are the existing open bins (0.172 m) instead of flat mats; x -0.10 (not -0.08) keeps the orange bin >= 4 cm from the dark reject bin
LANE_BIN_XY = {"bin_blue": (-0.10, -0.20), "bin_orange": (-0.10, 0.20)}
lane_ab = lambda c: ("bin_blue", "bin_orange") if c.get("lane_bins") else ("lane_blue", "lane_orange")
def lane_target(c, b):
    a, bb = lane_ab(c)
    rej = "reject_bin" if c.get("reject_bin") else "reject_basket"
    return {"z95112": a, "z90210": bb}.get(b, rej)
LANE_INSTR = ("Read each parcel's QR label: push ZIP 950-951 parcels onto the blue mat, ZIP 900-902 onto the orange mat; "
              "put anything else in the basket on the right; return home.")
LANE_INSTR_BINS = ("Read each parcel's QR label: put ZIP 950-951 parcels into the blue bin, ZIP 900-902 into the orange bin; "
                   "put anything else in the dark bin on the right; return home.")
LANE_INSTR_BIN = LANE_INSTR.replace("the basket on the right", "the dark bin on the right")
LANE_CASES = {
    # induction zone starts EMPTY (harness takes its empty-zone reference): every parcel is parked, a human "slides it in" at t
    7: dict(boxes=[], late={"z95112": 6.0}), 8: dict(boxes=[], late={"z90210": 6.0}), 9: dict(boxes=[], late={"z10001": 6.0}),
    10: dict(boxes=[], late={"nolabel": 6.0}), 11: dict(boxes=[], late={"garbage": 6.0}),
    13: dict(boxes=[], late={"nolabel": 6.0}, reject_bin=True), 14: dict(boxes=[], late={"garbage": 6.0}, reject_bin=True),
    12: dict(boxes=[], late={"z95112": 6.0, "z90210": 60.0, "nolabel": 120.0}, timeout=240),
}
LANE_CASES.update({
    15: dict(boxes=[], late={"z95112": 6.0}, reject_bin=True, lane_bins=True), 16: dict(boxes=[], late={"z90210": 6.0}, reject_bin=True, lane_bins=True),
    17: dict(boxes=[], late={"z95112": 6.0, "z90210": 60.0, "z10001": 120.0, "nolabel": 180.0, "garbage": 240.0}, reject_bin=True, lane_bins=True, timeout=320),
})
CASES.update({n: dict(c, lane=True) for n, c in LANE_CASES.items()})
BG = {"id": "booth", "asset": {"type": "asset_id", "id": "green_booth_mesh"}, "asset_module": MOD,
      "pose": {"position": [-0.452, -0.271, -0.985], "orientation": [0.0, 0.0, -0.704, 0.709]},
      "required_surfaces": ["tabletop"]}


reject_node = lambda c: "reject_bin" if c.get("reject_bin") else "reject_basket"


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
    lane = c.get("lane")
    reject = "reject_bin" if c.get("reject_bin") else "reject_basket"
    la, lb = lane_ab(c)
    fixed = [(la, f"labelloop_{la}"), (lb, f"labelloop_{lb}"), (reject, f"labelloop_{reject}")] if lane \
        else [("bin_blue", "labelloop_bin_blue"), ("bin_orange", "labelloop_bin_orange")]
    objs += [obj(i, a, ["container"], kinematic=True) for i, a in fixed]
    movable = [f"box_{b}" for b in boxes] + [f"decoy_{i}" for i in decoys]
    # bins sit where the fruit scene's basket does (x -0.14..-0.07, y +-0.08), split left/right of centre
    far = {"blue": [BIN_X, 0.26], "orange": [BIN_X, -0.26]} if c.get("bins_moved") else {"blue": [BIN_X, 0.15], "orange": [BIN_X, -0.15]}
    nodes = {o: {"object": o} for o in movable}
    if lane:
        far = {}
        for m, (x, y) in (LANE_BIN_XY if c.get("lane_bins") else MAT_XY).items():
            nodes[m] = {"object": m, "placement": {"type": "surface", "surface": "tabletop",
                        "ranges": {"x": [x - 0.01, x + 0.01], "y": [y - 0.01, y + 0.01]}, "orientation": [0, 0, 0, 1]}}
        if reject == "reject_bin":
            x, y = BIN_REJECT_XY
            nodes[reject] = {"object": reject, "placement": {"type": "surface", "surface": "tabletop",
                             "ranges": {"x": [x - 0.01, x + 0.01], "y": [y - 0.01, y + 0.01]}, "orientation": [0, 0, 0, 1]}}
        else:
            nodes[reject] = {"object": reject, "placement": {"type": "pose", "poses": [
                {"id": "floor", "position": [*to_env(*BASKET_XY), BASKET_Z], "orientation": [0, 0, 0, 1]}]}}
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
                   "groups": {} if not movable else {"box_start_region": {"type": "region", "nodes": movable, "surface": "tabletop",
                              "ranges": {"x": [0.05, 0.28], "y": [-0.38, 0.38]}, "yaws": [-0.1, 0.1],
                              "sample_method": "stratified", "spacing_mode": "footprint_gap", "min_spacing": 0.03,
                              "jitter": 0.35, "max_attempts": 20000}}},
        "variants": {"materialization": [], "assignments": []},
    }


def dag(n, c):
    lane = c.get("lane")
    assign = {f"box_{b}": (lane_target(c, b) if lane else "bin_blue" if b[0] == "a" else "bin_orange") for b in c["boxes"] + list(c.get("late", {}))}
    assign.update({f"decoy_{i}": None for i in c.get("decoys", [])})
    events = {}
    for b, t in c.get("late", {}).items():  # human adds a box: teleport it onto the table
        where = {"to_xyz": [0.2, -0.12, TABLE_Z + 0.05]} if not lane else {
            "to_xy_range": [INDUCT_ENV["x"], INDUCT_ENV["y"]],
            "yaw_range": [-0.3, 0.3], "z": TABLE_Z + 0.03, "avoid": [f"box_{o}" for o in list(c["boxes"]) + list(c["late"]) if o != b]}
        events[f"human_adds_{b}"] = {"_func": f"{MOD}.disturb", "mode": "interval", "interval_range_s": [t, t],
                                     "is_global_time": True, "object_name": f"box_{b}", **where}
    for b, (t, delta) in c.get("move", {}).items():  # human nudges a box (repeats every tick: long interval)
        events[f"human_moves_{b}"] = {"_func": f"{MOD}.disturb", "mode": "interval", "interval_range_s": [t, 1e6],
                                      "is_global_time": True, "object_name": f"box_{b}", "delta_xyz": delta}
    sections = {"instruction_component": {"instruction": (LANE_INSTR_BINS if c.get("lane_bins") else LANE_INSTR_BIN if c.get("reject_bin") else LANE_INSTR) if lane else "Read each box's QR label and put the box in the bin the code names "
                "(ZONE-A blue, ZONE-B orange); put unlabeled boxes aside; then return the arms home.", "topic": "qr_sort"}}
    if events:
        sections["event_component"] = {"terms": events}
    longer = {"episode_length_s": float(c["timeout"]), "max_steps_per_episode": c["timeout"] * 15} if c.get("timeout") else {}  # stage_base is 180 s
    tags = [{"mode": 0, "tags": SCENE(n)}]
    return {"graph_task_generator": {"definition": {
        "task_type": f"labelloop_qr_sort_case{n}",
        "layout": {"runtime_sample": {"id": "runtime", "targets": {
            "placements": [*([{"id": "box_start_region", "target": {"type": "group", "id": "box_start_region"}, "sample": {"type": "inherit"}}]
                             if c["boxes"] else []),
                           *({"id": k, "target": {"type": "node", "id": k}, "sample": {"type": "inherit"}} for k in ((*lane_ab(c), reject_node(c)) if lane else ("bin_blue", "bin_orange")))]}}},
        "component_sections": sections},
        "nodes": [
            {"id": "sort", "include": "labelloop/task/stage_base.yaml", "score": 8,
             "direct_task_generator": {"definition": {"task_type": f"labelloop_qr_sort_case{n}_sort", **longer},
                "component_sections": {"scene_constraint_component": {"constraints": tags},
                    "termination_component": {"terms": {"success": (
                        {"_func": f"{MOD}.lanes_sorted", "assignments": assign, "consecutive_steps": 5,
                                  **({"basket": "reject_bin", "basket_half": 0.07, "basket_z_max": 0.09} if c.get("reject_bin") else {}),
                                  **({"half": 0.07, "z_max": 0.09, "vel_max": 0.5} if c.get("lane_bins") else {})} if lane else
                        {"_func": f"{MOD}.boxes_sorted", "assignments": assign,
                        "bins": ["bin_blue", "bin_orange"], "xy": 0.07, "z_min": -0.02, "z_max": 0.15,
                        "vel_max": 0.5, "consecutive_steps": 5})}}}}},
            {"id": "return", "include": "labelloop/task/stage_base.yaml", "score": 2,
             "direct_task_generator": {"definition": {"task_type": f"labelloop_qr_sort_case{n}_return", **longer},
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
             "rollout": [{"env_ref": ENV(n), "num_episodes": 40, "episode_length_s": CASES[n].get("timeout", TIMEOUT_S), "max_steps": CASES[n].get("timeout", TIMEOUT_S) * 30,
                          "num_parallel_envs": 4, "show_evaluation_overlay": True, "planner": policy} for n in CASES]})
    dump(d / "labelloop.yaml", suite)
    rows = ["subcategory,task,x2real_package,n_id,id_envs,id_timeout_s"]
    rows += [f"labelloop,qr_sort_case{n},labelloop/qr_sort_case{n},1,{ENV(n)},{c.get('timeout', TIMEOUT_S)}" for n, c in CASES.items()]
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
    case_count: 17
    save_path: ./outputs/x2real_eval/labelloop/benchmark
""")
    print(f"wrote {len(CASES)} cases under {d}")


if __name__ == "__main__":
    main()
