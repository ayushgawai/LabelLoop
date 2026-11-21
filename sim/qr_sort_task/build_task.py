"""Emit LabelLoop scene/task/suite YAML into an X2Real checkout: the lane-sort cases 18-30.

  python sim/qr_sort_task/build_task.py --x2real $X2REAL_ROOT

Writes <x2real>/manaenv/defination/{labelloop.yaml, labelloop/scene/caseN.yaml, labelloop/task/{stage_base,caseN_dag}.yaml,
x2real/task/labelloop_leaderboard.csv}, deletes caseN files of cases no longer in CASES, and appends board `labelloop_tabletop`
to x2real/task/boards.yaml (once; later runs only refresh its case_count). Mirrors pick_by_attribute / image_puzzle YAML.
Cases 1-17 (earlier layouts) were removed on 2026-11-20; 18-23 keep their numbers so env names and past results still match.
"""
import argparse
import math
import re
from pathlib import Path

import yaml

from props import BINS, CUBE, GRIPPER_OPEN, PARCELS, ROUTES, SORT_PLAN, WALL, variants

MOD = "manaenv.assets.objects.labelloop_qr_sort"
SCENE = lambda n: f"labelloop_qr_sort_case{n}_scene"
ENV = lambda n: f"artixon_arm6a_G_labelloop_qr_sort_case{n}_id_env"
# MEASURED (probe episode, env positions read from the sim): the tabletop-local frame is rotated 180 degrees about the table centre,
# env = (TABLE_C.x - local.x, TABLE_C.y - local.y). Tabletop is 0.685 x 1.11 m.
# Head image: env +x = FAR, env +y = image LEFT (image x ~ 650 at env y -0.27, ~1000 px/m near the bins). Robot root env (-0.322, -0.259).
TABLE_C = (0.2047, -0.2682)
TABLE_HALF = (0.3425, 0.555)  # tabletop half extents, local x (depth) / y (across)
TABLE_Z = -0.235
to_local = lambda ex, ey: (TABLE_C[0] - ex, TABLE_C[1] - ey)  # noqa: E731 (env <-> tabletop-local is its own inverse)
# Layout, tabletop-local, the shoe boxes of props.py with their length pointing away from the robot. The two lane bins stand side by side
# in one row on the FAR side, centred across the table and pushed back to FAR_MARGIN from the far edge so a measured parcel standing
# lengthwise fits on the table in front of them: blue on the robot's left (y -LANE_Y, image-left), red on its right (y +LANE_Y). The
# black reject bin stands on the RIGHT side of the table, beside the right arm, EDGE_MARGIN in from the right edge, at mid-depth.
FAR_MARGIN, EDGE_MARGIN, ZONE_GAP = 0.015, 0.01, 0.03  # bins to the far edge; parcel to the table edges; parcel to the bins
BIN_X = round(-(TABLE_HALF[0] - FAR_MARGIN) + max(b["size"][0] for b in BINS.values()) / 2, 4)
LANE_Y = 0.13
BIN_XY = {"bin_blue": (BIN_X, -LANE_Y), "bin_red": (BIN_X, LANE_Y)}
REJECT_XY = (0.0, round(TABLE_HALF[1] - EDGE_MARGIN - BINS["reject_bin"]["size"][1] / 2, 4))
# induction zone (range of the parcel CENTRE), ENV frame (parcels are teleported with env coordinates): a human swipes parcels onto the
# table from the LEFT, in front of the LEFT arm (lateral -0.25..0.0 m, env y = TABLE_C.y - local y), which detects and handles them.
# zone() shrinks it per parcel and keeps the whole parcel, including the swipe's start point SLIDE_FROM further left, right of
# LEFT_GRIPPER_Y: the left gripper's home pose (preview 1: parcels swiped to lateral -0.35 slid into it).
INDUCT_ENV = {"x": [-0.08, 0.06], "y": [-0.2682, -0.0182]}
LEFT_GRIPPER_Y, SLIDE_FROM = -0.27, 0.08  # tabletop-local y of the left gripper's inner edge; swipe start offset (disturb's 0.08 m)
SLIDE_V = 1.25  # sqrt(2 * mu 1.0 * g * 0.08 m): slides the 8 cm from its entry point to the sampled spot; m/s toward the robot's right (env -y) given to a parcel as it is swiped in (disturb slide_v)
CUBE_SPEC = dict(size=CUBE, mass=(0.07, 0.09), yaw=0.3)  # any parcel not in props.PARCELS
TARGET = {b: SORT_PLAN.get(c.split("/")[1][:3], "reject_bin") for b, c in ROUTES.items()}  # labelled parcel id -> bin from its code's ZIP3
INSTR = ("Read each parcel's QR label: put ZIP 940, 941, 950 and 951 parcels into the blue bin, ZIP 900-902 into the red bin; "
         "put anything else in the black bin on the right; return home.")
# late: parcel asset suffix -> time (s) a human slides it into the induction zone; every parcel is parked off-table until then,
# so the zone is EMPTY at reset (the harness takes its empty-zone reference). The harness gives each parcel up to 3 pick-and-place
# tries (~40 s of sim each): 150 s per single parcel, stream parcels 150 s apart.
STREAM = ("z95112", "z90210", "z10001", "nolabel", "garbage", "z94110", "z90015", "z60601", "z95014", "wallet", "plainbox")
CASES = {
    18: dict(late={"z95112": 6.0}, timeout=150), 19: dict(late={"z90210": 6.0}, timeout=150),
    20: dict(late={"z10001": 6.0}, timeout=150),
    21: dict(late={"nolabel": 6.0}, timeout=150), 22: dict(late={"garbage": 6.0}, timeout=150),
    23: dict(late={"z95112": 6.0, "z90210": 156.0, "z10001": 306.0, "nolabel": 456.0, "garbage": 606.0}, timeout=780),
    **{n: dict(late={b: 6.0}, timeout=150) for n, b in zip(range(24, 30), ("z94110", "z90015", "z60601", "z95014", "wallet", "plainbox"))},
    30: dict(late={b: 6.0 + 150 * i for i, b in enumerate(STREAM)}, timeout=6 + 150 * (len(STREAM) - 1) + 174),  # all 11 parcels
}
BG = {"id": "booth", "asset": {"type": "asset_id", "id": "green_booth_mesh"}, "asset_module": MOD,
      "pose": {"position": [-0.452, -0.271, -0.985], "orientation": [0.0, 0.0, -0.704, 0.709]},
      "required_surfaces": ["tabletop"]}
BOARD = """  labelloop_tabletop:
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
    case_count: %d
    save_path: ./outputs/x2real_eval/labelloop/benchmark
"""


def footprints():
    """{bin id: (x0, x1, y0, y1)} tabletop-local outer footprint of each bin."""
    return {b: (x - BINS[b]["size"][0] / 2, x + BINS[b]["size"][0] / 2, y - BINS[b]["size"][1] / 2, y + BINS[b]["size"][1] / 2)
            for b, (x, y) in {**BIN_XY, "reject_bin": REJECT_XY}.items()}


def spec(b):
    """Parcel b as it arrives: size (x away from the robot, y across, z up), mass range, yaw limit."""
    return PARCELS.get(b, CUBE_SPEC)


def half_extent(b):
    """Half footprint (along x, along y) of parcel b turned by up to its yaw limit."""
    sx, sy, _ = spec(b)["size"]
    c, s = math.cos(spec(b)["yaw"]), math.sin(spec(b)["yaw"])
    return (sx * c + sy * s) / 2, (sy * c + sx * s) / 2


def zone(b):
    """[[x0, x1], [y0, y1]] env range of parcel b's centre: INDUCT_ENV, shrunk so the whole parcel (at any allowed yaw) stays
    EDGE_MARGIN inside the table, ZONE_GAP in front of the lane bins and clear of the reject bin. The 60 mm cubes get INDUCT_ENV unchanged."""
    hx, hy = half_extent(b)
    fp = footprints()
    near_bins = max(fp[b][1] for b in BIN_XY)  # tabletop-local x of the lane bins' near edge
    lx = (near_bins + ZONE_GAP + hx, TABLE_HALF[0] - EDGE_MARGIN - hx)
    ly = (max(-TABLE_HALF[1] + EDGE_MARGIN, LEFT_GRIPPER_Y + SLIDE_FROM) + hy, min(TABLE_HALF[1] - EDGE_MARGIN, fp["reject_bin"][2] - ZONE_GAP) - hy)
    ex, ey = sorted(TABLE_C[0] - v for v in lx), sorted(TABLE_C[1] - v for v in ly)  # -> env (the map flips the axes)
    x = [max(INDUCT_ENV["x"][0], ex[0]), min(INDUCT_ENV["x"][1], ex[1])]
    y = [max(INDUCT_ENV["y"][0], ey[0]), min(INDUCT_ENV["y"][1], ey[1])]
    assert x[1] - x[0] >= 0.01 and y[1] - y[0] >= 0.01, f"no room to drop {b} in front of the bins: x {x}, y {y}"
    if x == INDUCT_ENV["x"] and y == INDUCT_ENV["y"]:
        return [INDUCT_ENV["x"], INDUCT_ENV["y"]]
    inward = lambda lo, hi: [math.ceil(lo * 1e4) / 1e4, math.floor(hi * 1e4) / 1e4]  # noqa: E731 (0.1 mm steps, never outward)
    return [inward(*x), inward(*y)]


def check_layout(margin=0.01):
    """Every bin on the table and no two touching; every parcel of every case has room to arrive (zone() asserts)."""
    fp = footprints()
    for b, (x0, x1, y0, y1) in fp.items():
        assert -TABLE_HALF[0] + margin <= x0 and x1 <= TABLE_HALF[0] - margin, f"{b} off the table in depth"
        assert -TABLE_HALF[1] + margin <= y0 and y1 <= TABLE_HALF[1] - margin, f"{b} off the table across"
    for (a, (ax0, ax1, ay0, ay1)), (b, (bx0, bx1, by0, by1)) in ((p, q) for i, p in enumerate(fp.items()) for q in list(fp.items())[i + 1:]):
        assert ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0, f"{a} overlaps {b}"
    for b in {b for c in CASES.values() for b in c["late"]}:
        zone(b)


def parcel_report():
    """One line per parcel: size, target bin, drop zone, and WARN when its narrowest side seen from above leaves the gripper under
    ~1 cm to spare, its footprint doesn't fit the target bin's cavity as it arrives, or it is taller than the bin walls."""
    lines = []
    for b in sorted({b for c in CASES.values() for b in c["late"]}):
        sx, sy, sz = spec(b)["size"]
        tgt = TARGET.get(b, "reject_bin")
        L, W, H = BINS[tgt]["size"]
        warn = []
        if min(sx, sy) > GRIPPER_OPEN:
            warn.append(f"NOT GRASPABLE: narrowest side {min(sx, sy) * 1000:.0f} mm, gripper opens {GRIPPER_OPEN * 1000:.1f} mm")
        elif min(sx, sy) > GRIPPER_OPEN - 0.01:
            warn.append(f"tight grasp: {min(sx, sy) * 1000:.0f} mm, gripper opens {GRIPPER_OPEN * 1000:.1f} mm")
        if sx > L - 2 * WALL or sy > W - 2 * WALL:
            warn.append(f"footprint {sx * 1000:.0f} x {sy * 1000:.0f} mm does not fit {tgt} ({(L - 2 * WALL) * 1000:.0f} x {(W - 2 * WALL) * 1000:.0f})")
        if sz > H:
            warn.append(f"{(sz - H) * 1000:.0f} mm taller than the {tgt} walls")
        lines.append(f"  {b}: {sx * 1000:.0f} x {sy * 1000:.0f} x {sz * 1000:.0f} mm -> {tgt}, centre zone env x {zone(b)[0]} y {zone(b)[1]}"
                     + (f"  WARN: {'; '.join(warn)}" if warn else ""))
    return "\n".join(lines)


def success_region(b):
    """[half x, half y, z max] of the cavity, relative to the bin origin (centre of its underside): parcel centre inside = landed."""
    L, W, H = BINS[b]["size"]
    return [round(L / 2 - WALL, 4), round(W / 2 - WALL, 4), round(H, 4)]


def obj(oid, asset, tags, **kw):
    return {"id": oid, "asset": {"type": "asset_id", "id": asset}, "semantic_tags": tags, **kw}


def on_table(oid, x, y):
    return {"object": oid, "placement": {"type": "surface", "surface": "tabletop",
            "ranges": {"x": [x - 0.01, x + 0.01], "y": [y - 0.01, y + 0.01]}, "orientation": [0, 0, 0, 1]}}


def scene(n, c):
    late, mass = c["late"], list(CUBE_SPEC["mass"])  # one shared list for the cubes: dumped once with a YAML anchor
    vs = [v for b in late for v in variants(b)]  # every variant of every base id is spawned, parked; the event drops ONE per base
    objs = [obj(f"box_{v}", f"labelloop_box_{v}", ["pickable", "qr_box"], spawn_mass=list(PARCELS[b]["mass"]) if b in PARCELS else mass,
                spawn_static_friction=1.0, spawn_dynamic_friction=1.0) for b in late for v in variants(b)]
    objs += [obj(i, f"labelloop_{i}", ["container"], kinematic=True) for i in (*BIN_XY, "reject_bin")]
    nodes = {i: on_table(i, *xy) for i, xy in {**BIN_XY, "reject_bin": REJECT_XY}.items()}
    for i, v in enumerate(vs):  # parked far below the table until the human event fires (distinct spots)
        nodes[f"box_{v}"] = {"object": f"box_{v}", "placement": {"type": "pose", "poses": [
            {"id": "parked", "position": [0.5 + 0.5 * (i % 8), 3.0 + 0.5 * (i // 8), -5.0], "orientation": [0, 0, 0, 1]}]}}
    return {
        "definition": {"scene_type": SCENE(n), "metadata": {"name": f"labelloop_qr_sort_case{n}", "source": "labelloop",
                       "semantic_goal": "Sort QR-labelled parcels into the bin named by the code (A=blue, B=red, else black)"}},
        "render_settings": {"gen_method": 0, "tags": "green_booth"},
        "background": BG, "objects": objs,
        "layout": {"surfaces": {"tabletop": {"type": "ref", "ref": "background.tabletop"}}, "nodes": nodes, "groups": {}},
        "variants": {"materialization": [], "assignments": []},
    }


def dag(n, c):
    late = c["late"]
    assign = [[[f"box_{v}" for v in variants(b)], TARGET.get(b, "reject_bin")] for b in late]  # [variants of b, bin]: the dropped one must rest in it
    events = {f"human_adds_{b}": {  # human slides the parcel in: teleport it, resting on the table, to a random spot of its zone
        "_func": f"{MOD}.disturb", "mode": "interval", "interval_range_s": [t, t], "is_global_time": True, "object_name": [f"box_{v}" for v in variants(b)],
        "to_xy_range": zone(b), "slide_v": SLIDE_V, "yaw_range": [-spec(b)["yaw"], spec(b)["yaw"]], "z": TABLE_Z + spec(b)["size"][2] / 2,
        "avoid": [f"box_{v}" for o in late if o != b for v in variants(o)]} for b, t in late.items()}
    sections = {"instruction_component": {"instruction": INSTR, "topic": "qr_sort"}, "event_component": {"terms": events}}
    longer = {"episode_length_s": float(c["timeout"]), "max_steps_per_episode": c["timeout"] * 15}  # stage_base is 180 s
    tags = [{"mode": 0, "tags": SCENE(n)}]
    success = {"_func": f"{MOD}.lanes_sorted", "assignments": assign, "bins": {b: success_region(b) for b in BINS},
               "vel_max": 0.5, "consecutive_steps": 5}
    return {"graph_task_generator": {"definition": {
        "task_type": f"labelloop_qr_sort_case{n}",
        "layout": {"runtime_sample": {"id": "runtime", "targets": {
            "placements": [{"id": k, "target": {"type": "node", "id": k}, "sample": {"type": "inherit"}} for k in (*BIN_XY, "reject_bin")]}}},
        "component_sections": sections},
        "nodes": [
            {"id": "sort", "include": "labelloop/task/stage_base.yaml", "score": 8,
             "direct_task_generator": {"definition": {"task_type": f"labelloop_qr_sort_case{n}_sort", **longer},
                "component_sections": {"scene_constraint_component": {"constraints": tags},
                    "termination_component": {"terms": {"success": success}}}}},
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
    ap.add_argument("--timeout", type=int, help="preview: cut every case to this many seconds (rebuild without it to evaluate)")
    a = ap.parse_args()
    if a.timeout:
        for c in CASES.values():
            c["timeout"] = a.timeout
    check_layout()
    print(f"bins x {BIN_X}: " + ", ".join(f"{b} y {xy[1]}" for b, xy in {**BIN_XY, "reject_bin": REJECT_XY}.items()))
    print(parcel_report())
    d = Path(a.x2real) / "manaenv" / "defination"
    dump(d / "labelloop/task/stage_base.yaml", STAGE_BASE)
    policy = {"PolicyPlanner": {"policy_cfg": {"include": "policy/x2real_ps@v1.0.0.yaml"}}}
    for n, c in CASES.items():
        dump(d / f"labelloop/scene/case{n}.yaml", scene(n, c))
        dump(d / f"labelloop/task/case{n}_dag.yaml", dag(n, c))
    for f in [*(d / "labelloop/scene").glob("case*.yaml"), *(d / "labelloop/task").glob("case*_dag.yaml")]:
        if int(re.match(r"case(\d+)", f.name).group(1)) not in CASES:  # left over from removed cases
            f.unlink()
    # x2real_eval reads the suite raw (no include expansion), so copy x2real.yaml and replace its four lists.
    suite = yaml.safe_load((d / "x2real.yaml").read_text())
    suite.update({"scene_orders": [{"include": f"labelloop/scene/case{n}.yaml"} for n in CASES],
             "task_orders": [{"include": f"labelloop/task/case{n}_dag.yaml"} for n in CASES],
             "assembly_orders": [{"assembly_generator": {
                 "name": ENV(n), "scene_generator_name": SCENE(n), "robot_generator_name": "EX001-6R-G",
                 "task_generator_name": f"labelloop_qr_sort_case{n}", "num_envs": 1,
                 "metadata": {"x2real": {"embodiment": "artixon_arm6a", "capabilities": ["benchmark"],
                                         "robot_generator_identity": "EX001-6R-G", "board": "labelloop_tabletop"}}}} for n in CASES],
             "rollout": [{"env_ref": ENV(n), "num_episodes": 40, "episode_length_s": c["timeout"], "max_steps": c["timeout"] * 30,
                          "num_parallel_envs": 4, "show_evaluation_overlay": True, "planner": policy} for n, c in CASES.items()]})
    dump(d / "labelloop.yaml", suite)
    rows = ["subcategory,task,x2real_package,n_id,id_envs,id_timeout_s"]
    rows += [f"labelloop,qr_sort_case{n},labelloop/qr_sort_case{n},1,{ENV(n)},{c['timeout']}" for n, c in CASES.items()]
    csv = d / "x2real/task/labelloop_leaderboard.csv"
    csv.write_text("\n".join(rows) + "\n")
    boards = d / "x2real/task/boards.yaml"
    raw = boards.read_bytes()
    if b"labelloop_tabletop" not in raw:
        with boards.open("a") as f:
            f.write(BOARD % len(CASES))
    else:  # board added by an earlier run: keep its line endings, refresh only case_count
        boards.write_bytes(re.sub(rb"(labelloop_tabletop:\r?\n(?:    .*\r?\n)*?    case_count: )\d+", rb"\g<1>%d" % len(CASES), raw))
    print(f"wrote {len(CASES)} cases under {d}")


if __name__ == "__main__":
    main()
