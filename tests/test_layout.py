"""Sim table layout from sim/qr_sort_task/build_task.py: bins on the table, every parcel has room to arrive clear of them."""
import itertools
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim" / "qr_sort_task"))
import build_task as bt  # noqa: E402
import props  # noqa: E402


def corners(b, ex, ey, yaw):
    """Tabletop-local corners of parcel b with its centre at env (ex, ey), turned by yaw."""
    sx, sy, _ = bt.spec(b)["size"]
    lx, ly = bt.to_local(ex, ey)
    for cx, cy in itertools.product((-sx / 2, sx / 2), (-sy / 2, sy / 2)):
        yield lx + cx * math.cos(yaw) - cy * math.sin(yaw), ly + cx * math.sin(yaw) + cy * math.cos(yaw)


def test_layout_passes_and_every_parcel_arrives_clear_of_bins_and_on_the_table():
    bt.check_layout()
    fp = bt.footprints()
    near_bins, reject_y0 = max(fp[b][1] for b in bt.BIN_XY), fp["reject_bin"][2]
    for b in {b for c in bt.CASES.values() for b in c["late"]}:
        (x0, x1), (y0, y1) = bt.zone(b)
        yaw = bt.spec(b)["yaw"]
        for ex, ey, a in itertools.product((x0, x1), (y0, y1), (-yaw, 0.0, yaw)):
            for px, py in corners(b, ex, ey, a):
                assert px >= near_bins + bt.ZONE_GAP - 1e-6, f"{b} reaches the lane bins"
                assert py <= reject_y0 - bt.ZONE_GAP + 1e-6, f"{b} reaches the reject bin"
                assert abs(px) <= bt.TABLE_HALF[0] - bt.EDGE_MARGIN + 1e-6 and abs(py) <= bt.TABLE_HALF[1] - bt.EDGE_MARGIN + 1e-6


def test_every_zone_is_inside_the_induction_zone_spans_the_centre_and_clear_of_the_left_gripper():
    for b in {b for c in bt.CASES.values() for b in c["late"]}:
        (x0, x1), (y0, y1) = bt.zone(b)
        assert bt.INDUCT_ENV["x"][0] <= x0 < x1 <= bt.INDUCT_ENV["x"][1] and bt.INDUCT_ENV["y"][0] <= y0 < y1 <= bt.INDUCT_ENV["y"][1]
        left_edge = bt.to_local(0, y1)[1] - bt.SLIDE_FROM - bt.half_extent(b)[1]  # swipe start, robot's left side of the parcel
        assert left_edge >= bt.LEFT_GRIPPER_Y - 1e-4, b


def test_green_mat_covers_the_table_and_bins_and_parcels_sit_on_its_top():
    L, W, T = props.TABLE_MAT["size"]
    assert all(0 < h - m / 2 <= 0.01 for h, m in zip(bt.TABLE_HALF, (L, W))) and bt.SURFACE_Z == bt.TABLE_Z + T  # inset <= 1 cm per side
    sc, dg = bt.scene(30, bt.CASES[30]), bt.dag(30, bt.CASES[30])
    mat = next(o for o in sc["objects"] if o["id"] == "table_mat")
    assert mat["kinematic"] and mat["semantic_tags"] == [] and sc["layout"]["nodes"]["table_mat"]["placement"]["surface"] == "tabletop"
    for b in bt.BINS:
        assert sc["layout"]["nodes"][b]["placement"]["poses"][0]["position"][2] == bt.SURFACE_Z
    ev = dg["graph_task_generator"]["definition"]["component_sections"]["event_component"]["terms"]
    for b, e in ((b, ev[f"human_adds_{b}"]) for b in bt.CASES[30]["late"]):
        assert abs(e["z"] - bt.SURFACE_Z - bt.spec(b)["size"][2] / 2) < 1e-9


def test_success_region_is_each_bins_cavity():
    for b, p in bt.BINS.items():
        L, W, H = p["size"]
        assert bt.success_region(b) == [round(L / 2 - bt.WALL, 4), round(W / 2 - bt.WALL, 4), round(H, 4)]


def test_layout_lane_bins_far_row_reject_bin_right_side_zone_at_the_centre():
    (bx, by), (rx, ry), (_, zy) = bt.BIN_XY["bin_blue"], bt.BIN_XY["bin_red"], bt.zone("z94110")
    assert bx == rx and by < 0 < ry and by == -ry  # one row, centred, blue robot-left, red robot-right
    x, y = bt.REJECT_XY
    assert x == 0 and abs(y + bt.BINS["reject_bin"]["size"][1] / 2 - (bt.TABLE_HALF[1] - bt.EDGE_MARGIN)) < 1e-4
    lo, hi = sorted(bt.to_local(0, ey)[1] for ey in zy)
    assert lo < 0 < hi  # induction zone spans the table centre line: both arms reach it


def test_target_follows_the_zip3_of_the_code():
    assert {b: bt.TARGET[b] for b in bt.ROUTES} == {"z95112": "bin_blue", "z90210": "bin_red", "z10001": "reject_bin", "z94110": "bin_blue",
                                                    "z90015": "bin_red", "z60601": "reject_bin", "z95014": "bin_blue"}
    for b in bt.PARCELS:  # unlabelled props are not in TARGET -> reject
        assert bt.TARGET.get(b, "reject_bin") == "reject_bin" or b in bt.ROUTES


def test_variants_one_per_eligible_face_mailer_only_top_and_bottom():
    for b, code in bt.ROUTES.items():
        x, y, z = bt.PARCELS[b]["size"]
        dims = {"pz": (x, y), "nz": (x, y), "py": (x, z), "ny": (x, z), "px": (y, z), "nx": (y, z)}
        lw, lh = props.LABEL_SIZES[bt.PARCELS[b]["label"]]
        v = bt.variants(b)  # fits in either orientation
        assert list(v) == [f"{b}_{f}" for f in props.FACES if max(dims[f]) * 1000 >= max(lw, lh) - 1e-6 and min(dims[f]) * 1000 >= min(lw, lh) - 1e-6]
        assert all(p["labels"] == (vid.rsplit("_", 1)[1],) and p["size"] == bt.PARCELS[b]["size"] for vid, p in v.items())
    assert list(bt.variants("z10001")) == ["z10001_pz", "z10001_nz"]
    assert {b: bt.PARCELS[b]["label"] for b in bt.ROUTES} == {"z94110": "small", "z60601": "small", "z95112": "large", "z90210": "large",
                                                              "z10001": "large", "z90015": "large", "z95014": "large"}
    assert len(bt.variants("z60601")) == 6 and not props.fits((0.127, 0.064), "large") and props.fits((0.064, 0.127), "small")
    assert bt.variants("wallet") == {"wallet": bt.PARCELS["wallet"]} and bt.PARCELS["wallet"]["labels"] == ()
    assert bt.variants("nolabel") == {"nolabel": None}


def test_cases_scene_and_dag_group_variants_per_base_id():
    c = bt.CASES[30]
    assert len(c["late"]) == 11 and c["timeout"] == max(c["late"].values()) + 174
    sc, dg = bt.scene(30, c), bt.dag(30, c)
    names = {o["id"] for o in sc["objects"] if o["id"].startswith("box_")}
    assert names == {f"box_{v}" for b in c["late"] for v in bt.variants(b)}
    parked = [n["placement"]["poses"][0]["position"] for k, n in sc["layout"]["nodes"].items() if k in names]
    assert len({tuple(p) for p in parked}) == len(names)
    ev = dg["graph_task_generator"]["definition"]["component_sections"]["event_component"]["terms"]
    assert ev["human_adds_z10001"]["object_name"] == ["box_z10001_pz", "box_z10001_nz"] and ev["human_adds_z10001"]["slide_v"] == bt.SLIDE_V
    assert set(ev["human_adds_z10001"]["object_name"]) <= set(ev["human_adds_z95112"]["avoid"])
    succ = dg["graph_task_generator"]["nodes"][0]["direct_task_generator"]["component_sections"]["termination_component"]["terms"]["success"]
    assert [["box_z10001_pz", "box_z10001_nz"], "reject_bin"] in succ["assignments"]
    assert [["box_wallet"], "reject_bin"] in succ["assignments"]


def test_registered_assets_match_props_variants():
    import ast
    src = (Path(__file__).resolve().parents[1] / "sim" / "qr_sort_task" / "labelloop_qr_sort.py").read_text()
    consts = {n.targets[0].id: n.value for n in ast.parse(src).body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    allf = ast.literal_eval(consts["_ALL"])
    labelled = {k: (allf if isinstance(v, ast.Name) else ast.literal_eval(v)) for k, v in zip(consts["LABELLED"].keys, consts["LABELLED"].values)
                for k in [ast.literal_eval(k)]}
    assert labelled == {b: tuple(v.rsplit("_", 1)[1] for v in bt.variants(b)) for b in bt.ROUTES}


def test_label_registry_routes_match_sort_plan_and_code_pattern():
    import re
    import yaml
    pat = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "task_qr_sort.yaml").read_text())["code_pattern"]
    ids = [r["label_id"] for r in props.LABELS]
    assert len(set(ids)) == len(ids) and len({r["qr_payload"] for r in props.LABELS}) == len(ids)
    for r in props.LABELS:
        assert re.match(pat, r["qr_payload"]).group(1) == r["zip5"] and r["zip3"] == r["zip5"][:3]
        assert r["target_bin"] == bt.SORT_PLAN.get(r["zip3"], "reject_bin") and r["lane"] == props.LANE[r["target_bin"]]
        assert bt.TARGET.get(r["sim_id"], r["target_bin"]) == r["target_bin"]
    assert {r["sim_id"]: r["qr_payload"] for r in props.LABELS if r["sim_id"]} == props.ROUTES
    spares = [r for r in props.LABELS if not r["sim_id"]]
    assert [r["lane"] for r in spares] == ["A"] * 3 + ["B"] * 3 + ["reject"] * 3 and {r["size"] for r in spares} == {"large"}
