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
    sc, dg = bt.scene(36, bt.CASES[36]), bt.dag(36, bt.CASES[36])
    mat = next(o for o in sc["objects"] if o["id"] == "table_mat")
    assert mat["kinematic"] and mat["semantic_tags"] == [] and sc["layout"]["nodes"]["table_mat"]["placement"]["poses"][0]["position"] == [bt.TABLE_C[0], bt.TABLE_C[1], bt.TABLE_Z]
    for b in bt.BINS:
        assert sc["layout"]["nodes"][b]["placement"]["poses"][0]["position"][2] == bt.SURFACE_Z
    ev = dg["graph_task_generator"]["definition"]["component_sections"]["event_component"]["terms"]
    for b, e in ((b, ev[f"human_adds_{b}"]) for b in bt.CASES[36]["late"]):
        assert abs(e["z"] - bt.SURFACE_Z - bt.spec(b)["size"][2] / 2) < 1e-9


def test_success_region_is_each_bins_cavity():
    for b, p in bt.BINS.items():
        L, W, H = p["size"]
        H += max(props.PARCELS[a]["size"][2] for a in props.ACTIVE) if b == "reject_bin" else 0  # tray: rim + thickest parcel
        assert bt.success_region(b) == [round(L / 2 - bt.WALL, 4), round(W / 2 - bt.WALL, 4), round(H, 4)]


def test_layout_lane_bins_far_row_reject_bin_right_side_zone_at_the_centre():
    (bx, by), (rx, ry), (_, zy) = bt.BIN_XY["bin_blue"], bt.BIN_XY["bin_red"], bt.zone("bhindi")
    assert bx == rx and by < 0 < ry and by == -ry  # one row, centred, blue robot-left, red robot-right
    x, y = bt.REJECT_XY
    assert x == 0 and abs(y + bt.BINS["reject_bin"]["size"][1] / 2 - (bt.TABLE_HALF[1] - bt.EDGE_MARGIN)) < 1e-4
    lo, hi = sorted(bt.to_local(0, ey)[1] for ey in zy)
    assert lo < 0 < hi  # induction zone spans the table centre line: both arms reach it


def test_target_follows_the_zip3_of_the_code():
    assert {b: bt.TARGET[b] for b in bt.ROUTES} == {"z95112": "bin_blue", "z90210": "bin_red", "z10001": "bin_blue", "z94110": "bin_blue",
                                                    "z90015": "bin_red", "z60601": "reject_bin", "z95014": "bin_blue",
                                                    "bhindi": "bin_blue", "iqbar_a": "bin_red", "miro": "bin_red", "garbage": "bin_blue"}
    for b in bt.PARCELS:  # unlabelled props are not in TARGET -> reject
        assert bt.TARGET.get(b, "reject_bin") == "reject_bin" or b in bt.ROUTES


def test_variants_one_per_eligible_face_mailer_only_top_and_bottom():
    for b, code in bt.ROUTES.items():
        x, y, z = bt.PARCELS[b]["size"]
        dims = {"pz": (x, y), "nz": (x, y), "py": (x, z), "ny": (x, z), "px": (y, z), "nx": (y, z)}
        lw, lh = props.LABEL_SIZES[bt.PARCELS[b]["label"]]
        v = bt.variants(b)  # fits in either orientation
        assert list(v) == [f"{b}_{f}" for f in bt.PARCELS[b].get("faces", props.FACES) if max(dims[f]) * 1000 >= max(lw, lh) - 1e-6 and min(dims[f]) * 1000 >= min(lw, lh) - 1e-6]
        assert all(p["labels"] == (vid.rsplit("_", 1)[1],) and p["size"] == bt.PARCELS[b]["size"] for vid, p in v.items())
    assert list(bt.variants("z10001")) == ["z10001_pz", "z10001_nz"]
    for b in ("bhindi", "iqbar_a", "miro", "garbage"):  # real flat boxes: label on the big top or bottom face only
        assert list(bt.variants(b)) == [f"{b}_pz", f"{b}_nz"]
    assert {b: bt.PARCELS[b]["label"] for b in bt.ROUTES} == {"z94110": "small", "z60601": "small", "z95112": "large", "z90210": "large",
                                                              "z10001": "large", "z90015": "large", "z95014": "large", "bhindi": "large", "iqbar_a": "large", "miro": "large", "garbage": "large"}
    assert len(bt.variants("z60601")) == 6 and not props.fits((0.127, 0.064), "large") and props.fits((0.064, 0.127), "small")
    assert bt.variants("wallet") == {"wallet": bt.PARCELS["wallet"]} and bt.PARCELS["wallet"]["labels"] == ()
    assert bt.variants("nolabel") == {"nolabel": None}


def test_cases_scene_and_dag_group_variants_per_base_id():
    c = bt.CASES[36]
    assert len(c["late"]) == 7 and c["timeout"] == max(c["late"].values()) + 174
    assert list(c["late"].values()) == [6.0 + 150 * i for i in range(7)] and list(c["late"]) == list(props.ACTIVE)
    sc, dg = bt.scene(36, c), bt.dag(36, c)
    names = {o["id"] for o in sc["objects"] if o["id"].startswith("box_")}
    assert names == {f"box_{v}" for b in c["late"] for v in bt.variants(b)}
    parked = [n["placement"]["poses"][0]["position"] for k, n in sc["layout"]["nodes"].items() if k in names]
    assert len({tuple(p) for p in parked}) == len(names)
    ev = dg["graph_task_generator"]["definition"]["component_sections"]["event_component"]["terms"]
    assert ev["human_adds_z10001"]["object_name"] == ["box_z10001_pz", "box_z10001_nz"] and ev["human_adds_z10001"]["slide_v"] == bt.SLIDE_V
    assert set(ev["human_adds_z10001"]["object_name"]) <= set(ev["human_adds_bhindi"]["avoid"])
    succ = dg["graph_task_generator"]["nodes"][0]["direct_task_generator"]["component_sections"]["termination_component"]["terms"]["success"]
    assert [["box_z10001_pz", "box_z10001_nz"], "bin_blue"] in succ["assignments"]
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
    assert [r["lane"] for r in spares] == ["reject", "B", "reject", "reject", "reject"] and {r["size"] for r in spares} == {"large"}


def test_active_cases_are_exactly_the_users_seven_parcels_plus_calibration_and_stream():
    assert set(props.ACTIVE) == {"bhindi", "iqbar_a", "miro", "z10001", "iqbar_b", "garbage", "wallet"}
    assert set(bt.CASES) == {20, 22, 28, 31, 32, 33, 34, 35, 36, *range(40, 48)}
    single = {n: next(iter(c["late"])) for n, c in bt.CASES.items() if len(c["late"]) == 1 and n < 40}
    assert single == {32: "bhindi", 33: "iqbar_a", 34: "miro", 20: "z10001", 35: "iqbar_b", 22: "garbage", 28: "wallet"}
    assert all(bt.CASES[n] == dict(late={b: 6.0}, timeout=150) for n, b in single.items())
    assert bt.CASES[31] == dict(late={}, timeout=90) and tuple(bt.STREAM) == props.ACTIVE
    for n, c in bt.CASES.items():
        bt.scene(n, c), bt.dag(n, c)
    for b in props.ACTIVE:  # a drop zone for every item, and every item lies flat (z is the thin side)
        assert bt.zone(b)
        assert bt.spec(b)["size"][2] < min(bt.spec(b)["size"][:2]), b


def test_every_active_item_has_the_right_label_faces_and_routing():
    expect = {"bhindi": ("pz", "nz", "bin_blue", "LL-08"), "iqbar_a": ("pz", "nz", "bin_red", "LL-11"), "miro": ("pz", "nz", "bin_red", "LL-12"),
              "z10001": ("pz", "nz", "bin_blue", "LL-09"), "garbage": ("pz", "nz", "bin_blue", "LL-10")}
    for b, (f1, f2, tgt, lid) in expect.items():
        assert list(bt.variants(b)) == [f"{b}_{f1}", f"{b}_{f2}"] and bt.TARGET[b] == tgt and props.LABEL_OF[b]["label_id"] == lid
    for b in ("iqbar_b", "wallet"):  # no label anywhere -> reject
        assert bt.variants(b) == {b: bt.PARCELS[b]} and bt.PARCELS[b]["labels"] == () and b not in bt.ROUTES and bt.TARGET.get(b, "reject_bin") == "reject_bin"
    a, c = bt.PARCELS["iqbar_a"], bt.PARCELS["iqbar_b"]
    assert all(a[k] == c[k] for k in ("size", "mass", "yaw", "color", "tape")) and not hasattr(props, "lobar") and "lobar" not in bt.PARCELS


def test_reject_tray_is_low_black_lid_on_the_right_inside_the_table():
    t = bt.BINS["reject_bin"]
    assert t["size"] == (13.2 * props.IN, 9.2 * props.IN, 1.0 * props.IN) and t["outside"][0] < 60 and min(t["inside"]) > 220
    x0, x1, y0, y1 = bt.footprints()["reject_bin"]
    assert y0 > 0 and -bt.TABLE_HALF[0] <= x0 and x1 <= bt.TABLE_HALF[0] and y1 <= bt.TABLE_HALF[1] - bt.EDGE_MARGIN + 1e-6  # right side, on the table
    hx, hy, zmax = bt.success_region("reject_bin")  # parcel centre in the footprint and resting within rim + thickest parcel
    assert abs(hx - (t["size"][0] / 2 - bt.WALL)) < 1e-4 and zmax == round(t["size"][2] + 0.0635, 4)
    for p in props.ACTIVE:  # every flat parcel's resting centre is inside the region
        assert bt.WALL + bt.spec(p)["size"][2] / 2 <= zmax


def _sampler():
    """labelloop_qr_sort.py imports Isaac Sim / torch at the top: load it with stubs, only its pure-python sampler is used."""
    import importlib.util
    import types
    stubs = {"torch": {}, "isaaclab_arena": {}, "isaaclab_arena.assets": {}, "manaenv": {}, "manaenv.assets": {},
             "isaaclab_arena.assets.object_base": {"ObjectType": types.SimpleNamespace(RIGID=0)},
             "isaaclab_arena.assets.object_library": {"LibraryObject": object},
             "isaaclab_arena.assets.register": {"register_asset": lambda c: None},
             "manaenv.assets.resolver": {"resolve_runtime_asset_reference": lambda s: s}}
    saved = {k: sys.modules.get(k) for k in stubs}
    try:
        for k, attrs in stubs.items():
            sys.modules[k] = types.ModuleType(k)
            sys.modules[k].__dict__.update(attrs)
        spec = importlib.util.spec_from_file_location("lq_stub", Path(__file__).resolve().parents[1] / "sim" / "qr_sort_task" / "labelloop_qr_sort.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for k, v in saved.items():
            sys.modules.pop(k, None) if v is None else sys.modules.__setitem__(k, v)
    return mod


def test_anywhere_sampler_respects_every_constraint_on_10k_samples_per_item():
    import random
    lq = _sampler()
    rng = random.Random(7)
    for b in props.ACTIVE:
        cfg = bt.anywhere_cfg(b)
        area, yaw_frac = bt.anywhere_area(b)
        assert area > 0 and yaw_frac > 0, b
        for _ in range(10000):
            x, y, yaw = lq.anywhere_sample(cfg, [], rng)
            assert -math.pi <= yaw <= math.pi and bt.anywhere_ok(cfg, x, y, yaw) and lq._anywhere_ok(cfg, x, y, yaw), (b, x, y, yaw)
        for _ in range(3000):  # the two checks agree on arbitrary points, not only on accepted ones
            x, y, yaw = rng.uniform(-0.3, 0.8), rng.uniform(-0.9, 0.4), rng.uniform(-math.pi, math.pi)
            assert bt.anywhere_ok(cfg, x, y, yaw) == lq._anywhere_ok(cfg, x, y, yaw)
        # independent re-derivation from the footprint corners (tabletop-local): table, bins +3 cm, gripper strips, reach
        for _ in range(300):
            x, y, yaw = lq.anywhere_sample(cfg, [], rng)
            cs = list(corners(b, x, y, yaw))
            assert all(abs(px) <= bt.TABLE_HALF[0] - bt.ANY_EDGE + 1e-6 and abs(py) <= bt.TABLE_HALF[1] - bt.ANY_EDGE + 1e-6 for px, py in cs)
            assert all(bt.LEFT_GRIPPER_Y - 1e-6 <= py <= -bt.LEFT_GRIPPER_Y + 1e-6 for px, py in cs)
            for x0, x1, y0, y1 in bt.footprints().values():
                assert all(not (x0 - bt.ANY_BIN < px < x1 + bt.ANY_BIN and y0 - bt.ANY_BIN < py < y1 + bt.ANY_BIN) for px, py in cs)
            lx, ly = bt.to_local(x, y)
            assert any(math.hypot(bt.ROOT_LOCAL[0] - lx, bt.ROOT_LOCAL[1] - ly - s * bt.ARM_Y) <= bt.REACH + 1e-6 for s in (-1, 1))
    # avoid honoured: with another parcel sitting somewhere valid, the draw keeps cfg["clear"] away
    cfg, other = dict(bt.anywhere_cfg("bhindi"), clear=0.1), (0.05, -0.27)
    assert all(math.hypot(x - other[0], y - other[1]) >= cfg["clear"] for x, y, _ in (lq.anywhere_sample(cfg, [other], rng) for _ in range(500)))


def test_prefill_parcels_rest_label_up_inside_their_bins_and_are_not_assigned():
    for n in (*range(40, 48),):
        c = bt.CASES[n]
        sc, dg = bt.scene(n, c), bt.dag(n, c)
        pre = {o["id"]: o for o in sc["objects"] if o["id"].startswith("pre_")}
        assert set(pre) == {f"pre_{v}" for v in bt.PREFILL.values()} and all(o["kinematic"] for o in pre.values())
        for bn, v in bt.PREFILL.items():
            base = v.rsplit("_", 1)[0]
            assert v.endswith("_pz") and bt.variants(base)[v]["labels"] == ("pz",) and pre[f"pre_{v}"]["asset"]["id"] == f"labelloop_box_{v}"
            pose = sc["layout"]["nodes"][f"pre_{v}"]["placement"]["poses"][0]
            (px, py, pz), (bx, by, _) = pose["position"], sc["layout"]["nodes"][bn]["placement"]["poses"][0]["position"]
            sx, sy, sz = bt.variants(base)[v]["size"]
            assert pose["orientation"] == [0, 0, 0, 1]  # label face up (+z), yaw 0
            assert abs(pz - (bt.SURFACE_Z + bt.WALL + sz / 2)) < 1e-3
            L, W, _ = bt.BINS[bn]["size"]
            assert abs(px - bx) <= (L - sx) / 2 - bt.WALL + 1e-6 and abs(py - by) <= (W - sy) / 2 - bt.WALL + 1e-6  # inside the cavity
        nodes = dg["graph_task_generator"]["nodes"][0]["direct_task_generator"]["component_sections"]["termination_component"]["terms"]["success"]
        assigned = {o for objs, _ in nodes["assignments"] for o in objs}
        assert not any(o.startswith("box_pre_") or o.startswith("pre_") for o in assigned) and assigned == {f"box_{v}" for b in c["late"] for v in bt.variants(b)}
    assert not any(o["id"].startswith("pre_") for o in bt.scene(36, bt.CASES[36])["objects"])  # option off -> unchanged


def test_position_cases_40_to_47():
    assert all(bt.CASES[40 + i] == dict(late={b: 6.0}, timeout=150, anywhere=True, prefill=True) for i, b in enumerate(props.ACTIVE))
    s = bt.CASES[47]
    assert list(s["late"]) == list(props.ACTIVE) and list(s["late"].values()) == [6.0 + 150 * i for i in range(7)] and s["anywhere"] and s["prefill"]
    for n in range(40, 48):
        ev = bt.dag(n, bt.CASES[n])["graph_task_generator"]["definition"]["component_sections"]["event_component"]["terms"]
        for b, e in ev.items():
            assert "to_xy_range" not in e and e["yaw_range"] == [-math.pi, math.pi] and e["anywhere"]["size"] == bt.anywhere_cfg(b[len("human_adds_"):])["size"]
    for b in props.ACTIVE:
        assert bt.anywhere_area(b)[0] > 0
    assert "anywhere region" in bt.parcel_report()
