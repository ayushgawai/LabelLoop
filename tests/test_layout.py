"""Sim table layout from sim/qr_sort_task/build_task.py: bins on the table, every parcel has room to arrive clear of them."""
import itertools
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim" / "qr_sort_task"))
import build_task as bt  # noqa: E402


def corners(b, ex, ey, yaw):
    """Tabletop-local corners of parcel b with its centre at env (ex, ey), turned by yaw."""
    sx, sy, _ = bt.spec(b)["size"]
    lx, ly = bt.to_local(ex, ey)
    for cx, cy in itertools.product((-sx / 2, sx / 2), (-sy / 2, sy / 2)):
        yield lx + cx * math.cos(yaw) - cy * math.sin(yaw), ly + cx * math.sin(yaw) + cy * math.cos(yaw)


def test_layout_passes_and_every_parcel_arrives_clear_of_bins_and_on_the_table():
    bt.check_layout()
    near_bins = max(f[1] for f in bt.footprints().values())
    for b in {b for c in bt.CASES.values() for b in c["late"]}:
        (x0, x1), (y0, y1) = bt.zone(b)
        yaw = bt.spec(b)["yaw"]
        for ex, ey, a in itertools.product((x0, x1), (y0, y1), (-yaw, 0.0, yaw)):
            for px, py in corners(b, ex, ey, a):
                assert px >= near_bins + bt.ZONE_GAP - 1e-6, f"{b} reaches the bins"
                assert abs(px) <= bt.TABLE_HALF[0] - bt.EDGE_MARGIN + 1e-6 and abs(py) <= bt.TABLE_HALF[1] - bt.EDGE_MARGIN + 1e-6


def test_cubes_keep_the_original_zone_and_measured_parcels_get_their_own():
    for b in {b for c in bt.CASES.values() for b in c["late"]} - set(bt.PARCELS):  # still 60 mm cubes
        assert bt.zone(b) == [bt.INDUCT_ENV["x"], bt.INDUCT_ENV["y"]]
    for b in bt.PARCELS:
        (x0, x1), (y0, y1) = bt.zone(b)
        assert bt.INDUCT_ENV["x"][0] <= x0 < x1 <= bt.INDUCT_ENV["x"][1] and bt.INDUCT_ENV["y"][0] <= y0 < y1 <= bt.INDUCT_ENV["y"][1]


def test_success_region_is_each_bins_cavity():
    for b, p in bt.BINS.items():
        L, W, H = p["size"]
        assert bt.success_region(b) == [round(L / 2 - bt.WALL, 4), round(W / 2 - bt.WALL, 4), round(H, 4)]
