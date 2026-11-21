import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from labelloop.perception.table_grid import TableGrid, find_table_quad

W, D = 1.11, 0.685
TABLE = np.array([[0, D], [W, D], [W, 0], [0, 0]], np.float64)  # far-left, far-right, near-right, near-left (m)
MAT, BG, GRIPPER, NAVY, RED, KRAFT = (46, 178, 150), (70, 72, 75), (200, 200, 205), (40, 57, 73), (235, 30, 30), (165, 125, 85)
CORNERS = {"in_frame": [[330.4, 95.2], [975.7, 90.6], [1190.3, 600.8], [95.1, 615.5]],  # far-left, far-right, near-right, near-left (px)
           "near_off_frame": [[300.2, 80.5], [1010.6, 82.1], [1430.4, 680.3], [-150.7, 690.2]]}  # the sim geometry: near corners outside the image
OCCLUDERS = {"near_corners": [((0, 500), (230, 719), GRIPPER), ((1080, 480), (1279, 719), GRIPPER)],  # both grippers hide the near corners
             "far_left_near_right": [((260, 30), (470, 270), NAVY), ((1080, 480), (1279, 719), GRIPPER)]}  # a bin over far-left, a gripper over near-right


def render(corners, occluders=()):
    """1280x720 RGB: a green table-sized canvas warped into the image by the known homography table m -> corners px (bilinear, so the mat
    edge is anti-aliased), two bins standing on the far edge, a kraft parcel on the mat, then the occluding rectangles. -> (img, H_true)."""
    H = cv2.getPerspectiveTransform(TABLE.astype(np.float32), np.float32(corners)).astype(np.float64)
    A = np.array([[1e-3, 0, 0], [0, -1e-3, D], [0, 0, 1]])  # canvas px (1 mm, row 0 = far edge) -> table m
    a = cv2.warpPerspective(np.ones((round(D * 1000), round(W * 1000)), np.float32), H @ A, (1280, 720))[..., None]
    img = (np.float32(BG) * (1 - a) + np.float32(MAT) * a).round().astype(np.uint8)
    for (x0, x1), col in (((480, 620), NAVY), ((660, 800), RED)):  # bins hiding the middle of the far edge
        cv2.rectangle(img, (x0, 20), (x1, 230), col, -1)
    cv2.rectangle(img, (560, 380), (700, 470), KRAFT, -1)
    for p0, p1, col in occluders:
        cv2.rectangle(img, p0, p1, col, -1)
    return img, H


def table_pts():
    x, y = np.meshgrid(np.linspace(0.05, W - 0.05, 12), np.linspace(0.05, D - 0.05, 8))
    return x.ravel(), y.ravel()


@pytest.mark.parametrize("occ", OCCLUDERS)
@pytest.mark.parametrize("geom", CORNERS)
def test_corners_and_round_trip_with_two_corners_hidden(geom, occ):
    img, H = render(CORNERS[geom], OCCLUDERS[occ])
    q = find_table_quad(img)
    assert q is not None
    assert np.abs(q - CORNERS[geom]).max() < 4, q.round(1)
    g = TableGrid(q, W, D)
    x, y = table_pts()
    p = H @ np.stack([x, y, np.ones_like(x)])
    ex, ey = g.to_table(p[0] / p[2], p[1] / p[2])  # true pixel of a known table point -> estimated table point
    assert np.hypot(ex - x, ey - y).max() < 0.005
    bx, by = g.to_table(*g.to_pixel(x, y))
    assert np.hypot(bx - x, by - y).max() < 1e-6


def test_frame_orientation_and_scalars():
    g = TableGrid(CORNERS["in_frame"], W, D)
    for (u, v), t in zip(CORNERS["in_frame"], TABLE):
        assert np.allclose(g.to_table(u, v), t, atol=1e-6)
    x, y = g.to_table(*CORNERS["in_frame"][3])
    assert np.ndim(x) == 0 and abs(x) < 1e-6 and abs(y) < 1e-6  # near-left = origin
    u, v = g.to_pixel(W, 0)
    assert np.allclose((u, v), CORNERS["in_frame"][2])  # x = robot's right: near-right


def test_no_mat_none():
    img = np.full((720, 1280, 3), BG, np.uint8)
    img[300:400, 500:600] = MAT  # a small green patch is not a table
    assert find_table_quad(img) is None


def test_draw_copy_with_grid():
    img, _ = render(CORNERS["in_frame"])
    out = TableGrid(find_table_quad(img), W, D).draw(img)
    assert out.shape == img.shape and out is not img and (out != img).any()
    assert (img == render(CORNERS["in_frame"])[0]).all()  # input untouched


def test_preview_script(tmp_path):
    img, _ = render(CORNERS["in_frame"], OCCLUDERS["near_corners"])
    src, out = tmp_path / "head.jpg", tmp_path / "grid.png"
    cv2.imwrite(str(src), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    root = Path(__file__).resolve().parent.parent
    r = subprocess.run([sys.executable, str(root / "scripts" / "table_grid_preview.py"), str(src), "--w", "1.11", "--d", "0.685", str(out)],
                       capture_output=True, text=True, cwd=root)
    assert r.returncode == 0, r.stderr
    assert "far-left" in r.stdout and "near-left" in r.stdout and cv2.imread(str(out)) is not None
