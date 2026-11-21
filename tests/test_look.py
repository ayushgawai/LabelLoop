import numpy as np

from labelloop import look
from labelloop.config import load_yaml
from labelloop.pose import rpy_to_matrix
from labelloop.wallx_client import HORIZON, move_response
from pathlib import Path

LC = load_yaml(Path(__file__).parent.parent / "configs" / "task_qr_sort.yaml")["look"]


def test_pixel_to_table_centre_and_mirror():
    c = look.pixel_to_table(640, 360)
    assert abs(c[2] - 0.015) < 1e-9 and 0.3 < c[0] < 0.7 and abs(c[1]) < 1e-6
    a, b = look.pixel_to_table(400, 500), look.pixel_to_table(880, 500)
    assert np.allclose([a[0], a[1], a[2]], [b[0], -b[1], b[2]]) and a[1] > 0  # image-left = robot left (+y)
    assert np.allclose(look.pixel_to_table(320, 180, (640, 360)), look.pixel_to_table(640, 360))  # scaled image


def test_every_view_aims_the_wrist_camera_at_the_target():
    assert [a for a, *_ in look.views([0.35, 0.10, 0.015], LC)] == ["left_ee"] * 2 + ["right_ee"]  # far side: EE beyond reach 0.55
    for p in ([0.35, 0.10, 0.015], [0.35, 0.0, 0.015]):
        vs = look.views(p, {**LC, "reach": 0.7})
        assert [a for a, *_ in vs] == ["left_ee"] * 3 + ["right_ee"]
        for arm, pos, rpy in vs:
            R = rpy_to_matrix(rpy)
            cam = np.array(pos) + R @ look.CAM_OFF + [0, look.ARM_Y[arm], 0]
            axis = R @ rpy_to_matrix([0, look.CAM_TILT, 0]) @ [1, 0, 0]
            T = np.array(p) + [0, 0, LC["target_h"]]
            d = T - cam
            assert np.linalg.norm(d - (d @ axis) * axis) < 0.01 and d @ axis > 0
            assert abs(np.hypot(*d[:2]) - LC["distance"]) < 1e-6 and pos[2] >= 0.015 + 0.03 - 1e-3 and np.linalg.norm(pos) <= 0.7


def test_unreachable_views_are_skipped():
    assert len(look.views([1.5, 0, 0.015], LC)) == 0


def test_move_response_endpoints_and_hold():
    z = {"position": [0, 0, 0], "rotation_rpy": [0, 0, 0], "gripper": 0.3}
    st = {"left_ee": z, "right_ee": {**z, "position": [0.1, 0.2, 0.3]}}
    r = move_response("e", st, {"left_ee": {"position": [0.5, 0, 0.2], "rotation_rpy": [0, 0.2, 0.4]}})["actions"]
    assert len(r["left_ee"]["position"]) == HORIZON
    assert np.allclose(r["left_ee"]["position"][-1], [0.5, 0, 0.2]) and np.allclose(r["left_ee"]["rotation_rpy"][-1], [0, 0.2, 0.4])
    assert set(r["left_ee"]["gripper"]) == {0.3} and all(p == [0.1, 0.2, 0.3] for p in r["right_ee"]["position"])
