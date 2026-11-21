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


def hom(R=np.eye(3), p=(0, 0, 0)):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


def rot(axis, a):
    c, s = np.cos(a), np.sin(a)
    return {"x": [[1, 0, 0], [0, c, -s], [0, s, c]], "y": [[c, 0, s], [0, 1, 0], [-s, 0, c]], "z": [[c, -s, 0], [s, c, 0], [0, 0, 1]]}[axis]


# independent copy of ex001_6r_legacy.urdf (joint origin, axis, limits): homogeneous 4x4 chain, not look.fk
URDF = [((0, 0, 0.0613), "z", -2.7925, 2.7925), ((0.025, 0, 0.0542), "y", 0, 3.6652), ((-0.26, 0, 0), "y", -2.7925, 0),
        ((0.2315, 0, 0.075), "y", -1.5708, 1.5708), ((0.0825, 0, 0.085), "z", -1.5708, 1.5708), ((0.0395, 0, -0.0945), "x", -1.9199, 1.9199)]


def urdf_fk(q):
    T = np.eye(4)
    for (o, ax, lo, hi), a in zip(URDF, q):
        assert lo - 1e-6 <= a <= hi + 1e-6
        T = T @ hom(p=o) @ hom(rot(ax, a))
    return T


PARCELS = ([0.30, 0.25, 0.015], [0.30, 0.15, 0.015], [0.35, 0.05, 0.015])


def test_views_reproduce_their_ee_pose_through_the_urdf_chain_and_keep_the_fingers_clear():
    n = 0
    for ay in (0.29, 0.266):
        cfg = {**LC, "arm_y": ay}
        for p in PARCELS:
            for arm, ee, rpy, info in look.views(p, cfg):
                T = urdf_fk(info["q"])  # raises if a joint limit is violated
                assert np.linalg.norm(T[:3, 3] - ee) < 0.01 and np.linalg.norm(T[:3, :3] - rpy_to_matrix(rpy)) < 0.05
                root = np.array([0, (1 if arm == "left_ee" else -1) * ay, cfg["arm_z"]])
                tip = T[:3, 3] + T[:3, :3] @ [0.217, 0, -0.01] + root
                hw, c = cfg["hw"], np.array(p) + [0, 0, cfg["target_h"]]
                d = np.linalg.norm(np.maximum(np.abs(tip - c) - [hw, hw, cfg["target_h"]], 0))
                assert d >= cfg["finger_clear"] - 0.01 and tip[2] >= 0.015 + cfg["finger_clear"] - 0.01
                cam = T[:3, 3] + T[:3, :3] @ look.CAM_OFF + root
                assert np.hypot(*(cam[:2] - c[:2])) >= hw + cfg["standoff"] - 0.01  # camera standoff
                n += 1
    assert n >= 8  # honest floor: this many views survive over the 6 parcel / arm-spacing cases


def test_label_is_aimed_aim_down_below_the_target_direction():
    for arm, ee, rpy, _ in look.views(PARCELS[2], LC):
        R = rpy_to_matrix(rpy)
        cam = ee + R @ look.CAM_OFF + [0, (1 if arm == "left_ee" else -1) * LC["arm_y"], LC["arm_z"]]
        axis = R @ rpy_to_matrix([0, look.CAM_TILT, 0]) @ [1, 0, 0]
        d = np.array(PARCELS[2]) + [0, 0, LC["target_h"]] - cam
        below = np.arctan2(-axis[2], np.hypot(*axis[:2])) - np.arctan2(-d[2], np.hypot(*d[:2]))
        assert abs(below - LC["aim_down"]) < 0.01 and abs(np.arctan2(axis[1], axis[0]) - np.arctan2(d[1], d[0])) < 0.01


def test_unreachable_parcel_gets_no_views():
    assert look.views([1.5, 0, 0.015], LC) == []


def test_pixel_to_table_matches_the_sim_readme_measurement():
    # sim/README.md: head image x ~650 at root y ~0 (env y -0.27); ~1000 px/m near the mats (loose)
    assert abs(look.pixel_to_table(650, 450)[1]) < 0.05
    a, b = look.pixel_to_table(550, 450), look.pixel_to_table(750, 450)
    assert 800 < 200 / abs(a[1] - b[1]) < 1500
    assert look.pixel_to_table(650, 450, z=0.065)[0] < look.pixel_to_table(650, 450)[0]  # a higher plane meets the head ray nearer


def test_move_response_endpoints_and_hold():
    z = {"position": [0, 0, 0], "rotation_rpy": [0, 0, 0], "gripper": 0.3}
    st = {"left_ee": z, "right_ee": {**z, "position": [0.1, 0.2, 0.3]}}
    r = move_response("e", st, {"left_ee": {"position": [0.5, 0, 0.2], "rotation_rpy": [0, 0.2, 0.4]}})["actions"]
    assert len(r["left_ee"]["position"]) == HORIZON
    assert np.allclose(r["left_ee"]["position"][-1], [0.5, 0, 0.2]) and np.allclose(r["left_ee"]["rotation_rpy"][-1], [0, 0.2, 0.4])
    assert set(r["left_ee"]["gripper"]) == {0.3} and all(p == [0.1, 0.2, 0.3] for p in r["right_ee"]["position"])
