"""Scripted no-contact look-around: wrist camera visits each side of a parcel. numpy only. Frame = robot root (+x fwd, +y left, +z up)."""
import numpy as np

from labelloop.pose import rpy_to_matrix

TABLE_Z, ARM_Y = 0.015, {"left_ee": 0.266, "right_ee": -0.266}  # arm base = root + (0, ARM_Y, 0)
HEAD_POS = np.array([0.084, 0.0, 0.517])
# OpenGL camera (looks along -Z, +Y up). The brief's tuple (0.21263, -0.21263, -0.67438, 0.67438) only meets its own
# sanity check (forward (0.574, 0, -0.819), image-right = -y) read as (x, y, z, w).
_x, _y, _z, _w = 0.21263, -0.21263, -0.67438, 0.67438
HEAD_R = np.array([[1 - 2 * (_y**2 + _z**2), 2 * (_x * _y - _z * _w), 2 * (_x * _z + _y * _w)],
                   [2 * (_x * _y + _z * _w), 1 - 2 * (_x**2 + _z**2), 2 * (_y * _z - _x * _w)],
                   [2 * (_x * _z - _y * _w), 2 * (_y * _z + _x * _w), 1 - 2 * (_x**2 + _y**2)]])
FX, FY, CX, CY = 2.97 / 5.54 * 1280, 2.97 / 3.09 * 720, 640, 360
CAM_OFF, CAM_TILT = np.array([0.074088, 0, 0.097993]), 0.43633  # wrist camera in the EE frame (URDF)


def pixel_to_table(u, v, wh=(1280, 720)):
    """Root-frame point where the head camera ray through pixel (u, v) of a wh-sized image meets the tabletop."""
    u, v = u * 1280 / wh[0], v * 720 / wh[1]
    d = HEAD_R @ [(u - CX) / FX, -(v - CY) / FY, -1.0]
    return HEAD_POS + (TABLE_Z - HEAD_POS[2]) / d[2] * d


def _view(arm, T, side, cfg):
    """EE pose (arm frame) putting the wrist camera `distance` from T along `side` (unit xy), `height` over the table; None if unreachable."""
    h = cfg["height"]
    for _ in range(4):  # raise the camera until the EE clears the table + 0.03
        cam = np.array([*(T[:2] + cfg["distance"] * side), TABLE_Z + h])
        a = T - cam
        yaw, th = np.arctan2(a[1], a[0]), np.arctan2(-a[2], np.hypot(a[0], a[1]))  # camera axis: down th
        pitch = th - CAM_TILT
        rpy = np.array([0.0, pitch, yaw])
        ee = cam - rpy_to_matrix(rpy) @ CAM_OFF
        if ee[2] >= TABLE_Z + 0.03 - 1e-4:
            break
        h += TABLE_Z + 0.03 - ee[2]
    ee[1] -= ARM_Y[arm]
    if ee[2] < TABLE_Z + 0.03 - 1e-3 or np.linalg.norm(ee) > cfg["reach"]:
        return None
    return arm, ee, rpy


def views(p, cfg):
    """[(arm, ee_position_in_arm_frame, ee_rpy)] from the left, near, far and right of a parcel at root point p (unreachable ones skipped)."""
    T = np.asarray(p, float) + [0, 0, cfg["target_h"]]
    sides = (("left_ee", (0, 1)), ("left_ee", (-1, 0)), ("left_ee", (1, 0)), ("right_ee", (0, -1)))
    return [v for v in (_view(a, T, np.array(s, float), cfg) for a, s in sides) if v]
