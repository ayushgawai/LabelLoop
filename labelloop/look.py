"""Scripted no-contact look-around: wrist camera takes oblique views of a parcel, each gated by IK on the URDF chain. numpy only.
Frame = robot root (+x fwd, +y left, +z up); arm frame = root shifted by the arm base (0, +-arm_y, arm_z)."""
import numpy as np

from labelloop.pose import rpy_to_matrix

TABLE_Z, SIGN = 0.015, {"left_ee": 1.0, "right_ee": -1.0}  # arm base = root + (0, SIGN * look.arm_y, look.arm_z)
HEAD_POS = np.array([0.084, 0.0, 0.517])
# OpenGL camera (looks along -Z, +Y up). The brief's tuple (0.21263, -0.21263, -0.67438, 0.67438) only meets its own
# sanity check (forward (0.574, 0, -0.819), image-right = -y) read as (x, y, z, w).
_x, _y, _z, _w = 0.21263, -0.21263, -0.67438, 0.67438
HEAD_R = np.array([[1 - 2 * (_y**2 + _z**2), 2 * (_x * _y - _z * _w), 2 * (_x * _z + _y * _w)],
                   [2 * (_x * _y + _z * _w), 1 - 2 * (_x**2 + _z**2), 2 * (_y * _z - _x * _w)],
                   [2 * (_x * _z - _y * _w), 2 * (_y * _z + _x * _w), 1 - 2 * (_x**2 + _y**2)]])
FX, FY, CX, CY = 2.97 / 5.54 * 1280, 2.97 / 3.09 * 720, 640, 360
CAM_OFF, CAM_TILT = np.array([0.074088, 0, 0.097993]), 0.43633  # wrist camera in the EE frame (URDF)
TCP = np.array([0.217, 0, -0.01])  # G-gripper fingertips in the EE frame (stock ex001_6r.py)
# One arm, from ex001_6r_legacy.urdf: (joint origin xyz, axis) per joint 1-6, then the limits (rad).
_AX = (2, 1, 1, 1, 2, 0)
_ORG = np.array([(0, 0, 0.0613), (0.025, 0, 0.0542), (-0.26, 0, 0), (0.2315, 0, 0.075), (0.0825, 0, 0.085), (0.0395, 0, -0.0945)])
LO = np.array([-2.7925, 0, -2.7925, -1.5708, -1.5708, -1.9199])
HI = np.array([2.7925, 3.6652, 0, 1.5708, 1.5708, 1.9199])


def _rot(ax, a):
    c, s, R = np.cos(a), np.sin(a), np.eye(3)
    i, j = [(1, 2), (2, 0), (0, 1)][ax]
    R[i, i] = R[j, j] = c
    R[i, j], R[j, i] = -s, s
    return R


def fk(q):
    """(position, rotation) of the EE in the arm frame for joint angles q."""
    p, R = np.zeros(3), np.eye(3)
    for o, ax, a in zip(_ORG, _AX, q):
        p, R = p + R @ o, R @ _rot(ax, a)
    return p, R


def ik(pos, rpy, tries=8):
    """Damped least squares from several seeds: (q, position error m, rotation error ||R - Rt||_F), best of `tries`."""
    Rt, rng, best = rpy_to_matrix(rpy), np.random.default_rng(0), None

    def res(q):
        p, R = fk(q)
        return np.concatenate([p - pos, 0.3 * (R - Rt).ravel()])

    for k in range(tries):
        q = np.zeros(6) if k == 0 else rng.uniform(LO, HI)
        for _ in range(60):
            r, J = res(q), np.empty((12, 6))
            for j in range(6):
                d = np.zeros(6)
                d[j] = 1e-6
                J[:, j] = (res(q + d) - r) / 1e-6
            q = np.clip(q - np.linalg.solve(J.T @ J + 1e-4 * np.eye(6), J.T @ r), LO, HI)
        p, R = fk(q)
        e = (np.linalg.norm(p - pos), np.linalg.norm(R - Rt))
        if best is None or sum(e) < sum(best[1:]):
            best = (q, *e)
        if e[0] < 1e-3 and e[1] < 0.02:
            break
    return best




def pixel_to_table(u, v, wh=(1280, 720), z=TABLE_Z):
    """Root-frame point where the head camera ray through pixel (u, v) of a wh-sized image meets the plane height z."""
    u, v = u * 1280 / wh[0], v * 720 / wh[1]
    d = HEAD_R @ [(u - CX) / FX, -(v - CY) / FY, -1.0]
    return HEAD_POS + (z - HEAD_POS[2]) / d[2] * d


def footprint(box, wh, cfg):
    """Parcel half-width from the head-mask bbox (x0, y0, x1, y1): corners projected at the aim-point height, conservative
    (the front face and perspective inflate it); look.hw when unknown. Clipped to 0.06-0.25 m."""
    if box is None:
        return cfg["hw"]
    c = np.array([pixel_to_table(u, v, wh, TABLE_Z + cfg["target_h"])[:2] for u in (box[0], box[2]) for v in (box[1], box[3])])
    return float(np.clip(np.ptp(c, axis=0).max() / 2, 0.06, 0.25))


def frame(arm, cfg):
    """Arm base in root coordinates."""
    return np.array([0.0, SIGN[arm] * cfg["arm_y"], cfg["arm_z"]])


def box_dist(p, c, hw, hz):
    """Distance from point p to the axis-aligned box centred c with half extents (hw, hw, hz)."""
    return float(np.linalg.norm(np.maximum(np.abs(np.asarray(p) - c) - [hw, hw, hz], 0)))


def tip_of(arm, ee, rpy, cfg):
    """Fingertip point in root coordinates for an EE pose in the arm frame."""
    return np.asarray(ee) + rpy_to_matrix(rpy) @ TCP + frame(arm, cfg)


def _view(arm, phi, p, hw, cfg):
    """(arm, ee in arm frame, rpy, info) for the wrist camera at azimuth phi from the parcel, or None if IK/clearance fails.
    The optical axis aims `aim_down` below the target so the label lands in the upper image half, clear of the fingers."""
    c = np.array([p[0], p[1], TABLE_Z + cfg["target_h"]])  # box centre; its half height is target_h
    side = np.array([np.cos(phi), np.sin(phi)])
    cam = np.array([*(p[:2] + (hw + cfg["standoff"]) * side), TABLE_Z + cfg["cam_h"]])
    a = c - cam
    th = np.arctan2(-a[2], np.hypot(a[0], a[1])) + cfg["aim_down"]  # axis pitch below horizontal
    rpy = np.array([0.0, th - CAM_TILT, np.arctan2(a[1], a[0])])
    R = rpy_to_matrix(rpy)
    ee = cam - R @ CAM_OFF - frame(arm, cfg)
    q, e, er = ik(ee, rpy)
    tip = tip_of(arm, ee, rpy, cfg)
    clear = min(box_dist(tip, c, hw, cfg["target_h"]), tip[2] - TABLE_Z)  # to the parcel box; to the table
    if e > 0.01 or er > 0.05 or np.any(q < LO - 1e-6) or np.any(q > HI + 1e-6) or clear < cfg["finger_clear"] \
            or tip[2] - TABLE_Z < cfg["finger_clear"]:
        return None
    return arm, ee, rpy, {"ik_err": e, "rot_err": er, "clear": box_dist(tip, c, hw, cfg["target_h"]), "tip_z": tip[2], "q": q}


def views(p, cfg, hw=None):
    """[(arm, ee_in_arm_frame, ee_rpy, info)] views of a parcel at root table point p with half-width hw (default look.hw), from the
    near half-circle (azimuth 90-270 deg off +x, the far side is dropped). Every candidate (either arm) must pass IK + clearance;
    survivors are picked greedily, most frontal first, at least look.sep_deg apart (best clearance wins when both arms can), then grouped by arm.
    ponytail: an arm can only look about 90 deg either side of its own azimuth, so mostly the arm opposite the parcel (camera on the
    inner side) survives; an empty list is a valid answer."""
    hw = cfg["hw"] if hw is None else hw
    p, ok = np.asarray(p, float), []
    for deg in sorted(range(90, 271, 15), key=lambda d: abs(d - 180)):
        c = [v for v in (_view(a, np.radians(deg), p, hw, cfg) for a in ("left_ee", "right_ee")) if v]
        if c and all(abs(deg - d) >= cfg["sep_deg"] for d, _ in ok):
            ok.append((deg, max(c, key=lambda v: v[3]["clear"])))
    return sorted((v for _, v in ok), key=lambda v: v[0])


def point_pose(arm, tip, cfg):
    """(ee in arm frame, rpy, ik_err) holding the fingertips at root point `tip`, gripper pointing down if IK allows, else the
    least-tilted feasible pose, else the lowest residual. For the point_test frame check."""
    best = None
    for pitch in np.radians((90, 75, 60, 45, 30, 15, 0)):
        yaw = np.arctan2(tip[1] - frame(arm, cfg)[1], tip[0])
        rpy = np.array([0.0, pitch, yaw])
        ee = np.asarray(tip) - rpy_to_matrix(rpy) @ TCP - frame(arm, cfg)
        _, e, er = ik(ee, rpy)
        if e < 0.01 and er < 0.05:
            return ee, rpy, e
        if best is None or e + er < best[3]:
            best = (ee, rpy, e, e + er)
    return best[:3]
