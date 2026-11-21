"""Scripted no-contact look-around: wrist camera takes oblique views of a parcel, each gated by IK on the URDF chain. numpy + cv2.
Frame = robot root (+x fwd, +y left, +z up); arm frame = root shifted by the arm base (0, +-arm_y, arm_z)."""
import cv2
import numpy as np

from labelloop.pose import interp_rpy, rpy_to_matrix

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


def ik(pos, rpy, tries=8, q0=None):
    """Damped least squares from several seeds (first q0, e.g. the previous waypoint, else zeros): (q, position error m, rotation error ||R - Rt||_F), best of `tries`."""
    Rt, rng, best = rpy_to_matrix(rpy), np.random.default_rng(0), None

    def res(q):
        p, R = fk(q)
        return np.concatenate([p - pos, 0.3 * (R - Rt).ravel()])

    for k in range(tries):
        q = (np.zeros(6) if q0 is None else np.asarray(q0, float)) if k == 0 else rng.uniform(LO, HI)
        for _ in range(60):
            r, J = res(q), np.empty((12, 6))
            if np.linalg.norm(r) < 1e-4:
                break
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


def table_to_pixel(p, wh=(1280, 720)):
    """(u, v) in a wh-sized head image of root-frame point p: the inverse of pixel_to_table (any height)."""
    c = HEAD_R.T @ (np.asarray(p, float) - HEAD_POS)
    return (CX + FX * c[0] / -c[2]) * wh[0] / 1280, (CY - FY * c[1] / -c[2]) * wh[1] / 720


def footprint(box, wh, cfg):
    """Parcel half-width (m) from the head-mask bbox. CONVENTION: box = (x0, y0, x1, y1) in pixels of the FULL wh-sized head image
    (what scene.changed_box returns), never crop-relative. The corners are projected at the aim-point height and the half-width is
    half the largest corner-to-corner distance (conservative: the front face and perspective inflate it); cfg["hw"] when the box is
    unknown or implausible (< 0.04 or > 0.25 m)."""
    if box is None:
        return cfg["hw"]
    c = [pixel_to_table(u, v, wh, TABLE_Z + cfg["target_h"])[:2] for u in (box[0], box[2]) for v in (box[1], box[3])]
    hw = max(np.linalg.norm(a - b) for a in c for b in c) / 2
    return float(hw) if 0.04 <= hw <= 0.25 else cfg["hw"]


def frame(arm, cfg):
    """Arm base in root coordinates."""
    return np.array([0.0, SIGN[arm] * cfg["arm_y"], cfg["arm_z"]])


def to_arm(arm, p, cfg):
    """Root-frame point -> the arm frame, which is also the frame of the WALLX state / targets (absolute, per-arm base)."""
    return np.asarray(p, float) - frame(arm, cfg)


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


def reach(arm, tip, pitch, cfg, q0=None):
    """(ee in arm frame, rpy, q, ik_err) holding the fingertips at root point `tip` at `pitch` (90 deg = pointing down), yaw toward
    the point; None if IK fails, a joint limit binds or the tip is under TABLE_Z + 0.01. Never returns an infeasible pose."""
    d = to_arm(arm, tip, cfg)
    rpy = np.array([0.0, pitch, np.arctan2(d[1], d[0])])
    ee = d - rpy_to_matrix(rpy) @ TCP
    q, e, er = ik(ee, rpy, q0=q0)
    if e < 0.01 and er < 0.05 and tip[2] >= TABLE_Z + 0.01 and np.all(q >= LO - 1e-6) and np.all(q <= HI + 1e-6):
        return ee, rpy, q, e


def commanded_ok(arm, poses, cfg, ok_tip=None, step=0.02):
    """True if the arm can drive `poses` [(ee, rpy)] (arm frame, the first = where it is now) the way move_response does: every leg
    sampled (EE position linear, rpy geodesic) every <= step m / 0.15 rad; each sample needs warm-started IK (no random restarts, so
    joint-continuous: jump <= 0.5 rad), joint limits, fingertip >= TABLE_Z + 0.01 and ok_tip(leg, tip) (default: always).
    ponytail: fingertip geometry only, no arm-link / bin / other-arm collision model."""
    q = ik(*poses[0], q0=np.zeros(6))[0]  # the start pose only seeds the chain
    for k, ((p0, r0), (p1, r1)) in enumerate(zip(poses[:-1], poses[1:])):
        n = max(1, int(np.ceil(max(np.linalg.norm(p1 - p0) / step, np.linalg.norm(np.asarray(r1) - r0) / 0.15))))
        for j in range(1, n + 1):
            p, r = p0 + j / n * (p1 - p0), interp_rpy(r0, r1, j / n)
            tip = tip_of(arm, p, r, cfg)
            if tip[2] < TABLE_Z + 0.01 or (ok_tip and not ok_tip(k, tip)):
                return False
            q2, e, er = ik(p, r, tries=1, q0=q)
            if e > 0.01 or er > 0.05 or np.any(q2 < LO - 1e-6) or np.any(q2 > HI + 1e-6) or np.max(np.abs(q2 - q)) > 0.5:
                return False
            q = q2
    return True


def fit_parcel(mask, wh, cfg):
    """Oriented footprint of the parcel from the head-camera change mask (full wh-sized image): minAreaRect of the largest blob's convex hull projected
    to the table at the top-face height (TABLE_Z + parcel_h). -> {"c": centre xy, "axes": 2x2 unit rows, "half": face
    half-extents along them, "yaw"} or None (refuse to push) if implausible: a side outside [fit_min, fit_max] m, aspect > fit_aspect,
    or the blob fills < fit_fill of its rectangle. The blob also holds the front face, so the footprint is a little long toward the camera."""
    cs, _ = cv2.findContours(np.asarray(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    cnt = max(cs, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    tbl = np.array([pixel_to_table(u, v, wh, TABLE_Z + cfg["parcel_h"])[:2] for u, v in hull[:, 0]], np.float32)  # rectify first: perspective skews pixel rects
    rect = cv2.minAreaRect(tbl)
    q = cv2.boxPoints(rect)
    e0, e1 = q[1] - q[0], q[2] - q[1]
    L = np.array([np.linalg.norm(e0), np.linalg.norm(e1)])
    fill = cv2.contourArea(cnt) / max(cv2.contourArea(hull), 1) * cv2.contourArea(tbl) / max(L[0] * L[1], 1e-9)  # blob solidity x hull / rect
    if L.min() < cfg["fit_min"] or L.max() > cfg["fit_max"] or L.max() > cfg["fit_aspect"] * L.min() or fill < cfg["fit_fill"]:
        return None
    a0 = e0 / L[0]
    axes = np.array([a0, np.sign(a0[0] * e1[1] - a0[1] * e1[0]) * np.array([-a0[1], a0[0]])])
    return {"c": q.mean(0), "axes": axes, "half": L / 2, "yaw": float(np.arctan2(axes[0][1], axes[0][0]))}


def spin_plan(fit, cfg, sigma=None, home=None):
    """Face-normal push that yaws the fitted parcel: fingertips (gripper closed) start spin.out m outside a face facing the arm, at
    spin.reach_frac x the face half-length from the face centre (near a corner, so the normal force makes a yaw torque), go in to the
    face, push spin.push_len inward along the normal, retract along it. sigma = +-1 the wanted yaw direction (None: either, the plan says
    which it took, so later tries keep rotating the same way). Tips (root): above start, start, contact, end, start, above start; then
    scripted home. Fingertips at TABLE_Z + clip(0.3 parcel_h, .015, .03). First (arm nearest, face most facing the arm, pitch steepest)
    whose whole commanded trajectory from `home` ({arm: (ee, rpy)}, default q = 0) through the return home passes commanded_ok, with the
    tips outside the parcel box except the contact. -> {"arm","pitch","tips","wps","contact","normal","tip_z","sigma"} or None."""
    c, ax, hf = fit["c"], fit["axes"], fit["half"]
    zt, za = TABLE_Z + np.clip(0.3 * cfg["parcel_h"], 0.015, 0.03), TABLE_Z + cfg["approach_h"]
    top = TABLE_Z + cfg["parcel_h"]

    def inside(tip, m):
        loc = ax @ (tip[:2] - c)
        return abs(loc[0]) < hf[0] + m and abs(loc[1]) < hf[1] + m and tip[2] < top + m

    def ok_tip(k, tip):  # legs: 0 home->above, 1 down, 2 final approach, 3 push, 4 retract, 5 up, 6 home
        return True if k in (3, 4) else not inside(tip, -0.005 if k == 2 else 0.01)

    for arm in sorted(SIGN, key=lambda a: abs(c[1] - SIGN[a] * cfg["arm_y"])):
        d = frame(arm, cfg)[:2] - c
        d /= np.linalg.norm(d)
        h = (np.array(fk(np.zeros(6))[0]), np.zeros(3)) if home is None else home[arm]
        for n, i in sorted(((s * ax[i], i) for i in (0, 1) for s in (1, -1)), key=lambda f: -f[0] @ d):
            if n @ d < cfg["face_min_dot"]:
                break  # the remaining faces turn away from this arm
            t, l = ax[1 - i], hf[1 - i]
            for sg in (sigma,) if sigma else (1, -1):
                e = -sg * np.sign(t[0] * n[1] - t[1] * n[0])  # end of the face whose inward push turns the parcel by sg
                P = c + hf[i] * n + e * cfg["reach_frac"] * l * t
                S, E = P + cfg["out"] * n, P - cfg["push_len"] * n
                tips = np.array([(*S, za), (*S, zt), (*P, zt), (*E, zt), (*S, zt), (*S, za)])
                for pitch in np.radians((90, 75, 60, 45)):
                    rs = [reach(arm, tp, pitch, cfg) for tp in tips]
                    if all(rs):
                        wps = [r[:2] for r in rs]
                        if commanded_ok(arm, [h, *wps, h], cfg, ok_tip):
                            return {"arm": arm, "pitch": float(pitch), "tips": tips, "wps": wps, "contact": P, "normal": n,
                                    "tip_z": float(zt), "sigma": int(sg)}
