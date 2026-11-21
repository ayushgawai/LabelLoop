"""Scripted no-contact look-around: wrist camera takes oblique views of a parcel, each gated by IK on the URDF chain. numpy + cv2.
Frame = robot root (+x fwd, +y left, +z up); arm frame = root shifted by the arm base (0, +-arm_y, arm_z)."""
import cv2
import numpy as np

from labelloop.pose import interp_rpy, matrix_to_rpy, rpy_to_matrix

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
    c, s = np.cos(a), np.sin(a)
    return np.array(((1, 0, 0), (0, c, -s), (0, s, c)) if ax == 0 else ((c, 0, s), (0, 1, 0), (-s, 0, c)) if ax == 1 else ((c, -s, 0), (s, c, 0), (0, 0, 1)))


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
        """(residual, Jacobian): analytic, joint j rotates everything after it about its world axis z_j through its origin p_j."""
        p, R, zs, ps = np.zeros(3), np.eye(3), [], []
        for o, ax, a in zip(_ORG, _AX, q):
            p = p + R @ o
            zs.append(R[:, ax])
            ps.append(p)
            R = R @ _rot(ax, a)
        J = np.empty((12, 6))
        for j, (z, pj) in enumerate(zip(zs, ps)):
            J[:3, j] = np.cross(z, p - pj)
            J[3:, j] = 0.3 * (np.array(((0, -z[2], z[1]), (z[2], 0, -z[0]), (-z[1], z[0], 0))) @ R).ravel()
        return np.concatenate([p - pos, 0.3 * (R - Rt).ravel()]), J

    for k in range(tries):
        q = (np.zeros(6) if q0 is None else np.asarray(q0, float)) if k == 0 else rng.uniform(LO, HI)
        for _ in range(60):
            r, J = res(q)
            if np.linalg.norm(r) < 1e-4:
                break
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


def reach(arm, tip, pitch, cfg, q0=None, yaw=None):
    """(ee in arm frame, rpy, q, ik_err) holding the fingertips at root point `tip` at `pitch` (90 deg = pointing down), yaw toward
    the point (or the given yaw); None if IK fails, a joint limit binds or the tip is under TABLE_Z + 0.01. Never returns an infeasible pose."""
    d = to_arm(arm, tip, cfg)
    rpy = np.array([0.0, pitch, np.arctan2(d[1], d[0]) if yaw is None else yaw])
    ee = d - rpy_to_matrix(rpy) @ TCP
    q, e, er = ik(ee, rpy, q0=q0)
    if e < 0.01 and er < 0.05 and tip[2] >= TABLE_Z + 0.01 and np.all(q >= LO - 1e-6) and np.all(q <= HI + 1e-6):
        return ee, rpy, q, e


def commanded_ok(arm, poses, cfg, ok_tip=None, step=0.02, ok_pose=None, q0=None, ret_q=False):
    """True if the arm can drive `poses` [(ee, rpy)] (arm frame, the first = where it is now) the way move_response does: every leg
    sampled (EE position linear, rpy geodesic) every <= step m / 0.15 rad; each sample needs warm-started IK (no random restarts, so
    joint-continuous: jump <= 0.5 rad), joint limits, fingertip >= TABLE_Z + 0.01, ok_tip(leg, tip) and ok_pose(leg, ee, rpy) (default: always).
    q0 = the joints the arm is in at poses[0] (default: IK from zeros); ret_q -> the final joints instead of True (None on failure).
    ponytail: fingertip geometry only, no arm-link / bin / other-arm collision model."""
    q = ik(*poses[0], q0=np.zeros(6))[0] if q0 is None else np.asarray(q0, float)  # the start pose only seeds the chain
    for k, ((p0, r0), (p1, r1)) in enumerate(zip(poses[:-1], poses[1:])):
        n = max(1, int(np.ceil(max(np.linalg.norm(p1 - p0) / step, np.linalg.norm(np.asarray(r1) - r0) / 0.15))))
        for j in range(1, n + 1):
            p, r = p0 + j / n * (p1 - p0), interp_rpy(r0, r1, j / n)
            tip = tip_of(arm, p, r, cfg)
            if tip[2] < TABLE_Z + 0.01 or (ok_tip and not ok_tip(k, tip)) or (ok_pose and not ok_pose(k, p, r)):
                return None if ret_q else False
            q2, e, er = ik(p, r, tries=1, q0=q)
            if e > 0.01 or er > 0.05 or np.any(q2 < LO - 1e-6) or np.any(q2 > HI + 1e-6) or np.max(np.abs(q2 - q)) > 0.5:
                return None if ret_q else False
            q = q2
    return q if ret_q else True


def project(P, wh=(1280, 720)):
    """Pixels (N, 2) of root-frame points P (N, 3) in a wh-sized head image (vectorised table_to_pixel)."""
    c = (np.asarray(P, float) - HEAD_POS) @ HEAD_R
    return np.stack([(CX + FX * c[:, 0] / -c[:, 2]) * wh[0] / 1280, (CY - FY * c[:, 1] / -c[:, 2]) * wh[1] / 720], 1)


def box_corners(c, yaw, hx, hy, h, dz=0.0):
    """The 8 root-frame corners of a box standing on the table: centre xy c, yaw, half extents hx (along yaw) / hy, height h, lifted by dz."""
    cs, sn = np.cos(yaw), np.sin(yaw)
    xy = np.asarray(c, float) + np.array([[sx * hx * cs - sy * hy * sn, sx * hx * sn + sy * hy * cs] for sx in (-1, 1) for sy in (-1, 1)])
    return np.array([(*p, TABLE_Z + dz + z) for z in (0.0, h) for p in xy])


def box_mask(c, yaw, hx, hy, h, wh=(1280, 720), dz=0.0, out=None, off=(0, 0)):
    """Head-image silhouette (uint8 255) of a box: convex hull of its 8 projected corners, filled. out/off: draw into a crop (pixel offset off)."""
    px = project(box_corners(c, yaw, hx, hy, h, dz), wh) - off
    m = np.zeros(wh[::-1], np.uint8) if out is None else out
    cv2.fillConvexPoly(m, cv2.convexHull(np.round(px * 4).astype(np.int32)), 255, cv2.LINE_8, 2)
    return m


def fit_parcel(mask, wh, cfg):
    """3D box fit to the head-camera change mask (full wh-sized image): centre xy on the table, yaw, half extents and height of a box
    resting on the table whose projected silhouette (look.box_mask) has the best IoU with the largest blob. Starts: the blob's hull rectified
    at several top heights (minAreaRect); each start is refined by a shrinking pattern search over (x, y, yaw, hx, hy, h). The silhouette has the top
    AND the visible walls, so (unlike a one-height projection) height and footprint are told apart. -> {"c", "axes" (2x2 unit rows, axes[0]
    the long side), "half" (footprint), "yaw", "h" (height m), "iou"} or None (refuse) if implausible: a side outside [fit_min, fit_max] m,
    aspect > fit_aspect, height outside [fit_hmin, fit_hmax] or IoU < fit_iou (default 0.8)."""
    cs, _ = cv2.findContours(np.asarray(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    cnt = max(cs, key=cv2.contourArea)
    x0, y0, w, h_ = cv2.boundingRect(cnt)
    pad, off = 25, np.array([x0 - 25, y0 - 25])
    tgt = np.zeros((h_ + 2 * pad, w + 2 * pad), np.uint8)
    cv2.drawContours(tgt, [cnt - off], -1, 255, -1)
    tn, buf = cv2.countNonZero(tgt), np.zeros(tgt.shape, np.uint8)
    hmin, hmax = cfg.get("fit_hmin", 0.015), cfg.get("fit_hmax", 0.22)

    def iou(x):
        if not (hmin <= x[5] <= hmax and x[3] > 0.01 and x[4] > 0.01):
            return 0.0
        buf[:] = 0
        box_mask(x[:2], x[2], x[3], x[4], x[5], wh, out=buf, off=off)
        a, i = cv2.countNonZero(buf), cv2.countNonZero(cv2.bitwise_and(buf, tgt))
        return i / max(a + tn - i, 1)

    def refine(x, f, free):
        """Pattern search on (x, y, yaw, hx, hy, h): shrinking steps; moves = shift, yaw, grow, single-end (the silhouette has ridges along the view)."""
        step = 0.01
        while step > 0.0004:
            moved = False
            u = np.array([np.cos(x[2]), np.sin(x[2])])
            ax = (u, np.array([-u[1], u[0]]))
            mv = [(*(sg * step * a), 0, 0, 0, 0) for a in np.eye(2) for sg in (1, -1)] + [(0, 0, sg * 10 * step, 0, 0, 0) for sg in (1, -1)]
            mv += [(0, 0, 0, sg * step, 0, 0) for sg in (1, -1)] + [(0, 0, 0, 0, sg * step, 0) for sg in (1, -1)]
            mv += [(*(sg * step * ax[k]), 0, *(g * step * np.eye(2)[k]), 0) for k in (0, 1) for sg in (1, -1) for g in (1, -1)]
            mv += [(0, 0, 0, 0, 0, sg * step) for sg in (1, -1)] * free
            for d in mv:
                y = x + d
                fy = iou(y)
                if fy > f + 1e-9:
                    x, f, moved = y, fy, True
            if not moved:
                step *= 0.5
        return x, f

    hull, starts = cv2.convexHull(cnt), []
    for H in np.arange(0.02, 0.2, 0.01):  # coarse height grid: with h fixed, the far and near silhouette edges fix the footprint
        tbl = np.array([pixel_to_table(u, v, wh, TABLE_Z + H)[:2] for u, v in hull[:, 0]], np.float32)
        (cx, cy), (a, b), ang = cv2.minAreaRect(tbl)
        x = np.array([cx, cy, np.radians(ang), a / 2, b / 2, H])
        x, f = refine(x, iou(x), 0)
        starts.append(refine(x, f, 0)[::-1])  # (a second pass restarts the big steps: the ridge needs it)
    f, x = max((refine(*refine(x, f, 1), 1)[::-1] for f, x in sorted(starts, key=lambda s: -s[0])[:6]), key=lambda s: s[0])  # the 6 best heights, all free
    rng = np.random.default_rng(0)  # the optimum sits in a narrow curved valley (height vs footprint along the view): random polish across it
    for sd in (0.004, 0.002, 0.001):
        for _ in range(250):
            y = x + rng.normal(0, 1, 6) * [sd, sd, 2.5 * sd, sd, sd, 1.5 * sd]
            fy = iou(y)
            if fy > f:
                x, f = y, fy
    if f < 0.3:
        return None
    hx, hy, yaw = x[3], x[4], x[2]
    if hy > hx:
        hx, hy, yaw = hy, hx, yaw + np.pi / 2
    yaw = float((yaw + np.pi / 2) % np.pi - np.pi / 2)
    if hx - hy < 0.004:  # (nearly) square: the yaw is only defined mod 90 deg, take the one nearest 0
        yaw = float((yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4)
    a0 = np.array([np.cos(yaw), np.sin(yaw)])
    L = 2 * np.array([hx, hy])
    if L.min() < cfg["fit_min"] or L.max() > cfg["fit_max"] or L.max() > cfg["fit_aspect"] * L.min() or f < cfg.get("fit_iou", 0.8):
        return None
    return {"c": x[:2].copy(), "axes": np.array([a0, [-a0[1], a0[0]]]), "half": np.array([hx, hy]), "yaw": yaw, "h": float(x[5]), "iou": float(f)}


def spin_plan(fit, cfg, sigma=None, home=None):
    """Face-normal push that yaws the fitted parcel: fingertips (gripper closed) start spin.out m outside a face facing the arm, at
    spin.reach_frac x the face half-length from the face centre (near a corner, so the normal force makes a yaw torque), go in to the
    face, push spin.push_len inward along the normal, retract along it. sigma = +-1 the wanted yaw direction (None: either, the plan says
    which it took, so later tries keep rotating the same way). Tips (root): above start, start, contact, end, start, above start; then
    scripted home. Fingertips at TABLE_Z + clip(0.3 parcel_h, .015, .03). First (arm nearest, face most facing the arm, pitch steepest)
    whose whole commanded trajectory from `home` ({arm: (ee, rpy)}, default q = 0) through the return home passes commanded_ok, with the
    tips outside the parcel box except the contact. -> {"arm","pitch","tips","wps","contact","normal","tip_z","sigma"} or None."""
    c, ax, hf = fit["c"], fit["axes"], fit["half"]
    zt, za = TABLE_Z + np.clip(0.3 * fit.get("h", cfg["parcel_h"]), 0.015, 0.03), TABLE_Z + max(cfg["approach_h"], fit.get("h", cfg["parcel_h"]) + 0.06)
    top = TABLE_Z + fit.get("h", cfg["parcel_h"])

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


def _home(arm, home):
    return (np.array(fk(np.zeros(6))[0]), np.zeros(3)) if home is None else home[arm]


def too_wide(fit, cfg):
    """True if the parcel's narrow side does not fit between the fingers (grip_max minus grip_margin)."""
    return 2 * fit["half"].min() > cfg["grip_max"] - cfg["grip_margin"]


def grasp_plan(arm, fit, cfg, home=None):
    """Can `arm` GRASP the fitted parcel where it lies? Pregrasp (fingertips straight above the centre at approach_h) + grasp (fingertips at
    the top-face height minus grasp_depth), gripper pointing down within grasp_pitches_deg, yaw so the fingers close across the NARROW
    side (ponytail: jaw axis = EE y, so the tool y axis is set parallel to the narrow axis, either sign; CONFIRM on the robot), the narrow
    side <= grip_max - grip_margin, and home -> pregrasp -> grasp passing commanded_ok. -> {"arm","pitch","yaw","pre","grasp"} or None."""
    if too_wide(fit, cfg):
        return None
    n, c = fit["axes"][int(np.argmin(fit["half"]))], fit["c"]
    ph = fit.get("h", cfg["parcel_h"])
    pre, grasp = (*c, TABLE_Z + max(cfg["approach_h"], ph + 0.06)), (*c, TABLE_Z + ph - cfg["grasp_depth"])
    y0 = np.arctan2(-n[0], n[1])  # tool y = (-sin yaw, cos yaw) at pitch 90 (horizontal for any pitch, roll 0)
    for pitch in np.radians(cfg["grasp_pitches_deg"]):
        for yaw in sorted((y0, float(np.angle(np.exp(1j * (y0 + np.pi))))), key=lambda y: abs(y - np.arctan2(c[1] - frame(arm, cfg)[1], c[0]))):  # the two jaw orientations: y0 and y0 + pi
            rs = [reach(arm, t, pitch, cfg, yaw=yaw) for t in (pre, grasp)]
            if all(rs) and commanded_ok(arm, [_home(arm, home), *(r[:2] for r in rs)], cfg):
                return {"arm": arm, "pitch": float(pitch), "yaw": float(yaw), "pre": np.array(pre), "grasp": np.array(grasp)}


def footprint_clear(c, fit, cfg):
    """True if the fitted parcel footprint centred at c stays table_margin inside the table and bin_margin outside every bin (root xy
    boxes [x0, x1, y0, y1] in cfg["table"] / cfg["bins"]); the outline is sampled every ~1 cm."""
    a, h = fit["axes"], fit["half"]
    k = np.linspace(-1, 1, 9)
    pts = []
    for s in (-1, 1):
        pts += [c + s * h[0] * a[0] + t * h[1] * a[1] for t in k] + [c + t * h[0] * a[0] + s * h[1] * a[1] for t in k]
    x0, x1, y0, y1 = cfg["table"]
    m = cfg["table_margin"]
    for p in pts:
        if not (x0 + m <= p[0] <= x1 - m and y0 + m <= p[1] <= y1 - m):
            return False
        for bx0, bx1, by0, by1 in cfg["bins"].values():
            if np.hypot(max(bx0 - p[0], 0, p[0] - bx1), max(by0 - p[1], 0, p[1] - by1)) < cfg["bin_margin"]:
                return False
    return True


def slide_plan(fit, target, delivery, cfg, home=None):
    """Hand-off push by the arm that is NOT `delivery`: a face-normal push through the parcel's centre (contact at the face centre, so no
    yaw torque) on the face opposite the way to `target` (root xy), strokes of up to slide.stroke m along the normal (>= dir_min_dot cos
    to the target direction). None if no face is aligned, the predicted footprint (halfway and at the end) would come within bin_margin of a
    bin / table_margin of the table edge, or no pitch gives a feasible full commanded path (home, above, down, in, push, out, up, home).
    -> {"arm","pitch","tips","wps","contact","normal","dist","tip_z","c_after"}."""
    arm = next(a for a in SIGN if a != delivery)
    c, ax, hf = fit["c"], fit["axes"], fit["half"]
    v = np.asarray(target, float) - c
    L = np.linalg.norm(v)
    if L < 1e-3:
        return None
    d, dist = v / L, min(cfg["stroke"], L)
    n, i = max(((s * ax[i], i) for i in (0, 1) for s in (1, -1)), key=lambda f: -f[0] @ d)  # outward normal of the face behind the parcel
    if -n @ d < cfg["dir_min_dot"] or not all(footprint_clear(c - f * dist * n, fit, cfg) for f in (0.5, 1.0)):
        return None
    ph = fit.get("h", cfg["parcel_h"])
    zt, za, top = TABLE_Z + np.clip(0.3 * ph, 0.015, 0.03), TABLE_Z + max(cfg["approach_h"], ph + 0.06), TABLE_Z + ph

    def inside(tip, m, c=c):
        loc = ax @ (tip[:2] - c)
        return abs(loc[0]) < hf[0] + m and abs(loc[1]) < hf[1] + m and tip[2] < top + m

    def ok_tip(k, tip):  # legs: 0 home->above, 1 down, 2 final approach, 3 push, 4 retract, 5 up, 6 home
        if k in (3, 4):
            return True
        return not inside(tip, -0.005 if k == 2 else 0.01, c - dist * n if k > 4 else c)  # after the push the parcel sits at c_after

    P = c + hf[i] * n
    S, E = P + cfg["out"] * n, P - dist * n
    R = E + cfg["out"] * n
    tips = np.array([(*S, za), (*S, zt), (*P, zt), (*E, zt), (*R, zt), (*R, za)])
    h = _home(arm, home)
    for pitch in np.radians((90, 75, 60, 45)):
        rs = [reach(arm, tp, pitch, cfg) for tp in tips]
        if all(rs):
            wps = [r[:2] for r in rs]
            if commanded_ok(arm, [h, *wps, h], cfg, ok_tip):
                return {"arm": arm, "pitch": float(pitch), "tips": tips, "wps": wps, "contact": P, "normal": n, "dist": float(dist),
                        "tip_z": float(zt), "c_after": c - dist * n}


def swipe_plan(fit, cfg, home=None):
    """Reject swipe: fingertips (gripper closed) low behind the parcel push it along -y (the robot's right) toward and past the table's right edge
    (swipe.edge_y), through the centre of its face most opposite to -y (>= dir_min_dot cos), in a stroke of up to swipe.stroke m (to swipe.past m
    beyond the edge). The arm nearest first (then the other: the parcel is re-fitted and pushed on after each stroke, so the arms take turns mid-table);
    first (arm, pitch) whose whole commanded path (home, above, down, in, push, out, up, home) passes commanded_ok with the tips outside the parcel
    except the push. -> {"arm","pitch","tips","wps","contact","normal","dist","tip_z","c_after","off"} (off: the parcel's centre ends past the edge) or None.
    ponytail: fingertip geometry only; the swept arm is not checked against the table edge or the floor."""
    c, ax, hf = fit["c"], fit["axes"], fit["half"]
    d, ph = np.array([0.0, -1.0]), fit.get("h", cfg["parcel_h"])
    n, i = max(((s * ax[i], i) for i in (0, 1) for s in (1, -1)), key=lambda f: -f[0] @ d)
    if -n @ d < cfg["dir_min_dot"]:
        return None
    goal = c[1] - (cfg["edge_y"] - cfg["past"])  # y the centre still has to travel
    if goal <= 0:
        return None
    dist = min(cfg["stroke"], goal / (-n @ d))
    zt, za, top = TABLE_Z + np.clip(0.3 * ph, 0.015, 0.03), TABLE_Z + max(cfg["approach_h"], ph + 0.06), TABLE_Z + ph

    def inside(tip, m, c=c):
        loc = ax @ (tip[:2] - c)
        return abs(loc[0]) < hf[0] + m and abs(loc[1]) < hf[1] + m and tip[2] < top + m

    def ok_tip(k, tip):  # legs: 0 home->above, 1 down, 2 final approach, 3 push, 4 retract, 5 up, 6 home
        return True if k in (3, 4) else not inside(tip, -0.005 if k == 2 else 0.01, c - dist * n if k > 4 else c)

    P = c + hf[i] * n
    S, E = P + cfg["out"] * n, P - dist * n
    R = E + cfg["out"] * n
    tips = np.array([(*S, za), (*S, zt), (*P, zt), (*E, zt), (*R, zt), (*R, za)])
    for arm in sorted(SIGN, key=lambda a: abs(c[1] - SIGN[a] * cfg["arm_y"])):
        h = _home(arm, home)
        for pitch in np.radians((90, 75, 60, 45)):
            rs = [reach(arm, tp, pitch, cfg) for tp in tips]
            if all(rs):
                wps = [r[:2] for r in rs]
                if commanded_ok(arm, [h, *wps, h], cfg, ok_tip):
                    c_after = c - dist * n
                    return {"arm": arm, "pitch": float(pitch), "tips": tips, "wps": wps, "contact": P, "normal": n, "dist": float(dist),
                            "tip_z": float(zt), "c_after": c_after, "off": bool(c_after[1] <= cfg["edge_y"] - cfg["past"] + 1e-6)}


# --- SHOW: grasp the parcel, lift it in front of the head camera and turn it so each face looks at the head --------------------------
FACES = {"z+": (0, 0, 1), "y+": (0, 1, 0), "z-": (0, 0, -1), "y-": (0, -1, 0), "bottom": (1, 0, 0)}  # outward normals in the TOOL frame
# (tool x = EE -> fingertips = down at pitch 90, y = the jaw axis = the narrow side, z = the broad side); the top (-x) was seen at SCAN.


def align(axis, t, phi=0.0):
    """Rotation R with R @ axis = t (unit vectors), then spun by phi about t."""
    a, t = np.asarray(axis, float), np.asarray(t, float)
    v, c = np.cross(a, t), a @ t

    def K(w):
        return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])

    R0 = np.eye(3) + K(v) + K(v) @ K(v) / (1 + c) if c > -1 + 1e-9 else np.diag([-1.0, -1, 1])  # ponytail: antiparallel never happens for these faces
    return (np.eye(3) + np.sin(phi) * K(t) + (1 - np.cos(phi)) * K(t) @ K(t)) @ R0


def box_R(rpy, hold):
    """Orientation of the held parcel's box frame (x = its height axis, down when it lay flat) for tool orientation rpy: the parcel was gripped
    lying flat with the tool at pitch hold[3], so it sits tilted by 90 deg - pitch in the hand."""
    return rpy_to_matrix(rpy) @ _rot(1, np.pi / 2 - hold[3])


def held_corners(arm, ee, rpy, hold, cfg):
    """The 8 root-frame corners of the parcel hanging from the fingertips: hold = (narrow, broad, height, grasp pitch) m / rad; box axes (x height, y narrow,
    z broad) = the columns of box_R, the top grasp_depth above the fingertips, centred over the tool tip. ponytail: ignores the fingers / hand."""
    R = box_R(rpy, hold)
    c = tip_of(arm, ee, rpy, cfg) + (hold[2] / 2 - cfg["grasp_depth"]) * R[:, 0]
    return np.array([c + R @ (np.array([sx, sy, sz]) * [hold[2] / 2, hold[0] / 2, hold[1] / 2]) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


def held_low(arm, ee, rpy, hold, cfg):
    """Lowest z (root) of the held parcel (min over held_corners)."""
    return float(held_corners(arm, ee, rpy, hold, cfg)[:, 2].min())


def _pose_q(ee, rpy):
    q, e, er = ik(ee, rpy, tries=2)
    return q if e < 0.01 and er < 0.05 and np.all(q >= LO - 1e-6) and np.all(q <= HI + 1e-6) else None


def show_plan(arm, fit, cfg, home=None, grasp=None):
    """SHOW plan for `arm` holding the fitted parcel: grasp_plan pregrasp -> grasp (gripper closes) -> straight lift -> up to 4 side-face views
    (+ the bottom if IK allows) at a point P in front of the head camera (fingertips at P: z in show.tip_z, xy in show.points, mirrored for the
    right arm; the first P with show.stop_views views wins, else the one with most). A view = a tool orientation whose face normal
    points at the head within show.vis_min (cos) at show.elevations_deg, spun show.n_spin ways about it; faces are visited greedily by
    smallest rotation, each commanded leg (joint-continuous IK, limits, fingertips above the table, the hanging parcel's lowest point >=
    show.held_clear above it) must pass; an infeasible view is skipped. -> {"arm","grasp","P","hold","fit","faces","missing","calls"} or None;
    calls = [{"ee","rpy","grip","tag"[,"face","q"]}], one per scripted WALLX call: pre, descend, close, testlift (2 cm: the harness checks the grasp), lift, then (move, hold) per view."""
    g = grasp or grasp_plan(arm, fit, cfg, home)
    if not g:
        return None
    hold = (2 * fit["half"].min(), 2 * fit["half"].max(), fit.get("h", cfg["parcel_h"]), g["pitch"])
    h, rg = _home(arm, home), np.array([0.0, g["pitch"], g["yaw"]])
    lift, tlift = np.array(g["grasp"]) + [0, 0, cfg["lift"]], np.array(g["grasp"]) + [0, 0, cfg["test_lift"]]
    pose = lambda tip: (to_arm(arm, tip, cfg) - rpy_to_matrix(rg) @ TCP, rg)
    pre, gr, tl, up = pose(g["pre"]), pose(g["grasp"]), pose(tlift), pose(lift)
    clear = lambda low: (lambda k, p, r: held_low(arm, p, r, hold, cfg) >= low)
    held = lambda k, p, r: held_low(arm, p, r, hold, cfg) >= TABLE_Z - 0.003
    qs = [commanded_ok(arm, [h, pre], cfg, ret_q=True)]  # joints at each driven pose (pre, descend = close, test lift, lift)
    for leg, ok in (([pre, gr], None), ([gr, tl], held), ([tl, up], held)):
        qs.append(None if qs[-1] is None else commanded_ok(arm, leg, cfg, ok_pose=ok, q0=qs[-1], ret_q=True))
    if any(x is None for x in qs):
        return None
    q = qs[-1]
    def views_at(P):
        d = (HEAD_POS - P) / np.linalg.norm(HEAD_POS - P)
        az, cands = np.arctan2(d[1], d[0]), {}
        for f in (cfg["faces_flat"] if hold[2] < cfg["flat_h"] else cfg["faces"]):  # a flat parcel carries a label on its top / bottom only
            ax = FACES[f]
            for el in np.radians(cfg["elevations_deg"]):
                t = np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)])
                for phi in np.arange(cfg["n_spin"]) * 2 * np.pi / cfg["n_spin"]:
                    R = align(ax, t, phi) @ _rot(1, g["pitch"] - np.pi / 2)  # the box frame turned to face the head -> the tool
                    rpy = matrix_to_rpy(R)
                    ee = to_arm(arm, P, cfg) - R @ TCP
                    qp = _pose_q(ee, rpy) if t @ d >= cfg["vis_min"] and held_low(arm, ee, rpy, hold, cfg) >= TABLE_Z + cfg["held_clear"] else None
                    if qp is not None:
                        cands.setdefault(f, []).append(((ee, rpy), qp))
        okc = clear(TABLE_Z + cfg["held_clear"])
        nodes = [(f, c, qc) for f, cs in cands.items() for c, qc in sorted(cs, key=lambda c: -np.min(np.minimum(c[1] - LO, HI - c[1])))[:2]]  # 2 per face: most joint margin
        memo = {}

        def way(a, qa, c, qc, step=0.04):
            """Waypoints [(ee, rpy)] (usually none: the straight leg passes) that make the commanded legs a -> c feasible: 0, 1 or 2 points on the
            straight JOINT-space line between the two IK solutions (joint limits are a box, so it stays inside them), else None."""
            for m in range(3):
                W = [fk(qa + (k + 1) / (m + 1) * (qc - qa)) for k in range(m)]
                W = [(p_, matrix_to_rpy(R_)) for p_, R_ in W]
                if commanded_ok(arm, [a, *W, c], cfg, step=step, ok_pose=okc, q0=qa):
                    return W

        def leg(i, j):
            if (i, j) not in memo:
                a, qa = (up, q) if i < 0 else (nodes[i][1], nodes[i][2])
                memo[i, j] = way(a, qa, nodes[j][1], nodes[j][2])
            return memo[i, j]

        def score(path):
            return len(path), -sum(len(leg(i, j)) for i, j in zip([-1] + path, path))

        best_path = []

        def dfs(path, used):
            nonlocal best_path
            if score(path) > score(best_path):
                best_path = list(path)
            for j, (f, _, _) in enumerate(nodes):
                if len(best_path) < len(cands) and f not in used and leg(path[-1] if path else -1, j) is not None:
                    dfs(path + [j], used | {f})

        dfs([], frozenset())
        cur, qc, views = up, q, []
        for j in best_path:  # re-drive the chosen path with the real warm-started joints; keep the prefix that holds
            f, c, qj = nodes[j]
            W = way(cur, qc, c, qj)
            qn = W is not None and commanded_ok(arm, [cur, *W, c], cfg, ok_pose=okc, q0=qc, ret_q=True)
            if qn is None or qn is False:
                break
            views.append((f, c, qn, W))
            cur, qc = c, qn
        return views

    best = ([], None)
    for z in cfg["tip_z"]:
        for x, y in cfg["points"]:
            P = np.array([x, SIGN[arm] * abs(y), z])
            v = views_at(P)
            best = max(best, (v, P), key=lambda b: len(b[0]))
            if len(v) >= cfg["stop_views"]:
                break
        if len(best[0]) >= cfg["stop_views"]:
            break
    views, P = best
    if not views:
        return None
    ql = q
    op, cl = cfg["grip_open"], cfg["grip_close"]
    calls = [{"ee": pre[0], "rpy": pre[1], "grip": op, "tag": "pre", "q": qs[0]}, {"ee": gr[0], "rpy": gr[1], "grip": op, "tag": "descend", "q": qs[1]},
             {"ee": gr[0], "rpy": gr[1], "grip": cl, "tag": "close", "q": qs[1]}, {"ee": tl[0], "rpy": tl[1], "grip": cl, "tag": "testlift", "q": qs[2]},
             {"ee": up[0], "rpy": up[1], "grip": cl, "tag": "lift", "q": ql}]
    for f, (ee, rpy), qv, W in views:
        calls += [{"ee": w[0], "rpy": w[1], "grip": cl, "tag": "via"} for w in W]
        calls += [{"ee": ee, "rpy": rpy, "grip": cl, "tag": t, "face": f, "q": qv} for t in ("move", "hold")]
    shown = [v[0] for v in views]
    return {"arm": arm, "grasp": g, "P": P, "hold": hold, "fit": fit, "faces": shown, "missing": [f for f in FACES if f not in shown],
            "calls": calls, "home": h}


def set_down(plan, deliv, done, cfg, home=None):
    """Calls that put a SHOW parcel back after `done` (>= 1) of plan["calls"] were driven, for the delivery arm `deliv` ("left_ee" / "right_ee").
    Not yet gripped -> up to the pregrasp. Holding: the first (spot, tool yaw, pitch) whose parcel footprint is clear (footprint_clear), which the
    gripping arm can reach, and where `deliv` passes grasp_plan, spots = slide.handoff of the delivery side + show.spots offsets, yaws every 15 deg
    nearest the delivery arm's natural grasp yaw first, the grasp pitch (the parcel sits flat only there); else the spot it was lifted from (always feasible: it is the
    grasp reversed); the route there is direct, else back through the views. Then: above the spot, down, open, up. -> {"calls","spot","yaw","where","fit"}.
    ponytail: spots / yaws are tried in a fixed order, first feasible wins."""
    arm, cl, op, cs = plan["arm"], cfg["grip_close"], cfg["grip_open"], plan["calls"]
    i = done - 1
    out = lambda calls, where, spot=None, yaw=None, fit=None: {"calls": calls, "where": where, "spot": spot, "yaw": yaw, "fit": fit}
    if i < 2:  # the gripper never closed on it
        return out([{"ee": cs[0]["ee"], "rpy": cs[0]["rpy"], "grip": op, "tag": "up"}] * (i > 0), "none")
    cur = cs[i]
    hold, g, H = plan["hold"], plan["grasp"], plan["hold"][2]
    pose = lambda tip, pitch, yaw: reach(arm, tip, pitch, cfg, yaw=yaw)
    gr = (cs[2]["ee"], cs[2]["rpy"])
    if i <= 3:  # closed on it, still on the table (or 2 cm up: supported): back down to the grasp pose, let go, back up
        return out(([{"ee": gr[0], "rpy": gr[1], "grip": cl, "tag": "down"}] if i == 3 else []) + [{"ee": gr[0], "rpy": gr[1], "grip": op, "tag": "release"},
                   {"ee": cs[0]["ee"], "rpy": cs[0]["rpy"], "grip": op, "tag": "retreat"}], "orig", g["grasp"][:2], g["yaw"], plan["fit"])
    trail = [c for c in cs[4:i + 1] if c["tag"] != "hold"]  # lift, vias, moves up to the current one
    back = [(c["ee"], c["rpy"]) for c in trail[-2::-1]]  # the way back to just above the grasp spot
    up = (cs[4]["ee"], cs[4]["rpy"])
    here = (cur["ee"], cur["rpy"])
    j = max((k for k in range(i + 1) if cs[k].get("q") is not None), default=None)  # the last pose with known joints, replayed forward (warm-started IK) to here
    q0 = cs[j]["q"] if j is not None else ik(*_home(arm, home), q0=np.zeros(6))[0]
    q = commanded_ok(arm, [(cs[k]["ee"], cs[k]["rpy"]) for k in range(max(j or 0, 0), i + 1)], cfg, q0=q0, ret_q=True) if j != i else q0
    q = q0 if q is None else q

    def chain(A, Pp, fit2, where, spot, yaw):
        for via in ([], [up], back):  # direct, via the lift pose, back through every view
            pts = [here, *via, A, Pp]
            ok = commanded_ok(arm, pts, cfg, ok_pose=lambda k, p, r: held_low(arm, p, r, hold, cfg) >= (TABLE_Z - 0.003 if k == len(pts) - 2 else TABLE_Z + cfg["held_clear"]),
                              q0=q, ret_q=True)
            if ok is not None and ok is not False:
                calls = [{"ee": e, "rpy": r, "grip": cl, "tag": "route"} for e, r in pts[1:-1]]
                calls += [{"ee": Pp[0], "rpy": Pp[1], "grip": cl, "tag": "down"}, {"ee": Pp[0], "rpy": Pp[1], "grip": op, "tag": "release"},
                          {"ee": A[0], "rpy": A[1], "grip": op, "tag": "retreat"}]
                return out(calls, where, spot, yaw, fit2)

    side = deliv[:-3]
    c0 = np.array(cfg["handoff"][side], float)
    for off in cfg["spots"]:
        spot = c0 + off
        az = np.arctan2(spot[1] - frame(deliv, cfg)[1], spot[0])
        yaws = sorted(az + np.radians(15) * np.arange(-12, 12), key=lambda y: min(abs((y - az + np.pi) % (2 * np.pi) - np.pi), abs((y - az) % (2 * np.pi) - np.pi)))
        for yaw in yaws[:8]:
            yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
            fit2 = {"c": spot, "axes": np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]]), "half": np.array([hold[1], hold[0]]) / 2,
                    "yaw": float(yaw)}
            if not footprint_clear(spot, fit2, cfg):
                continue
            pitch = g["pitch"]  # the parcel is tilted in the hand by 90 deg - pitch: it sits flat only at the grasp pitch
            rp, rA = pose((*spot, TABLE_Z + H - cfg["grasp_depth"]), pitch, yaw), pose((*spot, TABLE_Z + H - cfg["grasp_depth"] + cfg["place_lift"]), pitch, yaw)
            if rp and rA and grasp_plan(deliv, fit2, cfg, home):
                r = chain(rA[:2], rp[:2], fit2, "handoff", spot, yaw)
                if r:
                    return r
    r = chain(up, gr, plan["fit"], "orig", np.array(g["grasp"][:2]), g["yaw"])
    if r:
        return r
    # no checked route: never let go in the air. Reverse the driven trail (every leg was driven forward with the parcel) to the grasp pose, then release there.
    calls = [{"ee": e, "rpy": r_, "grip": cl, "tag": "route"} for e, r_ in [*back, up]]
    calls += [{"ee": gr[0], "rpy": gr[1], "grip": cl, "tag": "down"}, {"ee": gr[0], "rpy": gr[1], "grip": op, "tag": "release"},
              {"ee": cs[0]["ee"], "rpy": cs[0]["rpy"], "grip": op, "tag": "retreat"}]
    return out(calls, "orig", np.array(g["grasp"][:2]), g["yaw"], plan["fit"])
