"""Pose math between X2Real sim wire (home-relative) and WALLX absolute. numpy only.

Euler convention (both sides): R = Rz(yaw) Ry(pitch) Rx(roll).
"""
import numpy as np


def rpy_to_matrix(rpy):
    r, p, y = (float(v) for v in rpy)
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def matrix_to_rpy(R):
    if abs(R[2, 0]) >= 1 - 1e-9:  # gimbal lock: pitch = +-90deg, fold roll into yaw
        return np.array([0.0, -np.sign(R[2, 0]) * np.pi / 2, np.arctan2(-R[0, 1], R[1, 1])])
    return np.array([np.arctan2(R[2, 1], R[2, 2]), np.arcsin(-R[2, 0]), np.arctan2(R[1, 0], R[0, 0])])


def interp_rpy(rpy0, rpy1, t):
    """rpy of R0 @ exp(t * log(R0.T @ R1)): geodesic rotation interpolation."""
    R0 = rpy_to_matrix(rpy0)
    D = R0.T @ rpy_to_matrix(rpy1)
    ang = np.arccos(np.clip((np.trace(D) - 1) / 2, -1, 1))
    if ang < 1e-8:
        return matrix_to_rpy(R0)
    if np.pi - ang < 1e-6:  # near pi the antisymmetric part vanishes: axis from the symmetric part
        B = (D + np.eye(3)) / 2
        i = int(np.argmax(np.diag(B)))
        axis = B[:, i] / np.sqrt(B[i, i])
    else:
        axis = np.array([D[2, 1] - D[1, 2], D[0, 2] - D[2, 0], D[1, 0] - D[0, 1]]) / (2 * np.sin(ang))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    a = t * ang
    return matrix_to_rpy(R0 @ (np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K))


def _home(robot_cfg, arm):
    h = robot_cfg["home_pose"][arm]
    return np.asarray(h["position"], float), rpy_to_matrix(h["rotation_rpy"])


def sim_to_wallx(follow1_7, follow2_7, robot_cfg):
    """Sim wire [rel_xyz, rel_rpy, gripper] per arm (follow1=left, follow2=right) -> WALLX state dict."""
    g = robot_cfg["gripper"]
    out = {}
    for arm, f in (("left_ee", follow1_7), ("right_ee", follow2_7)):
        f = np.asarray(f, float)
        hp, hR = _home(robot_cfg, arm)
        grip = np.clip(f[6] * g["wallx_max"] / g["sim_max"], 0, g["wallx_max"])
        out[arm] = {"position": (f[:3] + hp).tolist(),
                    "rotation_rpy": matrix_to_rpy(rpy_to_matrix(f[3:6]) @ hR).tolist(),
                    "gripper": float(grip)}
    return out


def wallx_to_sim(actions, robot_cfg):
    """WALLX actions {left_ee,right_ee: {position[N][3], rotation_rpy[N][3], gripper[N]}} -> (N, 14) float32."""
    g = robot_cfg["gripper"]
    cols = []
    for arm in ("left_ee", "right_ee"):
        a = actions[arm]
        pos, rpy, grip = np.asarray(a["position"], float), np.asarray(a["rotation_rpy"], float), np.asarray(a["gripper"], float)
        hp, hR = _home(robot_cfg, arm)
        rel_rpy = np.array([matrix_to_rpy(rpy_to_matrix(r) @ hR.T) for r in rpy])
        sim_g = np.clip(grip * g["sim_max"] / g["wallx_max"], 0, g["sim_max"])
        cols.append(np.column_stack([pos - hp, rel_rpy, sim_g]))
    return np.hstack(cols).astype(np.float32)
