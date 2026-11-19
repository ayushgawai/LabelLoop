"""FK of each arm's EE link in the robot root frame at the default joints -> YAML for configs/ex001_6r.yaml.

X2Real sends EE poses relative to this home pose and never sends the absolute one.
Default joints of EX001-6R (G): all 0 (init_state has no joint_pos, robot yaml has no initial_joint_positions).
"""
import argparse
import math
import xml.etree.ElementTree as ET

import numpy as np

EE = {"left_ee": "left_arm_gripper_base_link", "right_ee": "right_arm_gripper_base_link"}


def rot_rpy(r, p, y):
    """R = Rz(y) Ry(p) Rx(r)."""
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def rpy_of(R):
    p = -math.asin(max(-1.0, min(1.0, R[2, 0])))
    if abs(R[2, 0]) > 1 - 1e-9:  # gimbal lock: fold roll into yaw
        return 0.0, p, math.atan2(-R[0, 1], R[1, 1])
    return math.atan2(R[2, 1], R[2, 2]), p, math.atan2(R[1, 0], R[0, 0])


def axis_angle(axis, a):
    x, y, z = axis / np.linalg.norm(axis)
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def vec(s, default):
    return np.array([float(v) for v in s.split()]) if s else np.array(default, float)


def parse(urdf_text):
    """-> {child_link: (parent_link, joint_name, type, xyz, rpy, axis)}"""
    out = {}
    for j in ET.fromstring(urdf_text).findall("joint"):  # direct children only (ros2_control has <joint> too)
        o, ax = j.find("origin"), j.find("axis")
        out[j.find("child").get("link")] = (
            j.find("parent").get("link"), j.get("name"), j.get("type"),
            vec(o.get("xyz") if o is not None else None, [0, 0, 0]),
            vec(o.get("rpy") if o is not None else None, [0, 0, 0]),
            vec(ax.get("xyz") if ax is not None else None, [1, 0, 0]))
    return out


def fk(joints, root, ee, q=None):
    """Pose (pos, R) of link `ee` in link `root`; q = {joint_name: value}, default 0."""
    q, chain, link = q or {}, [], ee
    while link != root:
        if link not in joints:
            raise SystemExit(f"{ee} is not a descendant of {root}")
        chain.append(joints[link])
        link = joints[link][0]
    pos, R = np.zeros(3), np.eye(3)
    for _, name, typ, xyz, rpy, axis in reversed(chain):
        pos, R = pos + R @ xyz, R @ rot_rpy(*rpy)
        v = q.get(name, 0.0)
        if typ in ("revolute", "continuous"):
            R = R @ axis_angle(axis, v)
        elif typ == "prismatic":
            pos = pos + R @ (axis / np.linalg.norm(axis) * v)
    return pos, R


def self_test():
    urdf = """<robot name="t"><link name="base"/><link name="l1"/><link name="ee"/>
    <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/>
      <origin xyz="0 0 1" rpy="0 0 0"/><axis xyz="0 0 1"/></joint>
    <joint name="j2" type="fixed"><parent link="l1"/><child link="ee"/>
      <origin xyz="1 0 0" rpy="0 0 1.5707963267948966"/></joint></robot>"""
    j = parse(urdf)
    pos, R = fk(j, "base", "ee")  # q=0
    assert np.allclose(pos, [1, 0, 1]) and np.allclose(rpy_of(R), [0, 0, math.pi / 2])
    pos, R = fk(j, "base", "ee", {"j1": math.pi / 2})  # hand math: (0,0,1)+Rz(90)(1,0,0) = (0,1,1); yaw 180
    assert np.allclose(pos, [0, 1, 1]) and abs(abs(rpy_of(R)[2]) - math.pi) < 1e-9
    for rpy in [(0.3, -0.4, 1.2), (-2.0, 0.5, -0.7)]:
        assert np.allclose(rpy_of(rot_rpy(*rpy)), rpy)
    print("self-test OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--urdf")
    ap.add_argument("--root", default="base_platform_link")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.urdf:
        ap.error("--urdf required")
    joints = parse(open(a.urdf, encoding="utf-8").read())
    print("home_pose:   # root =", a.root, "; default joints = 0")
    for key, link in EE.items():
        pos, R = fk(joints, a.root, link)
        print(f"  {key}:\n    position: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]\n"
              f"    rotation_rpy: [{rpy_of(R)[0]:.4f}, {rpy_of(R)[1]:.4f}, {rpy_of(R)[2]:.4f}]")


if __name__ == "__main__":
    main()
