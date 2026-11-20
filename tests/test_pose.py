import numpy as np
import pytest

from labelloop.pose import matrix_to_rpy, rpy_to_matrix, sim_to_wallx, wallx_to_sim
from labelloop.wallx_client import hold_response

CFG = {"gripper": {"wallx_max": 1.526, "sim_max": 1.89},
       "home_pose": {"left_ee": {"position": [0.29, 0.02, 0.37], "rotation_rpy": [0.3, -0.36, -0.1]},
                     "right_ee": {"position": [0.28, -0.05, 0.39], "rotation_rpy": [0.03, -0.51, -0.37]}}}


def test_rpy_roundtrip_random():
    rng = np.random.default_rng(0)
    for _ in range(200):
        rpy = rng.uniform([-np.pi, -np.pi / 2 + 1e-3, -np.pi], [np.pi, np.pi / 2 - 1e-3, np.pi])
        assert np.allclose(matrix_to_rpy(rpy_to_matrix(rpy)), rpy, atol=1e-9)


@pytest.mark.parametrize("rpy", [[0, 0, 0], [0.3, np.pi / 2, 0.5], [-0.3, -np.pi / 2, 0.5], [0, 0, np.pi]])
def test_rpy_edges_same_rotation(rpy):
    R = rpy_to_matrix(rpy)
    assert np.allclose(rpy_to_matrix(matrix_to_rpy(R)), R, atol=1e-9)


def test_hand_computed_matrix():
    # yaw 90deg only: x axis -> y axis
    assert np.allclose(rpy_to_matrix([0, 0, np.pi / 2]) @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    # roll 90deg only: y axis -> z axis
    assert np.allclose(rpy_to_matrix([np.pi / 2, 0, 0]) @ [0, 1, 0], [0, 0, 1], atol=1e-12)


def test_hand_computed_sim_to_wallx():
    s = sim_to_wallx([0.1, 0, -0.1, 0, 0, 0, 1.89], [0, 0, 0, 0, 0, 0, 0.945], CFG)
    assert np.allclose(s["left_ee"]["position"], [0.39, 0.02, 0.27])
    assert np.allclose(s["left_ee"]["rotation_rpy"], [0.3, -0.36, -0.1])  # zero relative rotation = home
    assert s["left_ee"]["gripper"] == pytest.approx(1.526)
    assert s["right_ee"]["gripper"] == pytest.approx(0.763)


def test_roundtrip_identity():
    rng = np.random.default_rng(1)
    for _ in range(50):
        f1 = np.r_[rng.uniform(-.1, .1, 3), rng.uniform(-.5, .5, 3), rng.uniform(0, 1.89)]
        f2 = np.r_[rng.uniform(-.1, .1, 3), rng.uniform(-.5, .5, 3), rng.uniform(0, 1.89)]
        out = wallx_to_sim(hold_response("e", sim_to_wallx(f1, f2, CFG))["actions"], CFG)
        assert out.shape == (32, 14) and np.allclose(out, np.r_[f1, f2], atol=1e-6)


def test_gripper_map_endpoints_and_clip():
    g = lambda v: sim_to_wallx([0] * 6 + [v], [0] * 7, CFG)["left_ee"]["gripper"]  # noqa: E731
    assert g(0) == 0 and g(1.89) == pytest.approx(1.526) and g(5) == pytest.approx(1.526) and g(-1) == 0
    acts = hold_response("e", sim_to_wallx([0] * 7, [0] * 7, CFG))["actions"]
    acts["left_ee"]["gripper"] = [0.0, 1.526, 9.0] + [0.0] * 29
    assert np.allclose(wallx_to_sim(acts, CFG)[:3, 6], [0, 1.89, 1.89])


def test_interp_rpy_endpoints_and_home_response():
    from labelloop.pose import interp_rpy, rpy_to_matrix
    from labelloop.wallx_client import home_response
    for a, b in (([.3, -.4, -.1], [-.2, .5, .7]), ([0, 0, 3.1], [0, 0, -3.1]), ([0, 0, 0], [0, 0, 0])):
        assert np.allclose(rpy_to_matrix(interp_rpy(a, b, 0)), rpy_to_matrix(a), atol=1e-9)
        assert np.allclose(rpy_to_matrix(interp_rpy(a, b, 1)), rpy_to_matrix(b), atol=1e-9)
    mid = interp_rpy([0, 0, 3.1], [0, 0, -3.1], .5)  # short way across +-pi, not through 0
    assert abs(abs(mid[2]) - np.pi) < 1e-6
    st = {a: {"position": [.1, .2, .3], "rotation_rpy": [.3, -.4, -.1], "gripper": .1} for a in ("left_ee", "right_ee")}
    hm = {a: {"position": [0, 0, .5], "rotation_rpy": [0, .2, 3.1], "gripper": .9} for a in ("left_ee", "right_ee")}
    r = home_response("e", st, hm)["actions"]["left_ee"]
    assert len(r["position"]) == 32 and np.allclose(r["position"][-1], hm["left_ee"]["position"])
    assert np.allclose(rpy_to_matrix(r["rotation_rpy"][-1]), rpy_to_matrix(hm["left_ee"]["rotation_rpy"]), atol=1e-9)
    assert r["gripper"][-1] == pytest.approx(.9) and np.allclose(r["position"][0], st["left_ee"]["position"], atol=5e-3)
