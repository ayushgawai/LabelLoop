import numpy as np
import pytest

from labelloop.pose import sim_to_wallx
from labelloop.robots.x2real_server import X2RealLink
from wallx_stub import Stub, rgb
from labelloop.wallx_client import WallxClient

CFG = {"embodiment": "quanta_x1_pro/gripper-g", "gripper": {"wallx_max": 1.526, "sim_max": 1.89},
       "home_pose": {"left_ee": {"position": [.29, .02, .37], "rotation_rpy": [.3, -.36, -.1]},
                     "right_ee": {"position": [.28, -.05, .39], "rotation_rpy": [.03, -.51, -.37]}}}


class FakeHarness:
    def __init__(self, instruction="pick"):
        self.instruction, self.calls, self.replies, self.resets = instruction, [], [], 0
        self.cfg, self.state_name, self.home = {}, "IDLE", None

    def reset(self):
        self.resets += 1

    def step(self, images, state):
        self.calls.append((images, state))
        return self.instruction

    def log_reply(self, latency_s, error=None):
        self.replies.append(error)


def obs():
    return {"images": {"camera_front": rgb(1), "camera_left": rgb(2)[None], "camera_right": rgb(3)},
            "state": {"follow1_pos": np.array([.01, 0, 0, 0, 0, 0, .5], np.float32),
                      "follow2_pos": np.array([0, .01, 0, 0, 0, .1, 1.0], np.float32)}}


@pytest.fixture
def stub():
    s = Stub()
    yield s
    s.stop()


def test_step_shape_and_row0(stub):
    h = FakeHarness()
    link = X2RealLink(h, WallxClient(stub.url), CFG)
    link.reset({})
    out = link.step(obs())
    assert out.shape == (33, 14) and out.dtype == np.float32
    assert np.allclose(out[0], np.r_[obs()["state"]["follow1_pos"], obs()["state"]["follow2_pos"]])
    assert np.allclose(out[1:], out[0], atol=1e-5)  # stub echoes state -> chunk equals current pose
    assert h.replies == [None] and h.resets == 1
    assert stub.requests[0]["instruction"] == "pick"
    assert set(h.calls[0][0]) == {"front", "left_wrist", "right_wrist"}
    assert h.calls[0][1]["left_ee"]["position"][0] == pytest.approx(.30)  # home + rel


def test_hold_path_no_wallx_call(stub):
    link = X2RealLink(FakeHarness(None), WallxClient(stub.url), CFG)
    out = link.step(obs())
    assert out.shape == (33, 14) and np.allclose(out[1:], out[0], atol=1e-5) and stub.requests == []


def test_wallx_error_holds(stub):
    h = FakeHarness()
    link = X2RealLink(h, WallxClient(stub.url), CFG, embodiment="bad")
    out = link.step(obs())
    assert out.shape == (33, 14) and np.allclose(out[1:], out[0], atol=1e-5)
    assert h.replies[0].startswith("WallxError")


def test_malformed_obs_raises(stub):
    link = X2RealLink(FakeHarness(), WallxClient(stub.url), CFG)
    bad = obs()
    del bad["images"]["camera_left"]
    with pytest.raises(KeyError):
        link.step(bad)
    bad = obs()
    bad["state"]["follow1_pos"] = np.zeros(6)
    with pytest.raises(ValueError):
        link.step(bad)


def test_reset_history_first_call_and_subgoal(stub):
    h = FakeHarness()
    h.subgoal_changed = False
    link = X2RealLink(h, WallxClient(stub.url), CFG)
    link.step(obs())
    link.step(obs())
    h.subgoal_changed = True
    link.step(obs())
    link.reset({})
    h.subgoal_changed = False
    link.step(obs())
    assert ["reset_history" in r for r in stub.requests] == [True, False, True, True]


def test_large_images_are_resized_for_wallx_only():
    from labelloop.robots.x2real_server import _for_wallx
    big = np.zeros((720, 1280, 3), np.uint8)
    assert _for_wallx(big).shape == (480, 640, 3)
    small = np.zeros((480, 640, 3), np.uint8)
    assert _for_wallx(small) is small


def test_scripted_home_no_wallx_call(stub):
    h = FakeHarness("go home")
    h.cfg, h.state_name = {"scripted_home": True}, "HOME"
    h.home = sim_to_wallx(np.zeros(7), np.zeros(7), CFG)
    link = X2RealLink(h, WallxClient(stub.url), CFG)
    link.reset({})
    out = link.step(obs())
    assert stub.requests == [] and np.allclose(out[-1], 0, atol=1e-4)
