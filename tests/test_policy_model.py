"""policies/labelloop/model.py loads and serves a step, with a stand-in policy_space.base."""
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

from wallx_stub import Stub


class PolicyModel:
    def __init__(self, cfg):
        self.cfg = cfg


def test_model_wrapper_serves_33x14(tmp_path, monkeypatch):
    base = types.ModuleType("policy_space.base")
    base.PolicyModel = PolicyModel
    monkeypatch.setitem(sys.modules, "policy_space", types.ModuleType("policy_space"))
    monkeypatch.setitem(sys.modules, "policy_space.base", base)
    stub = Stub()
    for k, v in {"WALLX_ENDPOINT": stub.url, "WALLX_API_KEY": "", "WALLX_HEADERS_FILE": "", "AGENT_CHAT_URL": ""}.items():
        monkeypatch.setenv(k, v)
    path = Path(__file__).parent.parent / "policies/labelloop/model.py"
    spec = importlib.util.spec_from_file_location("ll_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    m = mod.Model({"log": str(tmp_path / "sim.jsonl")})
    try:
        m.initialize_episode({"instruction": "sort"})
        img = np.full((480, 640, 3), 128, np.uint8)
        m.ingest_observation({"images": {"camera_front": img, "camera_left": img, "camera_right": img},
                              "state": {"follow1_pos": np.zeros(7), "follow2_pos": np.zeros(7)}})
        for _ in range(3):  # no agent: SURVEY decides on call 1 and goes straight to LOOK
            assert m.infer_actions().shape == (33, 14)
        assert len(stub.requests) == 3 and (tmp_path / "sim.jsonl").exists()
    finally:
        m.close()
        stub.stop()
