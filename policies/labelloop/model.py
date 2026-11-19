"""Policy Space deployment of LabelLoop. Thin wrapper; all logic lives in labelloop/."""
from pathlib import Path

import numpy as np
from policy_space.base import PolicyModel

ROOT = Path(__file__).resolve().parents[2]


class Model(PolicyModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        from labelloop.brain.agent import VisionAgent
        from labelloop.config import load_env, load_yaml
        from labelloop.eval.runlog import RunLog
        from labelloop.harness import Harness
        from labelloop.robots.x2real_server import X2RealLink
        from labelloop.wallx_client import WallxClient

        load_env(str(ROOT / ".env"))
        robot_cfg = load_yaml(str(ROOT / cfg.get("robot", "configs/ex001_6r.yaml")))
        harness = Harness(load_yaml(str(ROOT / cfg.get("task", "configs/task_qr_sort.yaml"))),
                          VisionAgent.from_env(), RunLog(str(ROOT / cfg.get("log", "runs/sim.jsonl"))))
        self.wallx = WallxClient.from_env()
        self.link = X2RealLink(harness, self.wallx, robot_cfg)
        self._obs = None

    def initialize_episode(self, episode_info=None):
        self._obs = None
        self.link.reset(episode_info)

    def ingest_observation(self, obs):
        self._obs = obs

    def infer_actions(self) -> np.ndarray:
        if self._obs is None:
            raise RuntimeError("No observation buffered. Call ingest_observation first.")
        return self.link.step(self._obs)

    def close(self):
        self.wallx.close()
