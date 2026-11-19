import time

from labelloop.brain.state_machine import StateMachine
from labelloop.perception import scene
from labelloop.perception.qr import Confirmer, decode, locate


class Harness:
    """Perception + state machine + runlog (+ optional vision agent); one step() per WALLX call."""

    def __init__(self, task_cfg, agent=None, runlog=None, clock=time.monotonic):
        self.cfg, self.agent, self.runlog, self.clock = task_cfg, agent, runlog, clock
        self.sm = StateMachine(task_cfg)
        self.reset()

    def reset(self):
        self.sm.reset()
        self.confirmer = Confirmer(self.cfg["qr_confirm_frames"])
        self.calls, self.subgoal_changed, self.done, self.state_name = 0, False, False, self.sm.state

    def step(self, images, state):
        self.calls += 1
        before, cfg = self.sm.state, self.cfg["scene"]
        front = images["front"]
        located = locate(front)
        wrist = {t for t, view in decode({k: v for k, v in images.items() if k != "front"})}
        confirmed = self.confirmer.update({t for t, _ in located} | wrist)
        bins = scene.bins(front, cfg)
        blocks = scene.blocks(front, bins, cfg)
        grips = (state["left_ee"]["gripper"], state["right_ee"]["gripper"])
        verdict = None
        if self.agent and (before == "SURVEY" or self.calls % self.cfg["agent_every"] == 0):
            verdict = self.agent.ask(images, before, {"instruction": self.sm.instruction(), "calls": self.calls,
                                     "sorted": len(self.sm.sorted), "rejected": self.sm.rejected_count})
        kraft = scene.kraft_in_bins(front, bins, cfg)
        instr = self.sm.step({"h": front.shape[0], "w": front.shape[1], "front": located, "confirmed": confirmed,
                              "wrist": wrist, "bins": bins, "blocks": blocks, "grips": grips, "verdict": verdict,
                              "kraft": kraft})
        self.state_name, self.done = self.sm.state, self.sm.state == "DONE"
        self.subgoal_changed = self.sm.state != before
        if self.runlog:
            self.runlog.write(call=self.calls, state=before, instruction=instr, codes=located, wrist_codes=wrist,
                              confirmed=confirmed, bins=bins, blocks=len(blocks), kraft=kraft, left_gripper=grips[0],
                              right_gripper=grips[1], verdict=verdict, transitions=self.sm.transitions,
                              flags=self.sm.flags)
        return instr

    def log_reply(self, latency_s, error=None):
        if self.runlog:
            self.runlog.write(call=self.calls, latency_s=latency_s, error=error)
