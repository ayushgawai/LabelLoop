import time

from labelloop.brain.state_machine import StateMachine
from labelloop.perception.qr import Confirmer, decode


class Harness:
    """Perception + agent + state machine + runlog; one step() per WALLX call."""

    def __init__(self, task_cfg, agent=None, runlog=None, clock=time.monotonic):
        self.cfg, self.agent, self.runlog, self.clock = task_cfg, agent, runlog, clock
        self.sm = StateMachine(task_cfg)
        self.reset()

    def reset(self):
        self.sm.reset(self.clock())
        self.confirmer = Confirmer(self.cfg["qr_confirm_frames"])
        self.calls, self.subgoal_changed, self.done, self.state_name = 0, False, False, self.sm.state

    def step(self, images, state):
        self.calls += 1
        before = self.sm.state
        decoded = decode(images)
        confirmed = self.confirmer.update({t for t, _ in decoded})
        verdict = None
        if self.agent and (before == "SURVEY" or self.calls % self.cfg["agent_every"] == 0):
            verdict = self.agent.ask(images, before, {"instruction": self.sm.instruction(), "calls": self.calls,
                                     "sorted": len(self.sm.memory.sorted), "rejected": self.sm.memory.rejected_count})
        grip = state["left_ee"]["gripper"]
        instr = self.sm.step({"now": self.clock(), "grip": grip, "verdict": verdict,
                              "codes": [(t, v) for t, v in decoded if t in confirmed]})
        self.state_name, self.done = self.sm.state, self.sm.state == "DONE"
        self.subgoal_changed = self.sm.state != before
        if self.runlog:
            self.runlog.write(call=self.calls, state=before, instruction=instr, codes=decoded,
                              confirmed=confirmed, left_gripper=grip, right_gripper=state["right_ee"]["gripper"],
                              verdict=verdict, transitions=self.sm.transitions)
        return instr

    def log_reply(self, latency_s, error=None):
        if self.runlog:
            self.runlog.write(call=self.calls, latency_s=latency_s, error=error)
