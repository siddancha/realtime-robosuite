import time
from collections import OrderedDict
from typing import Optional

import numpy as np
import pytest

from robosuite.utils.async_env import AsyncSimulation, StepResult


class DummyEnv:
    def __init__(self, terminate_after: Optional[int] = None):
        self._terminate_after = terminate_after
        self.action_spec = (np.full(2, -1.0, dtype=np.float32), np.full(2, 1.0, dtype=np.float32))
        self.control_freq = 50.0
        self.control_timestep = 1.0 / self.control_freq
        self.horizon = 1000
        self.ignore_done = False
        self.done = False
        self.timestep = 0

    def initialize_time(self, control_freq):
        self.control_freq = control_freq
        self.control_timestep = 1.0 / control_freq

    def reset(self):
        self.timestep = 0
        self.done = False
        return OrderedDict(obs=np.array([self.timestep], dtype=np.float32))

    def step(self, action):
        self.timestep += 1
        observation = OrderedDict(obs=np.array([self.timestep], dtype=np.float32))
        reward = float(np.sum(action))
        done = False
        if self._terminate_after is not None and self.timestep >= self._terminate_after:
            done = True
            self.done = True
        info = {"timestep": self.timestep}
        return observation, reward, done, info


def test_async_simulation_runs_and_uses_latest_action():
    env = DummyEnv()
    sim = AsyncSimulation(env, action_freq=20.0, observation_freq=10.0)
    try:
        sim.start()
        initial_step = sim.observation_stream.latest()
        assert isinstance(initial_step, StepResult)
        assert env.control_freq == 20.0

        new_action = np.array([0.75, -0.25], dtype=np.float32)
        sim.action_stream.push(new_action)

        next_step = sim.observation_stream.get(timeout=1.5)
        assert np.allclose(next_step.action, new_action, atol=1e-6)
        assert next_step.reward == pytest.approx(float(new_action.sum()), abs=1e-5)
    finally:
        sim.stop()


def test_async_simulation_stops_when_episode_done_without_auto_reset():
    env = DummyEnv(terminate_after=3)
    sim = AsyncSimulation(env, action_freq=15.0, observation_freq=15.0, auto_reset=False)
    try:
        sim.start()
        deadline = time.time() + 2.0
        while time.time() < deadline:
            latest = sim.latest_step()
            if latest and latest.done:
                break
            time.sleep(0.05)
        latest = sim.latest_step()
        assert latest is not None and latest.done
    finally:
        sim.stop()


