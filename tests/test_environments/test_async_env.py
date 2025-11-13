import time
from functools import partial
from typing import Optional

import numpy as np
import pytest

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import EmptyArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.async_env import AsyncSimulation, StepResult


class EmptyArenaManipulationEnv(ManipulationEnv):
    def __init__(self, terminate_after: Optional[int] = None):
        self._terminate_after = terminate_after
        super().__init__(
            robots="UR5e",
            use_camera_obs=False,
            has_renderer=False,
            has_offscreen_renderer=False,
            render_camera=None,
            control_freq=50.0,
            lite_physics=False,
            horizon=1000,
        )

    def _load_model(self):
        super()._load_model()

        arena = EmptyArena()
        arena.set_origin([0, 0, 0])

        for robot in self.robots:
            robot.robot_model.set_base_xpos([0, 0, 0])

        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
        )

    def reward(self, action):
        return float(np.sum(action))

    def _post_action(self, action):
        reward = self.reward(action)
        done = False
        if self._terminate_after is not None and self.timestep >= self._terminate_after:
            done = True
        self.done = done
        return reward, done, {"timestep": self.timestep}

    def _get_observations(self, force_update=False):
        observations = super()._get_observations(force_update=force_update)
        observations["obs"] = np.array([self.timestep], dtype=np.float32)
        return observations


def make_manipulation_env(terminate_after: Optional[int] = None) -> ManipulationEnv:
    return EmptyArenaManipulationEnv(terminate_after=terminate_after)


def test_async_simulation_runs_and_uses_latest_action():
    template_env = make_manipulation_env()
    action_dim = template_env.action_dim
    template_env.close()

    sim = AsyncSimulation(make_manipulation_env, action_freq=20.0, observation_freq=10.0)
    try:
        sim.start()
        initial_step = sim.observation_stream.get(timeout=1.0)
        assert isinstance(initial_step, StepResult)

        new_action = np.zeros(action_dim, dtype=np.float32)
        new_action[:2] = np.array([0.75, -0.25], dtype=np.float32)
        sim.action_stream.push(new_action)

        deadline = time.time() + 1.5
        next_step = None
        while time.time() < deadline:
            remaining = max(0.0, deadline - time.time())
            next_step = sim.observation_stream.get(timeout=remaining or 0.05)
            if np.allclose(next_step.action, new_action, atol=1e-6):
                break
        assert next_step is not None and np.allclose(next_step.action, new_action, atol=1e-6)
        assert next_step.reward == pytest.approx(float(new_action.sum()), abs=1e-5)
    finally:
        sim.stop()


def test_async_simulation_stops_when_episode_done_without_auto_reset():
    env_factory = partial(make_manipulation_env, terminate_after=3)
    sim = AsyncSimulation(env_factory, action_freq=15.0, observation_freq=15.0, auto_reset=False)
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


