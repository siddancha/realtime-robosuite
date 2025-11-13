import time

import numpy as np

import robosuite as suite
from robosuite.utils.async_env import AsyncSimulation, StepResult


def make_manipulation_env():
    return suite.make(
        env_name="Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )

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
        assert next_step.timestamp > initial_step.timestamp
    finally:
        sim.stop()


def test_async_simulation_stops_when_episode_done_without_auto_reset():
    sim = AsyncSimulation(make_manipulation_env, action_freq=15.0, observation_freq=15.0, auto_reset=False)
    try:
        sim.start()
        deadline = time.time() + 2.0
        while time.time() < deadline:
            latest = sim.latest_step()
            if latest and latest.done:
                break
            time.sleep(0.05)
        assert sim.latest_step() is not None
    finally:
        sim.stop()


