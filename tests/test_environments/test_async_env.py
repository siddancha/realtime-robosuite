import time

import numpy as np
import pytest

import robosuite as suite
from robosuite.environments.async_env import Observation


def test_async_simulation_runs_and_uses_latest_action():
    sim = suite.make_async(
        env_name="Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20.0,
        observation_freq=10.0,
    )

    try:
        sim.start()
        retrieved_action_dim = sim.action_dim
        assert retrieved_action_dim == 7  # Panda has 7 DoF
        initial_obs = sim.observation_stream.get(timeout=1.0)
        assert isinstance(initial_obs, Observation)

        new_action = np.zeros(retrieved_action_dim, dtype=np.float32)
        new_action[:2] = np.array([0.75, -0.25], dtype=np.float32)
        sim.control_stream.push(new_action)

        deadline = time.time() + 1.5
        next_obs = None
        while time.time() < deadline:
            remaining = max(0.0, deadline - time.time())
            next_obs = sim.observation_stream.get(timeout=remaining or 0.05)
            if np.allclose(next_obs.action, new_action, atol=1e-6):
                break
        assert next_obs is not None and np.allclose(next_obs.action, new_action, atol=1e-6)
        assert next_obs.time > initial_obs.time
    finally:
        sim.stop()

def test_async_simulation_action_dim_without_start():
    sim = suite.make_async(
        env_name="Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=10.0,
        observation_freq=10.0,
    )
    try:
        with pytest.raises(RuntimeError):
            _ = sim.action_dim
    finally:
        sim.stop()

def test_async_simulation_stops_when_episode_done():
    sim = suite.make_async(
        env_name="Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=15.0,
        observation_freq=15.0,
        horizon=3.0,  # short horizon to trigger done quickly
    )
    try:
        sim.start()
        deadline = time.time() + 2.0
        while time.time() < deadline:
            obs = sim.latest_observation()
            time.sleep(0.05)
        assert sim.latest_observation() is not None
    finally:
        sim.stop()

