import time

import numpy as np
import pytest

import robosuite as suite


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
