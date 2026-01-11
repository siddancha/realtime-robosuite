"""
Quick demo script that samples random actions in the CircleDrawing environment and renders the result.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import tyro

import robosuite as suite


def run_demo(
    horizon: float = 10.0,
    seed: Optional[int] = None,
    real_time_rate: float = 1.0,
) -> None:
    """
    Visualize `TrajectoryFollowing` by executing random actions.

    Args:
        horizon (float): Simulation time horizon in seconds.
        seed (Optional[int]): Random seed for action sampling.
        real_time_rate (float): Target real-time rate for the simulation.
    """
    rng = np.random.default_rng(seed)

    env = suite.make_async(
      env_name="Lift",
      robots="Panda",
      has_renderer=True,
      has_offscreen_renderer=False,
      use_camera_obs=False,
      horizon=horizon,  # seconds of simulation time
      control_freq=50.0,     # new parameter
      observation_freq=30.0,  # new parameter
      target_real_time_rate=real_time_rate  # new parameter
  )

    env.start()

    while not env.done():
        action = rng.uniform(0, 1, size=env.action_dim)
        env.control_stream.push(action)  # delta actions

    env.stop()


def main():
    tyro.cli(run_demo)

if __name__ == "__main__":
    main()
