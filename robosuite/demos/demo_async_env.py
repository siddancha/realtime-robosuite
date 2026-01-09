"""
Quick demo script that samples random actions in the CircleDrawing environment and renders the result.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import robosuite as suite


def main(
    steps: int = -1,
    seed: Optional[int] = None,
) -> None:
    """
    Visualize `TrajectoryFollowing` by executing random actions.

    Args:
        steps (int): Number of environment steps to simulate. If -1, runs until the environment is done.
        seed (Optional[int]): Random seed for action sampling.
    """
    rng = np.random.default_rng(seed)

    env = suite.make_async(
      env_name="Lift",
      robots="Panda",
      has_renderer=True,
      has_offscreen_renderer=False,
      use_camera_obs=False,
      horizon=10.0,  # seconds of simulation time
      control_freq=50.0,     # new parameter
      observation_freq=30.0,  # new parameter
  )

    env.start()

    while not env.done():
        action = rng.uniform(0, 1, size=env.action_dim)
        env.control_stream.push(action)

    env.stop()


if __name__ == "__main__":
    main()