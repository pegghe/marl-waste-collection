"""A robot learns to wait for a go signal and reach a random target."""

from pathlib import Path

import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv


class WasteCollectionEnv(MujocoEnv):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 20,
    }

    def __init__(self, render_mode=None):
        super().__init__(
            model_path=str(Path(__file__).with_name("scene.xml")),
            frame_skip=5,
            observation_space=spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float64),
            render_mode=render_mode,
            default_camera_config={"distance": 8.0, "elevation": -90.0},
        )
        self.max_steps = 400
        self.success_radius = 0.25

    def _get_obs(self):
        # Position, velocity, relative target coordinates, go signal.
        return np.concatenate((
            self.data.qpos[:2],
            self.data.qvel[:2],
            self.target - self.data.qpos[:2],
            [self.go],
        ))

    def reset_model(self):
        self.steps = 0
        self.go_delay = self.np_random.integers(0, 101)
        self.go = int(self.go_delay == 0)

        qpos = self.init_qpos.copy()
        qpos[:2] = self.np_random.uniform(-2, 2, size=2)
        self.target = self.np_random.uniform(-2, 2, size=2)
        while np.linalg.norm(self.target - qpos[:2]) < 0.75:
            self.target = self.np_random.uniform(-2, 2, size=2)

        self.model.body("target").pos[:2] = self.target
        self.set_state(qpos, self.init_qvel.copy())
        return self._get_obs()

    def step(self, action):
        action = np.clip(action, -1, 1)
        self.do_simulation(action, self.frame_skip)
        self.steps += 1

        distance = float(np.linalg.norm(self.target - self.data.qpos[:2]))
        success = bool(self.go and distance <= self.success_radius)
        reward = -0.01  # Step penalty.
        if self.go:
            reward -= distance
        else:
            reward -= np.linalg.norm(self.data.qvel[:2])
        if success:
            reward += 10

        # The next action receives go=1 once the waiting period is over.
        self.go = int(self.steps >= self.go_delay)
        if self.render_mode == "human":
            self.render()

        info = {"distance": distance, "is_success": success}
        return self._get_obs(), float(reward), success, self.steps >= self.max_steps, info
