"""Shared MuJoCo simulation and MASB3 coordination for robot and drone."""

from pathlib import Path

import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from masb3.base import BaseMAEnv

from agents import DroneAgent, RobotAgent


class WasteCollectionEnv(MujocoEnv):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 20,
    }

    def __init__(self, render_mode=None, drone_training=False):
        self.drone_training = drone_training
        super().__init__(
            model_path=str(Path(__file__).with_name("scene.xml")),
            frame_skip=5,
            observation_space=spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float64),
            render_mode=render_mode,
            default_camera_config={
                "distance": 16.0, "elevation": -35.0, "azimuth": 135.0,
                "lookat": np.array([0.0, 0.0, 1.0]),
            },
        )
        self.max_steps = 2000
        self.robot = RobotAgent(self)
        self.drone = DroneAgent(self)
        self.drone_action_space = self.drone.action_space
        self.drone_observation_space = self.drone.observation_space

    @property
    def drone_reward(self):
        return self.drone.reward

    def get_drone_observation(self):
        return self.drone.get_observation()

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        info["drone_observation"] = self.get_drone_observation()
        return obs, info

    def render(self):
        viewer = self.mujoco_renderer._get_viewer(self.render_mode)
        self.drone.render_rays(viewer)
        return super().render()

    def _get_obs(self):
        return self.robot.get_observation()

    def reset_model(self):
        self.steps = 0
        qpos = self.robot.reset()
        self.model.body("target").pos[:2] = self.target
        self.drone.reset_position(randomize=self.drone_training)
        self.set_state(qpos, self.init_qvel.copy())
        self.shared_comm = np.zeros(3, dtype=np.float32)
        self.drone.reset()
        return self._get_obs()

    def move_drone(self, action):
        self.drone.apply_action(action)

    def step(self, action):
        self.robot.apply_action(action)
        self.steps += 1
        reward, success, distance = self.robot.compute_reward()
        self.drone.update_rays()
        self.drone.compute_reward()
        if self.render_mode == "human":
            self.render()

        info = {
            "distance": distance, "is_success": success,
            "drone_ray_distances": self.drone.ray_distances.copy(),
            "drone_ray_geomids": self.drone.ray_geomids.copy(),
            "drone_reward": self.drone.reward,
            "drone_reward_components": self.drone.reward_components.copy(),
            "drone_found_target": bool(self.shared_comm[0]),
            "drone_observation": self.get_drone_observation(),
        }
        terminated = bool(self.shared_comm[0]) if self.drone_training else success
        info["is_success"] = terminated
        truncated = self.steps >= self.max_steps and not terminated
        return self._get_obs(), float(reward), terminated, truncated, info


class RobotMAEnv(BaseMAEnv):
    def __init__(self, render_mode=None, drone_training=False):
        super().__init__()
        self.drone_training = drone_training
        self.episode_successes = []
        self.env = WasteCollectionEnv(render_mode=render_mode, drone_training=drone_training)
        self.register_agent("robot", self.env.observation_space, self.env.action_space)
        if drone_training:
            self.register_agent("drone", self.env.drone_observation_space, self.env.drone_action_space)

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        self.observation = obs
        self.previous_observation = self.get_observations()
        return self.previous_observation, info

    def step_agent(self, agent_id, action):
        if agent_id == "drone":
            self.drone_action = action
        else:
            self.action = action

    def sync_wait_for_actions_completion(self):
        if self.drone_training:
            self.env.move_drone(self.drone_action)
        self.observation, self.reward, self.terminated, self.truncated, self.info = (
            self.env.step(self.action)
        )
        if self.terminated or self.truncated:
            self.episode_successes.append(bool(self.info["is_success"]))

    def get_observation(self, agent_id):
        if agent_id == "drone":
            return self.env.get_drone_observation()
        return self.observation

    def evaluate_env_state(self):
        rewards = {"robot": self.reward}
        if self.drone_training:
            rewards["drone"] = self.env.drone_reward
        return rewards, self.terminated, self.truncated, self.info

    def close(self):
        self.env.close()
