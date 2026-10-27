"""Expose the single MuJoCo robot through the MASB3 interface."""

from masb3.base import BaseMAEnv

from waste_collection_env import WasteCollectionEnv


class RobotMAEnv(BaseMAEnv):
    def __init__(self, render_mode=None):
        super().__init__()
        self.env = WasteCollectionEnv(render_mode=render_mode)
        self.register_agent("robot", self.env.observation_space, self.env.action_space)

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        self.observation = obs
        self.previous_observation = {"robot": obs}
        return {"robot": obs}, info

    def step_agent(self, agent_id, action):
        self.action = action

    def sync_wait_for_actions_completion(self):
        self.observation, self.reward, self.terminated, self.truncated, self.info = (
            self.env.step(self.action)
        )

    def get_observation(self, agent_id):
        return self.observation

    def evaluate_env_state(self):
        return {"robot": self.reward}, self.terminated, self.truncated, self.info

    def close(self):
        self.env.close()
