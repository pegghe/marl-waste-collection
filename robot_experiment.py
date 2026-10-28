"""Train and evaluate the robot with MASB3."""

import argparse

import torch
from masb3.algos import get_algo
from masb3.base import MultiAgentSharedVecEnv
from masb3.utils import ma_evaluate, ma_train
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize

from robot_ma_env import RobotMAEnv


class RobotExperiment:
    def __init__(self, model_path="models/ppo_go/final_model"):
        self.model_path = model_path

    def train(self):
        torch.set_num_threads(1)
        robot_env = RobotMAEnv()
        robot_env.reset(seed=0)
        env = MultiAgentSharedVecEnv(robot_env, learning_agent_ids=["robot"])
        env = VecNormalize(VecMonitor(env), norm_obs=False, norm_reward=True)
        try:
            models = ma_train(
                ma_env=robot_env,
                agent_env_map={"robot": env},
                model_algo_map={"robot": (get_algo("sb3", "PPO"), {
                    "policy": "MlpPolicy", "seed": 0, "device": "cpu", "verbose": 1,
                })},
                total_timesteps_per_model=600_000,
                training_iterations=1,
            )
            models["robot"].save(self.model_path)
            return models
        finally:
            env.close()

    def evaluate(self, episodes=10, render=False):
        model = get_algo("sb3", "PPO").load(self.model_path, device="cpu")
        env = RobotMAEnv(render_mode="human" if render else None)
        try:
            avg_agent, avg_model = ma_evaluate(
                env, {"robot": model}, total_episodes=episodes, verbose=True,
            )
            print(f"Average reward per agent: {avg_agent}")
            print(f"Average reward per model: {avg_model}")
            return avg_agent, avg_model
        finally:
            env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "evaluate"])
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    experiment = RobotExperiment()
    if args.mode == "train":
        experiment.train()
    else:
        experiment.evaluate(episodes=args.episodes, render=args.render)

