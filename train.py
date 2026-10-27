"""Train the robot using the professor's MASB3 library."""

import torch
from masb3.algos import get_algo
from masb3.base import MultiAgentSharedVecEnv
from masb3.utils import ma_train
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize

from robot_ma_env import RobotMAEnv


if __name__ == "__main__":
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
        models["robot"].save("models/ppo_go/final_model")
    finally:
        env.close()
