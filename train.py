"""Train the robot with PPO and save the learned policy."""

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from waste_collection_env import WasteCollectionEnv


if __name__ == "__main__":
    torch.set_num_threads(1)
    env = make_vec_env(WasteCollectionEnv, n_envs=4, seed=0)
    # Normalize rewards only: the saved policy uses raw observations.
    env = VecNormalize(env, norm_obs=False, norm_reward=True)
    try:
        model = PPO("MlpPolicy", env, seed=0, device="cpu", verbose=1)
        model.learn(total_timesteps=600_000)
        model.save("models/ppo_go/final_model")
    finally:
        env.close()
