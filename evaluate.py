"""Run the trained robot on new random spawns."""

import argparse

from stable_baselines3 import PPO

from waste_collection_env import WasteCollectionEnv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    model = PPO.load("models/ppo_go/final_model.zip", device="cpu")
    env = WasteCollectionEnv(render_mode="human" if args.render else None)
    successes = 0
    try:
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=20_000 + episode)
            terminated = truncated = False
            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
            successes += int(info["is_success"])
        print(f"Targets reached: {successes}/{args.episodes}")
    finally:
        env.close()
