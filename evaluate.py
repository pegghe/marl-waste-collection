"""Run the trained robot through the MASB3 environment interface."""

import argparse

from masb3.algos import get_algo

from robot_ma_env import RobotMAEnv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    model = get_algo("sb3", "PPO").load("models/ppo_go/final_model.zip", device="cpu")
    env = RobotMAEnv(render_mode="human" if args.render else None)
    successes = 0
    try:
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=20_000 + episode)
            terminated = truncated = False
            while not (terminated or truncated):
                action, _ = model.predict(obs["robot"], deterministic=True)
                obs, rewards, terminated, truncated, info = env.step_all({"robot": action})
            successes += int(info["is_success"])
        print(f"Targets reached: {successes}/{args.episodes}")
    finally:
        env.close()
