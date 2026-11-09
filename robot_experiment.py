"""Train and evaluate the robot with MASB3."""

import argparse
from collections import deque
import json
from pathlib import Path

import numpy as np

import torch
from masb3.algos import get_algo
from masb3.base import MultiAgentSharedVecEnv
from masb3.utils import ma_evaluate, ma_train
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize

from waste_collection_env import RobotMAEnv


class RolloutRewardCallback(BaseCallback):
    """Select checkpoints by mean return over a full window of training episodes."""

    def __init__(self, output, window=100):
        super().__init__()
        if window < 1:
            raise ValueError('window must be positive')
        self.output = Path(output)
        self.episode_rewards = deque(maxlen=window)
        self.best_checkpoint_reward = -np.inf

    def _on_training_start(self):
        self.output.mkdir(parents=True, exist_ok=True)

    def _on_step(self):
        # Monitor reports complete episode returns, including failures and timeouts.
        for done, info in zip(self.locals['dones'], self.locals['infos']):
            if done and 'episode' in info:
                self.episode_rewards.append(float(info['episode']['r']))
        return True

    def _on_rollout_end(self):
        count = len(self.episode_rewards)
        self.logger.record('rollout/best_selection_episodes', count)
        if count < self.episode_rewards.maxlen:
            return
        mean_reward = float(np.mean(self.episode_rewards))
        self.logger.record('rollout/best_selection_mean_reward', mean_reward)
        if mean_reward > self.best_checkpoint_reward:
            # This callback runs before PPO updates the policy for this rollout.
            self.model.save(self.output / 'best_model')
            normalizer = self.model.get_vec_normalize_env()
            if normalizer is not None:
                normalizer.save(self.output / 'best_model_vecnormalize.pkl')
            self.best_checkpoint_reward = mean_reward
            (self.output / 'best_model_metrics.json').write_text(json.dumps({
                'metric': 'rollout_mean_episode_reward',
                'mean_reward': mean_reward, 'episodes': count,
                'timesteps': self.num_timesteps,
            }, indent=2) + '\n')
            print(f'New best rollout mean reward: {mean_reward:.2f} ({count} episodes)')
        self.logger.record('rollout/best_checkpoint_mean_reward', self.best_checkpoint_reward)


class RolloutPPO(get_algo("sb3", "PPO")):
    """Attach best-model selection to MASB3's existing training callbacks."""

    def __init__(self, *args, best_output=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_output = best_output

    def learn(self, *args, callback=None, **kwargs):
        if self.best_output is None:
            return super().learn(*args, callback=callback, **kwargs)
        rollout_callback = RolloutRewardCallback(self.best_output)
        callbacks = list(callback) if isinstance(callback, list) else ([callback] if callback else [])
        callbacks.append(rollout_callback)
        result = super().learn(*args, callback=callbacks, **kwargs)
        if len(rollout_callback.episode_rewards) < 100:
            print('Best model not selected: fewer than 100 completed episodes. '
                  'The final model will be saved as final_model.zip.')
        return result


class SuccessVecEnv(MultiAgentSharedVecEnv):
    def step_wait(self):
        obs, rewards, dones, infos = super().step_wait()
        for done, info in zip(dones, infos):
            if done:
                info["is_success"] = self.shared_env.info["is_success"]
        return obs, rewards, dones, infos


class RobotExperiment:
    def __init__(self, model_path=None, sorting=False):
        self.sorting = sorting
        self.model_path = model_path or (
            "models/ppo_sorting/final_model" if sorting else "models/ppo_go/final_model"
        )

    def model_directory(self, agent):
        return Path("models/ppo_drone") if agent == "drone" else Path(self.model_path).parent

    def resolve_model(self, agent, model=None, best=False):
        if model is not None:
            checkpoint = Path(model)
            if checkpoint.is_dir():
                checkpoint = checkpoint / ("best_model.zip" if best else "final_model.zip")
            elif best:
                checkpoint = checkpoint.with_name("best_model.zip")
            elif checkpoint.suffix != ".zip":
                checkpoint = Path(str(checkpoint) + ".zip")
        else:
            directory = self.model_directory(agent)
            runs = sorted((directory / "runs").glob("*/final_model.zip"),
                          key=lambda path: path.stat().st_mtime_ns)
            checkpoint = runs[-1] if runs else (
                directory / "final_model.zip" if agent == "drone" else Path(self.model_path)
            )
            if best:
                checkpoint = checkpoint.with_name("best_model.zip")
            elif checkpoint.suffix != ".zip":
                checkpoint = Path(str(checkpoint) + ".zip")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return checkpoint

    def train(self, agent="robot", steps=600_000, resume=False, model=None, two_robots=False,
              drone_model=None):
        if steps < 1:
            raise ValueError("steps must be positive")
        if model is not None and not resume:
            raise ValueError("Use --resume with --model when training")
        checkpoint = self.resolve_model(agent, model) if resume else None
        drone_checkpoint = self.resolve_model("drone", drone_model) if drone_model is not None else None
        runs_directory = self.model_directory(agent) / "runs"
        runs_directory.mkdir(parents=True, exist_ok=True)
        numbers = [int(path.name[4:]) for path in runs_directory.iterdir()
                   if path.is_dir() and path.name.startswith("run_") and path.name[4:].isdigit()]
        run_number = max(numbers, default=0) + 1
        while True:
            run_id = f"run_{run_number:03d}"
            output = runs_directory / run_id
            try:
                output.mkdir(exist_ok=False)
                break
            except FileExistsError:
                run_number += 1
        self.last_run_directory = output
        robot_env = RobotMAEnv(drone_training=agent == "drone", sorting=self.sorting,
                               two_robots=two_robots, drone_search=drone_checkpoint is not None)
        drone = robot_env.env.drone
        (output / "config.json").write_text(json.dumps({
            "agent": agent, "requested_steps": steps,
            "sorting": self.sorting,
            "two_robots": two_robots,
            "drone_model": str(drone_checkpoint.resolve()) if drone_checkpoint else None,
            "team_bonus": robot_env.env.team_bonus if self.sorting and two_robots else 0,
            "robot_observation_size": robot_env.env.observation_space.shape[0],
            "robot_avoidance": {
                "safe_distance": robot_env.env.robot.robot_safe_distance,
                "proximity_weight": robot_env.env.robot.robot_proximity_weight,
                "collision_penalty": robot_env.env.robot.robot_collision_penalty,
            } if self.sorting and two_robots else None,
            "resume_from": str(checkpoint.resolve()) if checkpoint else None,
            "robot_model": str(Path(self.model_path).resolve()) if agent == "drone" else None,
            "drone_search": {
                "speed": drone.speed, "min_search_speed": drone.min_search_speed,
                "boundary_penalty": drone.boundary_penalty,
                "stagnation_penalty": drone.stagnation_penalty,
            } if agent == "drone" else None,
        }, indent=2) + "\n")
        print(f"Training output: {output}")
        if checkpoint:
            print(f"Resuming from: {checkpoint}")
        torch.set_num_threads(1)
        robot_env.reset(seed=0)
        learning_ids = list(robot_env.env.robots) if agent == "robot" else [agent]
        env = SuccessVecEnv(robot_env, learning_agent_ids=learning_ids)
        env = VecMonitor(env)
        normalization_path = checkpoint.with_name(checkpoint.stem + "_vecnormalize.pkl") if checkpoint else None
        if normalization_path is not None and normalization_path.is_file():
            env = VecNormalize.load(normalization_path, env)
            env.training = True
        else:
            env = VecNormalize(env, norm_obs=False, norm_reward=True)
        agent_env_map = {agent: env}
        algo = get_algo("sb3", "PPO")
        params = {
            "policy": "MlpPolicy", "seed": 0, "device": "cpu", "verbose": 1,
            "tensorboard_log": "./tensorboard_logs/",
            "learning_rate": 0.0003 if agent == "drone" else 0.0003,
        }
        model_algo_map = {agent: (RolloutPPO, {**params, "best_output": str(output)})}
        models_to_load = None
        if agent == "drone":
            agent_env_map["robot"] = MultiAgentSharedVecEnv(robot_env, ["robot"])
            model_algo_map["robot"] = (algo, params)
            models_to_load = {"robot": self.model_path}
        if drone_checkpoint is not None:
            agent_env_map["drone"] = MultiAgentSharedVecEnv(robot_env, ["drone"])
            model_algo_map["drone"] = (algo, params)
            models_to_load = {"drone": str(drone_checkpoint)}
        if resume:
            models_to_load = {**(models_to_load or {}), agent: str(checkpoint)}
        try:
            models = ma_train(
                ma_env=robot_env,
                agent_env_map=agent_env_map,
                model_algo_map=model_algo_map,
                models_to_train=[agent],
                models_to_load=models_to_load,
                total_timesteps_per_model=steps,
                training_iterations=1,
                tb_log_suffix=f"{agent}_{'sorting_' if self.sorting else ''}{run_id}",
            )
            env.save(output / "final_model_vecnormalize.pkl")
            models[agent].save(output / "final_model.zip")
            print(f"Final model saved: {output / 'final_model.zip'}")
            return models
        finally:
            env.close()

    def evaluate(self, episodes=10, render=False, agent="robot", best=False, model=None,
                 debug_actions=False, two_robots=False, drone_model=None):
        checkpoint = self.resolve_model(agent, model, best)
        print(f"Evaluating: {checkpoint}")
        robot_path = checkpoint if agent == "robot" else self.model_path
        robot_model = get_algo("sb3", "PPO").load(robot_path, device="cpu")
        models = {"robot": robot_model}
        if agent == "drone":
            models["drone"] = get_algo("sb3", "PPO").load(checkpoint, device="cpu")
        if drone_model is not None:
            drone_checkpoint = self.resolve_model("drone", drone_model)
            models["drone"] = get_algo("sb3", "PPO").load(drone_checkpoint, device="cpu")
        env = RobotMAEnv(
            render_mode="human" if render else None, drone_training=agent == "drone",
            finish_on_discovery=False,
            two_robots=two_robots,
            sorting=self.sorting,
            drone_search=drone_model is not None,
        )
        env.env.drone.debug_actions = debug_actions
        env.set_agent_models(models)
        try:
            avg_agent, avg_model = ma_evaluate(
                env, models, total_episodes=episodes, verbose=True,
            )
            print(f"Average reward per agent: {avg_agent}")
            print(f"Average reward per model: {avg_model}")
            success_rate = sum(env.episode_successes) / len(env.episode_successes)
            print(f"success_rate: {success_rate:.2%}")
            return avg_agent, avg_model
        finally:
            env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "evaluate"])
    parser.add_argument("--agent", choices=["robot", "drone"], default="robot")
    parser.add_argument("--steps", type=int, default=600_000)
    parser.add_argument("--resume", action="store_true",
                        help="Continue from --model or the latest completed run (legacy final_model if no runs exist)")
    parser.add_argument("--model", type=Path,
                        help="Checkpoint file or run directory to resume or evaluate")
    parser.add_argument("--best", action="store_true",
                        help="Evaluate the selected agent's best_model instead of final_model")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--two-robots", action="store_true",
                        help="Train/evaluate two robots sharing the same PPO policy")
    parser.add_argument("--sorting", action="store_true",
                        help="Train/evaluate plastic/cardboard specialists with semantic rays")
    parser.add_argument("--drone-model", type=Path,
                        help="Use a frozen drone checkpoint to locate waste before dispatching sorting robots")
    parser.add_argument("--debug-actions", action="store_true",
                        help="Print drone actions, requested/actual positions, speed and boundary blocking during evaluation")
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()
    if args.debug_actions and (args.mode != "evaluate" or args.agent != "drone"):
        parser.error("--debug-actions requires evaluate --agent drone")
    if args.sorting and args.agent != "robot":
        parser.error("--sorting requires --agent robot")
    if args.drone_model is not None and (not args.sorting or args.agent != "robot"):
        parser.error("--drone-model requires --sorting --agent robot")

    if args.mode == "train" and args.model is not None and not args.resume:
        parser.error("--model requires --resume when training")
    if args.mode == "train" and args.best:
        parser.error("--best is for evaluation; resume a best checkpoint with --model path/to/best_model.zip")
    experiment = RobotExperiment(sorting=args.sorting)
    if args.mode == "train":
        experiment.train(agent=args.agent, steps=args.steps, resume=args.resume, model=args.model,
                         two_robots=args.two_robots, drone_model=args.drone_model)
    else:
        experiment.evaluate(episodes=args.episodes, render=args.render, agent=args.agent,
                            best=args.best, model=args.model, debug_actions=args.debug_actions,
                            two_robots=args.two_robots, drone_model=args.drone_model)

