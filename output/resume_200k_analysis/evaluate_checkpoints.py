"""Controlled, search-only checkpoint comparison using MASB3 ma_evaluate."""

import argparse
from contextlib import redirect_stdout
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from masb3.algos import get_algo
from masb3.utils import ma_evaluate
from waste_collection_env import RobotMAEnv


class SearchEvaluationEnv(RobotMAEnv):
    def __init__(self, seed_start=0):
        super().__init__(drone_training=True, finish_on_discovery=True)
        self.records = []
        self.episode_index = 0
        self.seed_start = seed_start

    def reset(self, seed=None):
        self.evaluation_seed = self.seed_start + self.episode_index
        torch.manual_seed(1000 + self.evaluation_seed)
        self.episode_index += 1
        self.total_reward = 0.0
        self.blocked_steps = 0
        self.stationary_steps = 0
        self.speed_sum = 0.0
        self.components = {}
        observation, info = super().reset(seed=self.evaluation_seed)
        self.initial_state = {
            'drone_xy': self.env.data.mocap_pos[self.env.drone.mocap_id, :2].tolist(),
            'robot_xy': self.env.data.qpos[:2].tolist(),
            'target_xy': self.env.target.tolist(),
        }
        return observation, info

    def sync_wait_for_actions_completion(self):
        super().sync_wait_for_actions_completion()
        info = self.info
        self.total_reward += info['drone_reward']
        self.blocked_steps += int(info['drone_blocked_fraction'] > 1e-8)
        self.stationary_steps += int(info['drone_speed'] < 1e-8)
        self.speed_sum += info['drone_speed']
        for name, value in info['drone_reward_components'].items():
            self.components[name] = self.components.get(name, 0.0) + value
        if self.terminated or self.truncated:
            self.records.append({
                'seed': self.evaluation_seed, 'steps': self.env.steps,
                'found': info['drone_found_target'], 'reward': self.total_reward,
                'blocked_steps': self.blocked_steps, 'stationary_steps': self.stationary_steps,
                'mean_speed': self.speed_sum / self.env.steps,
                'visited_cells': int(self.env.drone.visited_map.sum()),
                'reward_components': self.components.copy(),
                'initial_state': self.initial_state,
            })
            if len(self.records) % 10 == 0:
                print(f'{len(self.records)} episodes completed', file=sys.stderr, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('label')
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--seed-start', type=int, default=0)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    torch.set_num_threads(1)
    algo = get_algo('sb3', 'PPO')
    robot = algo.load(PROJECT / 'models/ppo_go/final_model', device='cpu')
    # Match MASB3 training: the non-learning robot follows its fixed policy deterministically.
    robot.predict = partial(robot.predict, deterministic=True)
    drone = algo.load(args.checkpoint, device='cpu')
    env = SearchEvaluationEnv(seed_start=args.seed_start)
    try:
        with open(os.devnull, 'w') as sink, redirect_stdout(sink):
            ma_evaluate(env, {'robot': robot, 'drone': drone}, total_episodes=args.episodes)
        episodes = env.records
    finally:
        env.close()
    steps = sum(e['steps'] for e in episodes)
    found = sum(e['found'] for e in episodes)
    summary = {
        'found': found, 'episodes': len(episodes),
        'mean_reward': float(np.mean([e['reward'] for e in episodes])),
        'mean_steps': float(np.mean([e['steps'] for e in episodes])),
        'mean_discovery_steps_when_found': (
            sum(e['steps'] for e in episodes if e['found']) / found if found else None
        ),
        'mean_speed': sum(e['mean_speed'] * e['steps'] for e in episodes) / steps,
        'blocked_step_percent': 100 * sum(e['blocked_steps'] for e in episodes) / steps,
        'stationary_step_percent': 100 * sum(e['stationary_steps'] for e in episodes) / steps,
        'mean_visited_cells': float(np.mean([e['visited_cells'] for e in episodes])),
        'mean_reward_components': {
            k: float(np.mean([e['reward_components'][k] for e in episodes]))
            for k in episodes[0]['reward_components']
        },
    }
    result = {
        'label': args.label, 'checkpoint': str(args.checkpoint),
        'sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        'method': 'MASB3 ma_evaluate, search ends on detection or 2000 steps, '
                  f'env seeds {args.seed_start}..{args.seed_start + args.episodes - 1}, '
                  'Torch seed 1000 + env seed per episode; drone stochastic, robot deterministic',
        'seed_start': args.seed_start, 'requested_episodes': args.episodes,
        'summary': summary, 'episodes': episodes,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / (args.label + '.json')
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(args.label, json.dumps(summary), flush=True)
