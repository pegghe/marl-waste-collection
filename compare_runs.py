"""Compare drone checkpoints on identical environment and action seeds."""
import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time


def evaluate_checkpoint(task):
    checkpoint, episodes, output = task
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.utils import set_random_seed
    from waste_collection_env import RobotMAEnv

    torch.set_num_threads(1)
    models = {
        'robot': PPO.load('models/ppo_go/final_model.zip', device='cpu'),
        'drone': PPO.load(checkpoint, device='cpu'),
    }
    env = RobotMAEnv(drone_training=True, finish_on_discovery=False)
    for key, model in models.items():
        space = env.agents[key].observation_space
        if model.observation_space.shape != space.shape:
            env.close()
            return dict(checkpoint=checkpoint, error=f'Observation shape {model.observation_space.shape}; environment {space.shape}')
    records = []
    started = time.monotonic()
    try:
        for seed in range(episodes):
            set_random_seed(seed)
            obs, _ = env.reset(seed=seed)
            start = env.env.robot.start_position.tolist()
            target = env.env.target.tolist()
            totals = {key: 0.0 for key in models}
            for step in range(1, env.env.max_steps + 1):
                actions = {
                    key: model.predict(obs[key], deterministic=False)[0]
                    for key, model in models.items()
                }
                obs, rewards, terminated, truncated, info = env.step_all(actions)
                for key in totals:
                    totals[key] += float(rewards[key])
                if terminated or truncated:
                    break
            records.append(dict(seed=seed, success=bool(info['is_success']),
                                steps=step, robot_reward=totals['robot'],
                                drone_reward=totals['drone'], start=start, target=target))
            if (seed + 1) % 10 == 0:
                print(f'{checkpoint}: {seed + 1}/{episodes}, successes={sum(r["success"] for r in records)}', flush=True)
        result = dict(checkpoint=checkpoint, episodes=episodes,
                      successes=sum(r['success'] for r in records),
                      success_rate=float(np.mean([r['success'] for r in records])),
                      mean_steps=float(np.mean([r['steps'] for r in records])),
                      mean_drone_reward=float(np.mean([r['drone_reward'] for r in records])),
                      elapsed_seconds=time.monotonic() - started, records=records)
        name = Path(checkpoint).parent.name + '_' + Path(checkpoint).stem
        (Path(output) / (name + '.json')).write_text(json.dumps(result, indent=2) + '\n')
        return result
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output', type=Path, default=Path('output/comparison_seed_0_99'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted(Path('models/ppo_drone/runs').glob('*/best_model.zip'))
    checkpoints += sorted(Path('models/ppo_drone/runs').glob('*/final_model.zip'))
    results = []
    pending = []
    for checkpoint in checkpoints:
        saved = args.output / (checkpoint.parent.name + '_' + checkpoint.stem + '.json')
        if saved.exists():
            result = json.loads(saved.read_text())
            if result['episodes'] == args.episodes:
                results.append(result)
                continue
        pending.append(checkpoint)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(evaluate_checkpoint, (str(p), args.episodes, str(args.output))) for p in pending]
        for future in as_completed(futures):
            result = future.result()
            if 'error' in result:
                print(result, flush=True)
                with (args.output / 'excluded.jsonl').open('a') as handle:
                    handle.write(json.dumps(result) + '\n')
                continue
            results.append(result)
            ranked = sorted(results, key=lambda r: r['success_rate'], reverse=True)
            fields = ['checkpoint', 'episodes', 'successes', 'success_rate', 'mean_steps', 'mean_drone_reward']
            with (args.output / 'summary.csv').open('w') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(ranked)
    fields = ['checkpoint', 'episodes', 'successes', 'success_rate', 'mean_steps', 'mean_drone_reward']
    with (args.output / 'summary.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(sorted(results, key=lambda r: r['success_rate'], reverse=True))
    # Check that each checkpoint actually saw identical robot/target starts.
    reference = [(r['start'], r['target']) for r in results[0]['records']]
    assert all([(r['start'], r['target']) for r in result['records']] == reference for result in results)
    print((args.output / 'summary.csv').read_text(), flush=True)


if __name__ == '__main__':
    main()
