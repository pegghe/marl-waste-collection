# marl-waste-collection

A single robot learns to reach a target using PPO and MuJoCo.
The target is a plastic bottle lying on the floor, drawn with simple MuJoCo shapes.
Robot and target spawn randomly at each reset, at least 0.75 m apart.
The robot knows the target coordinates from the beginning.

The `go` signal starts at 0 for a random waiting period of 0–100 steps,
then becomes 1. The robot learns to wait with `go=0` and move with `go=1`.
There is no drone in this version.

## Run

```bash
pip install -r requirements.txt
python robot_experiment.py train
python robot_experiment.py evaluate --render
```

Run these commands from this project directory. Installation expects the professor's
library at `../Multiagents/masb3` and installs it in editable mode.

Training uses MASB3's `ma_train` and `MultiAgentSharedVecEnv` with one robot
in one simulation for approximately 600,000 steps and saves
`models/ppo_go/final_model.zip`. Evaluation loads this policy.
For evaluation without a window: `python robot_experiment.py evaluate --episodes 100`.

## Files

- `waste_collection_env.py`: random spawning, observations, rewards, and episode logic.
- `scene.xml`: robot, floor, target, and motors.
- `robot_ma_env.py`: adapts the MuJoCo environment to MASB3's `BaseMAEnv` interface.
- `robot_experiment.py`: `RobotExperiment` class with `train()` and `evaluate()`,
  using MASB3's `ma_train` and `ma_evaluate`.

MASB3 comes from `Multiagents/masb3`. The environments in
`Multiagents/masb3-experiments` serve as examples; they are not required here.
PPO collects 2,048 steps per rollout with this single learning agent.
Observations, rewards, and the 400-step episode limit are unchanged.
Existing SB3 PPO checkpoints remain compatible because observations and actions
are unchanged. MASB3 also writes training checkpoints under `checkpoints/`.

## Observation and actions

Observation: `[x, y, vx, vy, target_x - x, target_y - y, go]`.
Actions: two motor commands in `[-1, 1]`, along X and Y.

## Reward

- Step penalty: `-0.01` at every step.
- With `go=1`: subtract the distance to the target.
- With `go=0`: subtract the robot's speed and `5 * distance_from_start` to encourage waiting.
- On success: add `10`.

Existing checkpoints were trained with the previous reward. Run `python robot_experiment.py train`
to train a new policy with this reward.

Success means `go=1` and distance to the target ≤ 0.25 m.
An episode lasts at most 400 steps, including waiting. Each step simulates 0.05 s.
Waiting is learned, so small movements can remain before the go signal.
