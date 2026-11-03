# marl-waste-collection

A single robot learns to reach a target using PPO and MuJoCo.
The target is a plastic bottle lying on the floor, drawn with simple MuJoCo shapes.
Robot and target spawn randomly in an 6 × 6 m square at each reset, at least 0.75 m apart.
The displayed floor is 8 × 8 m, with robot joint limits of ±3.5 m on both axes.
In robot-only training the robot knows the target coordinates from the beginning.
With the drone active, target coordinates stay hidden until the drone detects it.

The `go` signal starts at 0 for a random waiting period of 0–100 steps,
then becomes 1. The robot learns to wait with `go=0` and move with `go=1`.
A drone at a fixed 3 m altitude carries nine rays originating at its bottom center and spreading downward
to a 3 × 3 grid on the ground with 0.6 m spacing at the initial altitude. Red markers show the measured segments when rendering.
Sensors use `Raycaster` from the local `raycast.py`, copied from the raycast1 project.
Distances and hit geometry IDs are exposed in step info as `drone_ray_distances`
and `drone_ray_geomids` (`-1` means no hit within 5 m). Drone-only training uses the saved robot policy without updating it.
Robot observations, actions, and rewards are unchanged.

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
`models/ppo_go/runs/<run-id>/final_model.zip`. Evaluation loads the latest completed run.
For evaluation without a window: `python robot_experiment.py evaluate --episodes 100`.

## Files

- `agents.py`: `RobotAgent` handles robot observations, actions, waiting, and
  navigation reward; `DroneAgent` handles drone movement, ray sensors, observations,
  exploration reward, and discovery. Both use the same MuJoCo simulation.
- `waste_collection_env.py`: shared MuJoCo simulation (`WasteCollectionEnv`) and
  MASB3 interface (`RobotMAEnv`), coordinating reset, communication, and episode
  termination. Agent state is accessible through `env.robot` and `env.drone`.
- `scene.xml`: robot, floor, target, and motors.
- `robot_experiment.py`: `RobotExperiment` class with `train()` and `evaluate()`,
  using MASB3's `ma_train` and `ma_evaluate`.

MASB3 comes from `Multiagents/masb3`. The environments in
`Multiagents/masb3-experiments` serve as examples; they are not required here.
PPO collects 2,048 steps per rollout with this single learning agent.
Robot observations and rewards retain their existing format. Episodes last at
most 2,000 steps.
Existing robot SB3 PPO checkpoints remain compatible because robot observations
and actions are unchanged. Drone checkpoints need the exploration-map observation
format described below. MASB3 also writes training checkpoints under `checkpoints/`.

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
An episode lasts at most 2,000 steps, including waiting. Each step simulates 0.05 s.
Waiting is learned, so small movements can remain before the go signal.

## Drone search reward

The environment reports `drone_reward`, `drone_reward_components`, and
`drone_found_target` in step info. Before discovery, each step costs -0.05;
entering a new 0.5 m grid cell earns +0.5 once per episode. The initial cell is
already visited at reset. Detection requires a ray to hit a bottle geometry.
The first detection earns `20 * max(0, 1 - (steps - 1) / max_steps)`;
there is no time penalty on that step. All drone rewards stop after discovery.
During search, commands clipped by the movement limits also receive a
`boundary` cost, proportional to the rejected displacement and capped at -0.2.
Moving inward or along a boundary has no boundary cost. Actual speeds below
0.3 m/s receive a linear `stagnation` cost: -0.05 when stationary, falling to
zero at 0.3 m/s. These costs are additional to the -0.05 time cost and are zero
on discovery and afterwards. A fully blocked outward command costs -0.3 per
step; staying still costs -0.1. The maximum speed remains 1 m/s per axis.
Step info exposes `drone_speed` in m/s and `drone_blocked_fraction` in [0, 1].
In drone mode, discovery ends the episode; robot success does not end it.
The robot's training reward remains unchanged.

Drone observations are available through `get_drone_observation()` and
`info["drone_observation"]` on reset and step. The 219-element float32 vector is:
shared communication [0:3] = [found, bottle_x / 5, bottle_y / 5], drone XY / 5
[3:5], nine ray pairs [5:23] = [normalized distance, is_bottle], then the
flattened exploration map [23:].
Communication is zero until first detection, then stores the bottle position
for the episode. Missing rays have normalized distance 1.

The map is a 14 × 14 grid covering [-3.5, 3.5] m on both axes, with 0.5 m cells.
Each entry is 0 for an unvisited cell and 1 for a cell the drone has entered.
Rows follow increasing world Y and columns follow increasing world X, starting
at -3.5 m. The positive boundary belongs to the last row or column. Recover the
grid with `observation[23:].reshape(14, 14)`. The drone's current position stays
in [3:5]. The grid records visited positions, not the whole area seen by the rays;
it never reveals an undetected bottle. Rays still determine bottle detection.

At reset only the starting cell is visited. Every step updates the map before
returning the next observation, and the exploration bonus uses the same map to
reward each cell once. `info["drone_visited_map"]` provides a separate 2D copy on
reset and step for inspection. Robot observations remain unchanged.

Drone policies trained with 23 observations cannot use the new 219-element
input. Start a fresh drone training without `--resume`:

```bash
python robot_experiment.py train --agent drone --steps 600000
```

Previous runs are preserved. Resume or evaluate only drone models trained with
the exploration map. The saved robot policy can still be used as before.

## Train only the drone

```bash
python robot_experiment.py train --agent drone --steps 600000
python robot_experiment.py evaluate --agent drone --render
```

The saved robot policy at `models/ppo_go/final_model.zip` controls the robot
without learning. Only the drone is trained, and its policy is saved separately
at `models/ppo_drone/runs/<run-id>/final_model.zip`. Drone actions command X/Y velocity up to
1 m/s per axis, with fixed altitude 3 m and positions limited to [-3.5, 3.5] m.
Drone spawn is randomized in [-3, 3] m on reset. A drone episode ends on bottle
detection or after 2,000 steps. Use `--agent robot` for the original robot task.

Reward changes affect learning, so a saved policy needs further training to
learn to avoid blocked commands. Map-based checkpoints retain their 219-element
observation format and can be resumed. Every new run records the speed and
search penalty parameters under `drone_search` in `config.json`; older
checkpoints are preserved. Use `--model` to choose the new run explicitly when
evaluating.

## TensorBoard and success rate

Training writes TensorBoard events to `tensorboard_logs/`, including
`rollout/success_rate` (the fraction of successful episodes in the last 100
completed episodes), rewards, and PPO losses. The training console also shows
success rate once episodes finish. Evaluation prints the success rate across
all requested episodes. Success means bottle detection in drone mode and target
reaching after the go signal in robot mode; timeouts count as failures.

```bash
tensorboard --logdir tensorboard_logs
```

Open the local URL printed by TensorBoard. Logging uses TensorBoard with the
existing PyTorch PPO implementation.

Training also selects `best_model.zip` in the same directory as each agent's
`final_model.zip`. As in `raycast_update_copy`, selection runs at rollout end,
before the PPO update, using the mean unnormalized return of the last 100
completed training episodes (including failures and timeouts). The first best
model requires 100 completed episodes; later saves require a strictly higher
mean. `best_model_metrics.json` records the reward, episode count, and timestep.
TensorBoard also records `rollout/best_selection_mean_reward`,
`rollout/best_selection_episodes`, and `rollout/best_checkpoint_mean_reward`.
Evaluation continues to load `final_model.zip` by default.

To continue training the saved drone policy for 2,000 additional timesteps:

```bash
python robot_experiment.py train --agent drone --steps 2000 --resume
```

`--resume` loads the latest completed run (or the legacy `final_model.zip` if no
runs exist), including policy and
optimizer state, and saves the updated policy in a new run directory. The source checkpoint is preserved. The robot stays
frozen during drone training. PPO completes full rollouts, so with the default
2,048 steps per rollout, a request for 2,000 collects 2,048 steps. Without
`--resume`, the selected agent starts fresh. Reward normalization statistics from
older runs were not saved and therefore start fresh when resuming those models.

Evaluate the best drone checkpoint with
`python robot_experiment.py evaluate --agent drone --render --best`.
The robot still uses its final policy. The best checkpoint must already exist;
training selects one after at least 100 completed episodes.

## Separate model versions

Every training, including `--resume`, creates a numbered directory (`run_001`,
`run_002`, and so on) under
`models/ppo_drone/runs/` or `models/ppo_go/runs/`. Each run stores its final
model, optional best model, normalization statistics, and `config.json` with
the source checkpoint. TensorBoard and periodic checkpoints use the run ID too.
Existing checkpoints are never replaced by a later training.

Choose a previous version explicitly:

```bash
python robot_experiment.py train --agent drone --steps 2000 --resume --model models/ppo_drone/runs/RUN_ID/final_model.zip
python robot_experiment.py evaluate --agent drone --render --model models/ppo_drone/runs/RUN_ID
python robot_experiment.py evaluate --agent drone --render --model models/ppo_drone/runs/RUN_ID --best
```

A run directory selects its final model, or its best model with `--best`.
Without `--model`, evaluation and resume select the latest completed run, not
the run with the highest success rate. To select the original checkpoint, use
`--model models/ppo_drone/final_model.zip`. Best selection still uses the
last 100 episodes of each individual training; a short run may have no best.

## Drone-to-robot target communication

With `--agent drone`, the robot observation keeps its seven elements, but the
relative target vector is `[0, 0]` and `go=0` until detection. The drone writes
`[found, target_x / 5, target_y / 5]` into `shared_comm`; the robot then receives
the relative vector reconstructed from that broadcast and `go=1`. Communication
persists until reset. The random go delay applies only to robot-only mode.

Drone training still ends at discovery. Drone evaluation continues after
discovery until the robot reaches the target or the episode times out, so the
reported success rate measures completion by the robot. The same render command
shows search followed by navigation:

```bash
python robot_experiment.py evaluate --agent drone --render --model models/ppo_drone/runs/run_001
```

Robot checkpoint shapes remain compatible. The robot now receives hidden
target coordinates while waiting, unlike its earlier training observations;
its waiting behavior under this input should be evaluated.

As in the thesis (section 6.2.1), with the drone active, robot motor commands
are forced to zero until the drone broadcasts detection. The policy controls
the robot starting with the next action after discovery. This gate is enforced
by the environment and requires no retraining. Robot-only mode keeps policy
control during its original waiting exercise.

The robot has a yellow-and-black quadruped appearance inspired by Spot, with
four bent legs, rubber feet, and a front sensor panel. These parts are decorative:
the existing XY controls, collision cylinder, mass, and inertia are preserved.
The legs do not articulate; locomotion still uses the existing sliding joints.

## Two robots and head-mounted rays

To preview two robots using the existing navigation policy:

```bash
python robot_experiment.py evaluate --two-robots --render --episodes 1
```

The yellow `robot` and blue `robot_2` have independent XY controls and share the
selected navigation policy. Both advance in one 0.05 s physics step. Their random
starts are at least 1 m apart and 0.75 m from the target. Add `--agent drone` to
wait for the drone's broadcast. Without `--two-robots`, there is one robot.
Both training and evaluation support the two-robot mode.

Each robot, including in single-robot mode, has nine rays originating at the
front of its head. Three horizontal angles (-60, 0, +60 degrees) combine with
three downward pitches (0, 15, 30 degrees), giving nine rays with a 2 m range.
The rays follow the head's orientation and ignore the emitting robot and floor.
They are yellow for `robot`, cyan for `robot_2`, and red for the drone.

Reset and step info expose independent sensor snapshots:

```python
sensors = info["robot_sensors"]["robot_2"]  # or "robot"
distances = sensors["distances"]          # nine distances; -1 for no hit
geomids = sensors["geomids"]              # nine geometry IDs; -1 for no hit
types = sensors["types"]                  # 0 plastic, 1 cardboard, 2 reserved, 3 obstacle, 4 robot, 5 none
```

Without `--sorting`, the PPO input remains seven navigation values. There is
one shared bottle. In two-robot mode both must reach within
0.25 m of it with `go=1`. Each robot receives its success bonus once, then zero
motor commands and reward while the other finishes. This differs from the
historical single-robot evaluation task.

## Single-robot material training

```bash
python robot_experiment.py train --sorting --steps 600000
python robot_experiment.py evaluate --sorting --render --episodes 10
```

Each episode has one robot and one visible waste item. Robot specialization
and waste material are independently sampled as plastic or paper. Positions,
the random go delay and the two XY motor actions follow the original task.
Robot and waste centers start at least 3 m apart, outside initial ray detection.
The paper object replaces the bottle in paper episodes. There are no static
obstacles or second robot in this training mode.

The float32 policy observation reserves three materials and contains 76 values:

| Indices | Meaning |
|---|---|
| 0:7 | Original navigation observation, including go |
| 7:10 | Robot specialization: plastic/cardboard/material_c one-hot |
| 10:13 | Material previously detected: three bits, all zero if unknown |
| 13:76 | Nine rays: [distance/2, plastic, cardboard, material_c, obstacle, robot, none] |

`material_c` is reserved: episodes still sample only plastic and cardboard.
No third waste object or robot has been added. To activate it later, add its
scene geometry, map its ray category to 2, and enable it in the spawn sampling.
The policy observation will stay at 76 values, but the policy will still need
training examples of that material to learn the new behavior.

Ray categories are one-hot. A missing hit has normalized distance 1. Material
is revealed only by actual ray intersections and remembered for the episode;
the simulator does not supply the hidden material directly to the policy.
This is simulated semantic sensing, not learned image classification.

After go, matching waste is automatically collected when the robot is within
0.25 m and has identified it. Incompatible waste is automatically reported at
the first identification outside that radius. Both outcomes end the episode
successfully with a +10 bonus, in addition to the existing navigation reward.
Entering the 0.25 m radius of incompatible waste ends the episode as a failure
with an extra -10 penalty. The limit is evaluated on the robot-center distance;
it is a training objective, not a movement clamp. Timeouts are failures.
Reporting is a programmed communication rule, not an additional learned action.

In sorting mode `info["shared_comm"]` contains
`[found, target_x/5, target_y/5, material]`, with material 0 unknown, 1 plastic,
2 cardboard, 3 reserved. It is zero until an incompatible item is reported, and resets each
episode. Step info also exposes `specialization`, `target_type`, `collected`,
`reported`, `wrong_target` and `is_success`. Success rate combines successful
collections and reports. The drone policy is inactive unless `--drone-model` is supplied.

Models are saved separately under `models/ppo_sorting/runs/`, and TensorBoard
names include `sorting`. Start a new model: the old seven-input and 65-input
checkpoints cannot be resumed directly with the 76-input observation. Subsequent sorting runs can
use the existing `--resume`, `--model` and `--best` options. Training uses one
robot unless `--two-robots` is set; legacy modes remain selectable without `--sorting`.

## Two sorting robots with the shared PPO policy

```bash
python robot_experiment.py evaluate --sorting --two-robots --render --episodes 10
```

MASB3's `BaseMAEnv` registers `robot` and `robot_2` with the same
`model_name="robot"`, as in the Multiagents examples. `set_agent_models` binds
both to the single loaded PPO model; `ma_evaluate` batches their observations
into one `predict` call. Each receives its own 76-value observation and XY action.
A single-robot sorting checkpoint trained with 76 inputs can be used directly.

The yellow `robot` always specializes in plastic, and the blue `robot_2` in
cardboard. Cardboard uses the existing paper geometry (material index 1), named
`cardboard` in info. Single- and two-robot modes have the same policy input layout.
Single-robot training still randomizes the specialization. There is still one
randomly typed waste item, at least 3 m from both robots. Only the closest robot
is initially assigned to it. Assignment uses center-to-target Euclidean distance;
an exact tie selects `robot`. The assignment stays fixed while it approaches.
The assigned robot receives `go=1`; the other receives `go=0` and its requested
actions are ignored. This replaces the random go delay only in two-robot sorting.

An incompatible robot reports and stops, transferring the assignment to the
closest available compatible robot. Its message supplies the waste material
to its teammate's existing observation memory, including when that teammate has
not seen the waste with its own rays. A report alone no longer ends a two-robot
episode: success requires collection by the compatible robot. Either robot
entering the wrong collection radius causes failure; timeout also remains a
failure. `info["robot_specializations"]` identifies the roles by agent name.
`info["active_robot"]` identifies the current assignee, or is `None` after task
termination. Unassigned robots are held at their current positions with zero
velocity, including after a handoff; this prevents inertia or contact from
moving a waiting robot. The PPO observation remains 76 values.

## Joint training and team reward

```bash
python robot_experiment.py train --sorting --two-robots --steps 600000
python robot_experiment.py train --sorting --two-robots --steps 600000 --resume
```

The training wrapper uses MASB3's `MultiAgentSharedVecEnv` with both robot IDs
as learning agents. Their transitions update a single shared PPO policy, as in
the push example. A 76-input single-robot checkpoint can be resumed with two
robots; seven-input or 65-input checkpoints require adaptation first.

Each robot receives its individual reward plus a team component:

- Individual: existing waiting/navigation penalties, +10 for its first correct
  collection or incompatible-waste report, -10 for entering the wrong radius.
  In two-robot sorting, an unassigned robot has zero individual reward while
  waiting, unless it violates the wrong collection radius.
- Team: +10 to each robot when compatible collection completes the task without
  a wrong-radius violation. This includes a robot that already reported and
  stopped. Reports alone, failures and timeouts earn no team bonus.

The team bonus is paid once per episode and applies only to two-robot sorting.
It is set by `WasteCollectionEnv.team_bonus`. Step info exposes
`individual_rewards`, the shared scalar `team_reward`, and combined
`robot_rewards`, which are the values consumed by MASB3 and PPO. Config files
record the robot count, observation size and team bonus.

`--steps` counts learning transitions: with two robots, one world step supplies
two transitions. PPO still collects full rollouts (2,048 world steps, or 4,096
transitions with two robots). Episode limits remain measured in world steps.
Best-model selection averages the last 100 completed agent-episode returns;
with two agents this can correspond to 50 shared episodes.

## Drone search before robot dispatch

```bash
python robot_experiment.py evaluate --sorting --two-robots --render --episodes 10 \
  --drone-model models/ppo_drone/runs/border_penalty_1.4mln/best_model.zip
```

With `--drone-model`, both robots wait with go=0 and a hidden target vector until
the drone detects waste. The drone's rays recognize both the plastic bottle and
the cardboard box as generic waste; they do not distinguish material. Its old
`is_bottle` sensor flag now means any waste in sorting mode. The 219-value drone
observation retains its format, so existing map-based drone policies can be loaded.

Detection broadcasts `[1, target_x/5, target_y/5, 0]`. Material 0 means unknown;
it must not be interpreted as a robot's classification. The nearest robot is
assigned only after that message. The drone then stops moving. Material is
subsequently recognized by robot rays and, when incompatible, reported with the
fourth field set to 1 for plastic or 2 for cardboard. The compatible robot
receives the handoff. Drone discovery alone does not terminate the episode or
pay the robot team bonus: compatible collection is still required.

The same `--drone-model` option can be used with `train --sorting --two-robots`.
Only the shared robot policy is updated; the loaded drone policy stays frozen.
`--agent drone` remains the separate legacy drone-training mode.
