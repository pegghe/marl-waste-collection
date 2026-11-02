# Drone boundary and speed check

Search-only comparison on seeds 0–9. Each policy receives the same initial positions and a Torch seed of 1000 + episode seed. Actions are stochastic, matching MASB3 evaluation. Search ends on detection or after 2,000 steps. The maximum speed stays at 1 m/s per axis.

| Checkpoint | Mean speed (m/s) | Steps with clipped commands | Stationary steps | Detections | Mean visited cells |
| --- | ---: | ---: | ---: | ---: | ---: |
| models/ppo_drone/runs/run_004/final_model.zip | 0.275 | 56.8% | 6.34% | 2/10 | 28.2 |
| models/ppo_drone/runs/run_004/best_model.zip | 0.507 | 31.7% | 5.46% | 5/10 | 29.6 |
| models/ppo_drone/runs/run_005/final_model.zip | 0.447 | 12.3% | 0.22% | 4/10 | 24.6 |

run_005 continues run_004/best_model.zip for 65,536 steps with boundary and stagnation costs. Existing checkpoints were preserved. New runs record search parameters in config.json.

The trial reduces boundary clipping and stationary steps. It is not evidence of a general improvement in bottle detection: on this small sample, run_004/best detected 5/10 and run_005/final detected 4/10. These are drone detections, not completed robot navigation episodes.

Evaluate the trained trial explicitly:

```bash
python robot_experiment.py evaluate --agent drone --render --model models/ppo_drone/runs/run_005
```

The 15 unit/integration tests pass. Per-episode data is stored in drone_boundary_comparison.json.
