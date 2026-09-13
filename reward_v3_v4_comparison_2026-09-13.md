# Reward v3 vs v4 Evaluation

Runs were trained with seed 1 and evaluated with 100 episodes, 20 seconds per episode.

## Hanging Reset

| Algorithm | Reward | Success rate | Longest hold (s) | Upright fraction | Swing-up time (s) | Mean abs torque (Nm) | Action saturation | Episode return |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PPO | v3 | 0% | 0.00 | 0.00% | n/a | 5.545 | 85.10% | -134279.14 |
| PPO | v4 | 66% | 12.12 | 77.45% | 1.22 | 0.730 | 4.54% | 3930.88 |
| SAC | v3 | 100% | 18.28 | 91.39% | 1.72 | 0.317 | 0.00% | 4599.83 |
| SAC | v4 | 3% | 0.59 | 2.96% | 1.44 | 4.628 | 46.40% | -63826.75 |

## Upright Reset

| Algorithm | Reward | Success rate | Longest hold (s) | Upright fraction | First stable time (s) | Mean abs torque (Nm) | Action saturation | Episode return |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PPO | v3 | 0% | 0.04 | 0.24% | 0.02 | 5.615 | 89.37% | -138500.29 |
| PPO | v4 | 57% | 11.07 | 77.92% | 0.36 | 0.714 | 4.22% | 3971.93 |
| SAC | v3 | 100% | 19.59 | 98.01% | 0.32 | 0.124 | 0.03% | 4909.62 |
| SAC | v4 | 72% | 14.31 | 71.67% | 0.19 | 0.825 | 5.75% | -4435.36 |

## Notes

- PPO improves sharply from v3 to v4.
- SAC v3 remains the strongest policy in this run.
- SAC v4 can balance from upright but performs poorly from hanging, with high torque and saturation.
- Training `stable_fraction` alone is not enough to judge swing-up. The hanging-reset evaluation is the stricter result.

## Result Files

- `runs/combined/ppo/reward_v3_seed1/evaluation_hanging.json`
- `runs/combined/ppo/reward_v3_seed1/evaluation_upright.json`
- `runs/combined/ppo/reward_v4_seed1/evaluation_hanging.json`
- `runs/combined/ppo/reward_v4_seed1/evaluation_upright.json`
- `runs/combined/sac/reward_v3_seed1/evaluation_hanging.json`
- `runs/combined/sac/reward_v3_seed1/evaluation_upright.json`
- `runs/combined/sac/reward_v4_seed1/evaluation_hanging.json`
- `runs/combined/sac/reward_v4_seed1/evaluation_upright.json`
