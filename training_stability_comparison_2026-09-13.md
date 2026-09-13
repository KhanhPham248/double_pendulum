# PPO learning-rate diagnosis and SAC v3/v4 comparison

Date: 2026-09-13. All policy comparisons below use deterministic ONNX inference,
the same MuJoCo model, the `hanging` reset, 20 s episodes, and seeds starting at
10001.

## PPO adaptive learning-rate failure

The reward-v4 adaptive run used `learning_rate=1e-3`, `schedule=adaptive`,
`desired_kl=0.01`, 5 learning epochs, and 4 mini-batches. RSL-RL updates the
learning rate after every mini-batch from actor KL:

- KL below 0.005 multiplies the learning rate by 1.5;
- KL above 0.02 divides the learning rate by 1.5;
- the allowed range is 1e-5 to 1e-2.

This test happens 20 times per PPO iteration. The same optimizer and learning
rate update both actor and critic, although the schedule only measures actor KL.
Once the actor is close to a stable solution, its KL becomes small and the
scheduler raises the critic learning rate with it. The logged learning rate
ranged from 1e-5 to 1e-2 and changed from 4.38e-3 to 1e-5 over adjacent logged
iterations. The value loss then rose from 1.62e3 at iteration 1128 to 9.48e5 at
iteration 1155, followed by the policy collapse.

The fixed run used `learning_rate=3e-4` for all 2000 iterations. After iteration
1130, its maximum value loss was 1.78e3; the adaptive run reached 1.47e6. The
fixed run's logged mean reward remained above 2321 after that point, while the
adaptive run fell to -62493. The fixed final checkpoint still passed all 10
periodic hanging evaluations; the adaptive final checkpoint passed 66% of 100
episodes.

### Best PPO reward-v4 policies, 100 episodes

| Metric | Adaptive, iteration 1050 | Fixed 3e-4, iteration 1900 |
|---|---:|---:|
| Success rate | 100% | 100% |
| Swing-up time | 1.1556 s | 1.9384 s |
| Longest hold | 18.8390 s | 18.0608 s |
| Upright fraction | 94.203% | 90.305% |
| Mean absolute torque | 0.2723 Nm | 0.4653 Nm |
| Action saturation | 1.158% | 6.166% |
| Episode return | 4707.09 | 4519.65 |

The fixed run first reached 100% in its periodic 10-episode evaluation at
iteration 1550. It prevents late collapse, but the strongest adaptive checkpoint
is still the better controller. Keeping the automatically selected `best/`
policy is therefore necessary even with a fixed learning rate.

## SAC reward-v3 versus reward-v4

Every stored checkpoint was evaluated over 10 hanging episodes at roughly 100k
transition intervals. Both versions first reached 100% success at transition
200192. At transition 100352 neither passed, but v3 already held upright for
0.766 s while v4 held for 0 s.

The live training metric gives finer 12.8k-transition resolution:

| Convergence marker | SAC v3 | SAC v4 |
|---|---:|---:|
| First stable fraction >= 50% | 153600 | 166400 |
| First stable fraction >= 90% | 204800 | 192000 |
| Stable fraction >= 95% for five logs | 204800 | 294400 |
| First deterministic checkpoint with 100% success | 200192 | 200192 |

At checkpoint resolution the convergence time is tied. Using sustained 95%
training stability, v3 converged 89600 transitions earlier and was more
consistent during the initial learning phase.

Candidate checkpoints were screened over 50 episodes, then the selected best
checkpoint from each reward version was evaluated over 100 episodes:

| Metric | SAC v3, transition 1500160 | SAC v4, transition 1100288 |
|---|---:|---:|
| Success rate | 100% | 100% |
| Swing-up time | 1.2540 s | 1.3896 s |
| Longest hold | 18.7456 s | 18.6082 s |
| Upright fraction | 93.729% | 93.049% |
| Mean absolute torque | 0.5331 Nm | 0.4795 Nm |
| Mean angular velocity | 0.5435 rad/s | 0.4058 rad/s |
| Action saturation | 0.001% | 0.252% |

The v3 policy swings up 0.1356 s faster and holds 0.1374 s longer. The v4
policy uses 10.1% less mean torque and has 25.3% lower mean angular velocity,
which matches the new global velocity penalty. Episode return is omitted from
the cross-version conclusion because v3 and v4 use different reward formulas.

SAC v4 was less stable late in training: its final checkpoint passed 3% of 100
episodes, while the selected v4 checkpoint passed 100%. SAC v3's final
checkpoint retained 100% success.

## Saved artifacts

- PPO adaptive best: `runs/combined/ppo/reward_v4_seed1/best_adaptive_hanging_100/`
- PPO fixed run and best policy: `runs/combined/ppo/reward_v4_fixed_lr_seed1/`
- SAC v3 best: `runs/combined/sac/reward_v3_seed1/best_hanging_100/`
- SAC v4 best: `runs/combined/sac/reward_v4_seed1/best_hanging_100/`
- SAC checkpoint curves: `checkpoint_evaluations_hanging_10.json` and
  `checkpoint_candidates_hanging_50.json` inside each SAC run.

The PPO W&B data was recorded in offline mode because this machine has no W&B
API key. After `wandb login`, it can be uploaded with:

```bash
wandb sync runs/combined/ppo/reward_v4_fixed_lr_seed1/wandb/offline-run-*
```
