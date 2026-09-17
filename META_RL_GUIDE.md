# Double-pendulum RMA-meta

The existing `combined` task remains the legacy single-policy route. This
experiment adds an isolated high-level selector over two already-trained
full-task policies:

- `policy_sac`: SAC reward-v3, evaluated from hanging and upright states;
- `policy_ppo`: PPO reward-v4 fixed-learning-rate checkpoint, evaluated from
  hanging and upright states.

Both policies keep the same `6D observation -> 1D normalized base torque`
contract. They are not swing-up and balance specialists: each policy performs
the complete swing-up, capture and balance behavior.

## Runtime structure

```text
observation(6D)
  -> history(50 x 6) -> GRU adapter -> z_hat(4D)
  -> meta selector([observation, z_hat, previous expert])
  -> {policy_sac, policy_ppo, HOLD} at 10 Hz
  -> selected expert at 50 Hz
  -> normalized action * 6 Nm -> base_joint
```

The latent represents train-only dynamics uncertainty. It must not represent
the swing-up/balance phase, because both experts solve both phases. The initial
factor vector contains mass, damping, torque, delay and observation-noise
scales. Only factors that are actually randomized in the simulator may be
used for a valid adaptation experiment.

## Selected expert bundles

The selection is pinned in
`configs/meta_rl/double_pendulum_expert_registry_v1.yaml`. The loader checks
the policy manifest hash, ONNX shape, and equality of both policy contracts.

Run the contract preflight from the repository root:

```bash
PYTHONPATH=src python scripts/check_meta_rl.py
```

Expected result:

```text
META_PREFLIGHT=PASS
```

## Training order

1. Validate each frozen expert independently with the same MuJoCo model and
   evaluation protocol.
2. Add and log per-environment physical randomization. Do not train the
   adaptation module against constant nominal factors.
3. Train privileged teacher PPO. The teacher receives the factor encoder
   output and chooses between the two full-task experts plus `HOLD`.
4. Generate `(history_50x6, teacher_latent)` pairs and train the GRU adapter
   with MSE/cosine validation split by episode, not by individual frame.
5. Optionally fine-tune only the meta actor/critic with estimated `z_hat`.
   Keep the two experts and the adapter frozen during this stage.
6. Export selector and adapter only after checkpoint/config/hash parity passes.

The meta target should come from short counterfactual rollouts of both experts
from the same state. A state-based `q1` threshold is useful as a diagnostic,
but is not sufficient evidence that the selector chose the better controller.

## Acceptance gates

The implementation must report these separately:

- expert contract and ONNX hash parity;
- adapter latent MSE/cosine, which is only an adaptation metric;
- selected versus proposed expert fractions;
- policy switch count, rejected switches due to dwell, and `HOLD` fraction;
- hanging success rate, time-to-success, stable hold, escapes, torque and
  saturation under each randomized dynamics case.

Expert switching is crossfaded in physical torque. `HOLD` preserves the current
expert when confidence is low; it is not the same as a numerical-failure
safety latch.
