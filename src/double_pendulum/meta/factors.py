"""Train-only physical factors for the double-pendulum RMA teacher."""

from __future__ import annotations

import torch


FACTOR_NAMES: tuple[str, ...] = (
  "mass_scale_delta",
  "base_damping_scale_delta",
  "elbow_damping_scale_delta",
  "torque_limit_scale_delta",
  "actuator_delay_steps_norm",
  "observation_noise_std_norm",
)


def build_privileged_factors(
  mass_scale: torch.Tensor,
  base_damping_scale: torch.Tensor,
  elbow_damping_scale: torch.Tensor,
  torque_limit_scale: torch.Tensor,
  actuator_delay_steps: torch.Tensor,
  observation_noise_std: torch.Tensor,
) -> torch.Tensor:
  """Build the normalized factor vector used only by the teacher encoder.

  The first four values are deltas from nominal scale one. Delay is normalized
  by four control steps and observation noise by a 0.1 rad/s reference. The
  resulting vector is never part of the deployable actor input.
  """
  values = (
    mass_scale,
    base_damping_scale,
    elbow_damping_scale,
    torque_limit_scale,
    actuator_delay_steps,
    observation_noise_std,
  )
  shapes = {name: value.shape for name, value in zip(FACTOR_NAMES, values, strict=True)}
  if any(value.ndim != 1 for value in values):
    raise ValueError(f"privileged factors must be rank-1 tensors: {shapes}")
  batch = values[0].shape[0]
  if any(value.shape != (batch,) for value in values):
    raise ValueError(f"privileged factor batch shapes differ: {shapes}")
  factors = torch.stack(
    (
      mass_scale - 1.0,
      base_damping_scale - 1.0,
      elbow_damping_scale - 1.0,
      torque_limit_scale - 1.0,
      actuator_delay_steps / 4.0,
      observation_noise_std / 0.1,
    ),
    dim=-1,
  )
  factors = torch.clamp(factors, -2.0, 2.0)
  if not torch.isfinite(factors).all():
    raise ValueError("privileged factors contain non-finite values")
  return factors
