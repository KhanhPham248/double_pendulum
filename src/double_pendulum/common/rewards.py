"""Backend-agnostic reward math shared by training and evaluation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from .observations import wrapped_angle_error


@dataclass(frozen=True)
class CombinedRewardSpec:
  """Reward parameters for the single-policy swing-up and balancing task."""

  formula_version: int = 2
  link_alignment_weight: float = 1.0
  upright_capture_weight: float = 1.0
  balancing_bonus_weight: float = 2.0
  upright_velocity_weight: float = -0.1
  torque_weight: float = -0.01
  action_rate_weight: float = -0.02
  capture_angle_sigma_rad: float = 0.5

  def __post_init__(self) -> None:
    if self.formula_version not in (1, 2):
      raise ValueError("unsupported combined reward formula")
    rewards = (
      self.link_alignment_weight,
      self.upright_capture_weight,
      self.balancing_bonus_weight,
    )
    penalties = (
      self.upright_velocity_weight,
      self.torque_weight,
      self.action_rate_weight,
    )
    if any(weight < 0.0 for weight in rewards):
      raise ValueError("reward weights must be non-negative")
    if any(weight > 0.0 for weight in penalties):
      raise ValueError("penalty weights must be non-positive")
    if self.capture_angle_sigma_rad <= 0.0:
      raise ValueError("capture angle sigma must be positive")

  def as_dict(self) -> dict[str, float | int]:
    return asdict(self)


DEFAULT_COMBINED_REWARD = CombinedRewardSpec()
LEGACY_COMBINED_REWARD = CombinedRewardSpec(
  formula_version=1,
  upright_capture_weight=0.0,
  balancing_bonus_weight=0.5,
  upright_velocity_weight=-0.03,
  torque_weight=-0.002,
  action_rate_weight=0.0,
  capture_angle_sigma_rad=0.35,
)


def absolute_link_alignment(qpos: Any, array_api: Any) -> Any:
  """Mean upright alignment of both links in world coordinates."""

  link1_angle = qpos[..., 0]
  link2_angle = qpos[..., 0] + qpos[..., 1]
  return 0.5 * (
    array_api.cos(link1_angle - math.pi)
    + array_api.cos(link2_angle - math.pi)
  )


def upright_proximity(qpos: Any, array_api: Any, *, sigma_rad: float) -> Any:
  """Smooth capture reward around the configuration where both links are upright."""

  link1_error = wrapped_angle_error(qpos[..., 0], math.pi, array_api)
  link2_error = wrapped_angle_error(
    qpos[..., 0] + qpos[..., 1], math.pi, array_api
  )
  error_sq = array_api.square(link1_error) + array_api.square(link2_error)
  return array_api.exp(-error_sq / sigma_rad**2)


def upright_velocity_l2(
  qpos: Any,
  qvel: Any,
  array_api: Any,
  *,
  sigma_rad: float,
) -> Any:
  """Penalize absolute link velocities primarily inside the capture region."""

  proximity = upright_proximity(qpos, array_api, sigma_rad=sigma_rad)
  link1_velocity = qvel[..., 0]
  link2_velocity = qvel[..., 0] + qvel[..., 1]
  velocity_l2 = 0.5 * (
    array_api.square(link1_velocity) + array_api.square(link2_velocity)
  )
  return proximity * velocity_l2


def action_l2(action: Any, array_api: Any) -> Any:
  return array_api.sum(array_api.square(action), -1)


def action_rate_l2(action: Any, previous_action: Any, array_api: Any) -> Any:
  return array_api.sum(array_api.square(action - previous_action), -1)
