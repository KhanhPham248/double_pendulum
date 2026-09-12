"""Single source of truth for training and deployment dimensions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PolicyContract:
  observation_names: tuple[str, ...] = (
    "sin_q1",
    "cos_q1",
    "sin_q2",
    "cos_q2",
    "qdot1",
    "qdot2",
  )
  action_names: tuple[str, ...] = ("base_torque",)
  joint_names: tuple[str, ...] = ("base_joint", "elbow_joint")
  actuated_joint: str = "base_joint"
  torque_limit_nm: float = 6.0
  physics_timestep_s: float = 0.002
  control_timestep_s: float = 0.02

  def __post_init__(self) -> None:
    if len(self.observation_names) != 6:
      raise ValueError("double-pendulum observation must have six values")
    if len(self.action_names) != 1:
      raise ValueError("base-actuated pendulum must have one action")
    if self.actuated_joint not in self.joint_names:
      raise ValueError("actuated_joint must be present in joint_names")
    if min(
      self.torque_limit_nm,
      self.physics_timestep_s,
      self.control_timestep_s,
    ) <= 0.0:
      raise ValueError("torque limit and timesteps must be positive")
    ratio = self.control_timestep_s / self.physics_timestep_s
    if abs(ratio - round(ratio)) > 1e-9:
      raise ValueError("control timestep must be an integer multiple of physics")

  @property
  def observation_dim(self) -> int:
    return len(self.observation_names)

  @property
  def action_dim(self) -> int:
    return len(self.action_names)

  @property
  def decimation(self) -> int:
    return round(self.control_timestep_s / self.physics_timestep_s)

  @property
  def control_frequency_hz(self) -> float:
    return 1.0 / self.control_timestep_s

  def as_dict(self) -> dict[str, Any]:
    return asdict(self)

  @classmethod
  def from_dict(cls, values: dict[str, Any]) -> "PolicyContract":
    converted = dict(values)
    for key in ("observation_names", "action_names", "joint_names"):
      converted[key] = tuple(converted[key])
    return cls(**converted)


DEFAULT_CONTRACT = PolicyContract()


@dataclass(frozen=True)
class EvaluationSpec:
  angle_threshold_rad: float = 0.21
  velocity_threshold_rad_s: float = 1.0
  success_hold_s: float = 5.0
  default_duration_s: float = 20.0

  def __post_init__(self) -> None:
    if min(
      self.angle_threshold_rad,
      self.velocity_threshold_rad_s,
      self.success_hold_s,
      self.default_duration_s,
    ) <= 0.0:
      raise ValueError("evaluation thresholds and durations must be positive")
    if self.success_hold_s >= self.default_duration_s:
      raise ValueError("success hold must be shorter than evaluation duration")

  def as_dict(self) -> dict[str, float]:
    return asdict(self)


DEFAULT_EVALUATION = EvaluationSpec()
