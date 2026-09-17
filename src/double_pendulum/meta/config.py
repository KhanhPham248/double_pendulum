"""Versioned configuration for double-pendulum RMA-meta selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


META_RL_CONFIG_VERSION = 1
HOLD_NAME = "HOLD"


@dataclass(frozen=True)
class MetaRlModelCfg:
  observation_dim: int = 6
  factor_dim: int = 6
  latent_dim: int = 4
  history_steps: int = 50
  adaptation_hidden_dim: int = 64
  adaptation_num_layers: int = 1
  factor_hidden_dims: tuple[int, ...] = (32, 16)
  meta_hidden_dims: tuple[int, ...] = (128, 64)

  def __post_init__(self) -> None:
    scalar_values = (
      self.observation_dim,
      self.factor_dim,
      self.latent_dim,
      self.history_steps,
      self.adaptation_hidden_dim,
      self.adaptation_num_layers,
    )
    if any(value <= 0 for value in scalar_values):
      raise ValueError("all meta model dimensions must be positive")
    if not self.factor_hidden_dims or not self.meta_hidden_dims:
      raise ValueError("meta MLP hidden dimensions cannot be empty")
    if any(value <= 0 for value in (*self.factor_hidden_dims, *self.meta_hidden_dims)):
      raise ValueError("meta MLP hidden dimensions must be positive")


@dataclass(frozen=True)
class MetaRlPpoCfg:
  meta_decimation: int = 5
  rollout_steps: int = 32
  learning_epochs: int = 5
  mini_batches: int = 4
  learning_rate: float = 3.0e-4
  finetune_learning_rate: float = 1.0e-4
  clip_param: float = 0.2
  value_loss_coef: float = 1.0
  entropy_coef: float = 0.01
  gamma: float = 0.99
  gae_lambda: float = 0.95
  max_grad_norm: float = 1.0
  target_kl: float = 0.02
  switch_penalty: float = 0.01
  routing_loss_coef: float = 0.25
  max_iterations: int = 3000
  finetune_iterations: int = 1000
  save_interval: int = 100

  def __post_init__(self) -> None:
    integer_values = (
      self.meta_decimation,
      self.rollout_steps,
      self.learning_epochs,
      self.mini_batches,
      self.max_iterations,
      self.finetune_iterations,
      self.save_interval,
    )
    if any(value <= 0 for value in integer_values):
      raise ValueError("meta PPO integer settings must be positive")
    if self.rollout_steps % self.mini_batches:
      raise ValueError("rollout_steps must be divisible by mini_batches")
    if min(self.learning_rate, self.finetune_learning_rate) <= 0.0:
      raise ValueError("meta PPO learning rates must be positive")
    if self.max_grad_norm <= 0.0:
      raise ValueError("max_grad_norm must be positive")
    if not 0.0 < self.clip_param < 1.0:
      raise ValueError("clip_param must be in (0, 1)")
    if not 0.0 < self.gamma <= 1.0 or not 0.0 < self.gae_lambda <= 1.0:
      raise ValueError("gamma and gae_lambda must be in (0, 1]")
    if min(
      self.value_loss_coef,
      self.entropy_coef,
      self.target_kl,
      self.routing_loss_coef,
    ) < 0.0:
      raise ValueError("meta PPO coefficients cannot be negative")
    if self.switch_penalty < 0.0:
      raise ValueError("switch_penalty cannot be negative")


@dataclass(frozen=True)
class MetaRlRuntimeCfg:
  low_level_hz: float = 50.0
  meta_hz: float = 10.0
  confidence_threshold: float = 0.60
  crossfade_s: float = 0.20
  min_dwell_s: float = 0.20

  def __post_init__(self) -> None:
    if self.low_level_hz <= 0.0 or self.meta_hz <= 0.0:
      raise ValueError("runtime frequencies must be positive")
    ratio = self.low_level_hz / self.meta_hz
    if abs(ratio - round(ratio)) > 1.0e-6:
      raise ValueError("low_level_hz must be an integer multiple of meta_hz")
    if not 0.0 <= self.confidence_threshold <= 1.0:
      raise ValueError("confidence_threshold must be in [0, 1]")
    if self.crossfade_s < 0.0 or self.min_dwell_s < 0.0:
      raise ValueError("crossfade and dwell durations cannot be negative")

  @property
  def meta_decimation(self) -> int:
    return round(self.low_level_hz / self.meta_hz)

  @property
  def crossfade_steps(self) -> int:
    return max(1, round(self.crossfade_s * self.low_level_hz))

  @property
  def min_dwell_decisions(self) -> int:
    return max(0, round(self.min_dwell_s * self.meta_hz))


@dataclass(frozen=True)
class MetaRlCfg:
  config_version: int = META_RL_CONFIG_VERSION
  task_id: str = "combined"
  expert_registry: str = "configs/meta_rl/double_pendulum_expert_registry_v1.yaml"
  expert_names: tuple[str, ...] = ("policy_sac", "policy_ppo")
  initial_expert: str = "policy_sac"
  factor_names: tuple[str, ...] = (
    "mass_scale_delta",
    "base_damping_scale_delta",
    "elbow_damping_scale_delta",
    "torque_limit_scale_delta",
    "actuator_delay_steps_norm",
    "observation_noise_std_norm",
  )
  model: MetaRlModelCfg = field(default_factory=MetaRlModelCfg)
  ppo: MetaRlPpoCfg = field(default_factory=MetaRlPpoCfg)
  runtime: MetaRlRuntimeCfg = field(default_factory=MetaRlRuntimeCfg)

  def __post_init__(self) -> None:
    if self.config_version != META_RL_CONFIG_VERSION:
      raise ValueError(
        f"meta config version must be {META_RL_CONFIG_VERSION}, "
        f"got {self.config_version}"
      )
    if not self.expert_names or len(set(self.expert_names)) != len(self.expert_names):
      raise ValueError("expert_names must be non-empty and unique")
    if self.initial_expert not in self.expert_names:
      raise ValueError("initial_expert must be present in expert_names")
    if HOLD_NAME in self.expert_names:
      raise ValueError(f"{HOLD_NAME} is reserved and cannot be an expert name")
    if len(self.factor_names) != self.model.factor_dim:
      raise ValueError("factor_names length must equal model.factor_dim")
    if self.ppo.meta_decimation != self.runtime.meta_decimation:
      raise ValueError("PPO and runtime meta decimation must match")

  @property
  def initial_expert_index(self) -> int:
    return self.expert_names.index(self.initial_expert)

  @property
  def hold_index(self) -> int:
    return len(self.expert_names)

  @property
  def class_names(self) -> tuple[str, ...]:
    return (*self.expert_names, HOLD_NAME)

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)


def _tuples(payload: dict[str, Any], keys: tuple[str, ...]) -> None:
  for key in keys:
    if key in payload:
      payload[key] = tuple(payload[key])


def meta_rl_cfg_from_dict(payload: dict[str, Any]) -> MetaRlCfg:
  values = dict(payload)
  model = dict(values.pop("model", {}))
  ppo = dict(values.pop("ppo", {}))
  runtime = dict(values.pop("runtime", {}))
  _tuples(model, ("factor_hidden_dims", "meta_hidden_dims"))
  _tuples(values, ("expert_names", "factor_names"))
  return MetaRlCfg(
    **values,
    model=MetaRlModelCfg(**model),
    ppo=MetaRlPpoCfg(**ppo),
    runtime=MetaRlRuntimeCfg(**runtime),
  )


def load_meta_rl_cfg(path: str | Path) -> MetaRlCfg:
  payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
  if not isinstance(payload, dict):
    raise ValueError("meta RL config must contain a YAML mapping")
  return meta_rl_cfg_from_dict(payload)
