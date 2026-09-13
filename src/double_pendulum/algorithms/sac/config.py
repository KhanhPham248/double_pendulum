"""Hyperparameters for the standalone SAC implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class SACConfig:
  gamma: float = 0.995
  tau: float = 0.005
  learning_rate: float = 3e-4
  minimum_learning_rate: float = 3e-5
  learning_rate_schedule: Literal["constant", "linear"] = "constant"
  learning_rate_decay_steps: int = 100_000
  hidden_dims: tuple[int, ...] = (256, 256)
  init_alpha: float = 0.2
  target_entropy: float | None = None
  reward_scale: float = 1.0
  max_grad_norm: float = 10.0
  normalize_observations: bool = True
  observation_clip: float = 10.0
  nan_guard: bool = True

  def __post_init__(self) -> None:
    if not 0.0 <= self.gamma <= 1.0:
      raise ValueError("gamma must be in [0, 1]")
    if not 0.0 < self.tau <= 1.0:
      raise ValueError("tau must be in (0, 1]")
    if self.learning_rate <= 0.0 or self.minimum_learning_rate < 0.0:
      raise ValueError("learning rates must be non-negative and initial lr positive")
    if self.minimum_learning_rate > self.learning_rate:
      raise ValueError("minimum learning rate cannot exceed initial learning rate")
    if self.learning_rate_decay_steps <= 0:
      raise ValueError("learning_rate_decay_steps must be positive")
    if not self.hidden_dims or any(width <= 0 for width in self.hidden_dims):
      raise ValueError("hidden_dims must contain positive widths")
    if min(self.init_alpha, self.reward_scale, self.max_grad_norm) <= 0.0:
      raise ValueError("alpha, reward scale and max grad norm must be positive")
    if self.observation_clip <= 0.0:
      raise ValueError("observation_clip must be positive")


@dataclass(frozen=True)
class SACTrainConfig:
  task: str = "combined"
  device: str = "cuda:0"
  num_envs: int = 512
  total_transitions: int = 2_000_000
  replay_capacity: int = 1_000_000
  batch_size: int = 256
  learning_starts: int = 25_000
  utd_ratio: float = 0.25
  checkpoint_interval: int = 200_000
  log_interval: int = 25
  seed: int = 1
  run_dir: str | None = None
  resume: str | None = None
  export_fail_fast: bool = False
  evaluation_episodes: int = 10
  use_wandb: bool = False
  wandb_project: str = "double-pendulum"
  wandb_entity: str | None = None
  wandb_run_name: str | None = None
  agent: SACConfig = field(default_factory=SACConfig)

  def __post_init__(self) -> None:
    if self.num_envs <= 0 or self.total_transitions <= 0:
      raise ValueError("num_envs and total_transitions must be positive")
    if self.replay_capacity < self.batch_size:
      raise ValueError("replay_capacity must be at least batch_size")
    if self.learning_starts < self.batch_size:
      raise ValueError("learning_starts must be at least batch_size")
    if self.utd_ratio <= 0.0:
      raise ValueError("utd_ratio must be positive")
    if min(self.checkpoint_interval, self.log_interval) <= 0:
      raise ValueError("checkpoint and log intervals must be positive")
    if self.evaluation_episodes < 0:
      raise ValueError("evaluation episodes cannot be negative")
    if self.use_wandb and not self.wandb_project:
      raise ValueError("W&B project cannot be empty")
    if self.run_dir is not None and self.resume is not None:
      raise ValueError("run_dir and resume cannot be used together")
