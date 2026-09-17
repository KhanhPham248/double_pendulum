"""Reset-safe fixed observation history for RMA adaptation."""

from __future__ import annotations

import torch


class ObservationHistory:
  def __init__(
    self,
    num_envs: int,
    history_steps: int,
    observation_dim: int,
    *,
    device: torch.device | str,
  ) -> None:
    if min(num_envs, history_steps, observation_dim) <= 0:
      raise ValueError("history dimensions must be positive")
    self.num_envs = num_envs
    self.history_steps = history_steps
    self.observation_dim = observation_dim
    self.device = torch.device(device)
    self.buffer = torch.zeros(
      num_envs, history_steps, observation_dim, device=self.device
    )
    self.valid_steps = torch.zeros(num_envs, device=self.device, dtype=torch.long)

  @property
  def ready(self) -> torch.Tensor:
    return self.valid_steps >= self.history_steps

  def append(self, observation: torch.Tensor) -> None:
    expected = (self.num_envs, self.observation_dim)
    if observation.shape != expected:
      raise ValueError(f"history observation must have shape {expected}")
    if not torch.isfinite(observation).all():
      raise ValueError("history observation contains non-finite values")
    self.buffer[:, :-1].copy_(self.buffer[:, 1:].clone())
    self.buffer[:, -1].copy_(observation)
    self.valid_steps.add_(1).clamp_(max=self.history_steps)

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    if env_ids is None:
      self.buffer.zero_()
      self.valid_steps.zero_()
      return
    ids = env_ids.to(self.device, dtype=torch.long)
    self.buffer[ids] = 0.0
    self.valid_steps[ids] = 0
