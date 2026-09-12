"""Device-resident replay buffer with portable, compact checkpoints."""

from __future__ import annotations

from typing import Any, NamedTuple

import torch


class TransitionBatch(NamedTuple):
  observations: torch.Tensor
  actions: torch.Tensor
  rewards: torch.Tensor
  next_observations: torch.Tensor
  dones: torch.Tensor


class ReplayBuffer:
  _NAMES = ("observations", "actions", "rewards", "next_observations", "dones")

  def __init__(self, capacity: int, obs_dim: int, action_dim: int, device: str):
    if min(capacity, obs_dim, action_dim) <= 0:
      raise ValueError("buffer dimensions must be positive")
    self.capacity = capacity
    self.obs_dim = obs_dim
    self.action_dim = action_dim
    self.device = torch.device(device)
    self.size = 0
    self.position = 0
    self.observations = torch.empty((capacity, obs_dim), device=self.device)
    self.actions = torch.empty((capacity, action_dim), device=self.device)
    self.rewards = torch.empty((capacity, 1), device=self.device)
    self.next_observations = torch.empty((capacity, obs_dim), device=self.device)
    self.dones = torch.empty((capacity, 1), device=self.device)

  def __len__(self) -> int:
    return self.size

  def add(
    self,
    observations: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_observations: torch.Tensor,
    dones: torch.Tensor,
  ) -> None:
    count = observations.shape[0]
    if count == 0:
      return
    if count > self.capacity:
      observations = observations[-self.capacity :]
      actions = actions[-self.capacity :]
      rewards = rewards[-self.capacity :]
      next_observations = next_observations[-self.capacity :]
      dones = dones[-self.capacity :]
      count = self.capacity
    if observations.shape != (count, self.obs_dim):
      raise ValueError("invalid observation shape")
    if actions.shape != (count, self.action_dim):
      raise ValueError("invalid action shape")
    if next_observations.shape != (count, self.obs_dim):
      raise ValueError("invalid next_observation shape")
    if rewards.numel() != count or dones.numel() != count:
      raise ValueError("reward and done must contain one value per transition")

    indices = (torch.arange(count, device=self.device) + self.position) % self.capacity
    self.observations[indices] = observations.detach()
    self.actions[indices] = actions.detach()
    self.rewards[indices] = rewards.detach().reshape(count, 1)
    self.next_observations[indices] = next_observations.detach()
    self.dones[indices] = dones.detach().reshape(count, 1).float()
    self.position = (self.position + count) % self.capacity
    self.size = min(self.capacity, self.size + count)

  def sample(self, batch_size: int) -> TransitionBatch:
    if not 0 < batch_size <= self.size:
      raise ValueError(f"cannot sample {batch_size} transitions from {self.size}")
    indices = torch.randint(self.size, (batch_size,), device=self.device)
    return TransitionBatch(*(tensor[indices] for tensor in self._tensors()))

  def state_dict(self) -> dict[str, Any]:
    state: dict[str, Any] = {
      "capacity": self.capacity,
      "obs_dim": self.obs_dim,
      "action_dim": self.action_dim,
      "size": self.size,
      "position": self.position,
    }
    for name, tensor in zip(self._NAMES, self._tensors()):
      state[name] = tensor[: self.size].detach().cpu()
    return state

  def load_state_dict(self, state: dict[str, Any]) -> None:
    expected = (self.capacity, self.obs_dim, self.action_dim)
    actual = (int(state["capacity"]), int(state["obs_dim"]), int(state["action_dim"]))
    if actual != expected:
      raise ValueError(
        f"replay dimensions differ: checkpoint={actual}, runtime={expected}"
      )
    size = int(state["size"])
    if not 0 <= size <= self.capacity:
      raise ValueError("invalid replay size")
    for name, target in zip(self._NAMES, self._tensors()):
      source = state[name].to(self.device)
      if source.shape != target[:size].shape:
        raise ValueError(f"invalid replay shape for {name}")
      target[:size].copy_(source)
    self.size = size
    self.position = int(state.get("position", size % self.capacity)) % self.capacity

  def _tensors(self) -> tuple[torch.Tensor, ...]:
    return tuple(getattr(self, name) for name in self._NAMES)
