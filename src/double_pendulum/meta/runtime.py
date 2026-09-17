"""Fail-closed selection and torque handover state for meta inference."""

from __future__ import annotations

import torch


class ExpertSelectionState:
  def __init__(
    self,
    num_envs: int,
    num_experts: int,
    initial_expert_index: int,
    *,
    min_dwell_decisions: int = 0,
    device: torch.device | str,
  ) -> None:
    if min(num_envs, num_experts) <= 0:
      raise ValueError("selection dimensions must be positive")
    if not 0 <= initial_expert_index < num_experts:
      raise ValueError("initial expert index is outside the registry")
    if min_dwell_decisions < 0:
      raise ValueError("min_dwell_decisions cannot be negative")
    self.num_envs = num_envs
    self.num_experts = num_experts
    self.hold_index = num_experts
    self.initial_expert_index = initial_expert_index
    self.min_dwell_decisions = min_dwell_decisions
    self.device = torch.device(device)
    self.selected = torch.full(
      (num_envs,), initial_expert_index, device=self.device, dtype=torch.long
    )
    self.dwell_decisions = torch.full(
      (num_envs,), min_dwell_decisions, device=self.device, dtype=torch.long
    )

  @property
  def one_hot(self) -> torch.Tensor:
    return torch.nn.functional.one_hot(
      self.selected, num_classes=self.num_experts
    ).float()

  def resolve(self, proposed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if proposed.shape != (self.num_envs,):
      raise ValueError("meta action must have shape [num_envs]")
    proposed = proposed.to(device=self.device, dtype=torch.long)
    if torch.any(proposed < 0) or torch.any(proposed > self.hold_index):
      raise ValueError("meta action contains an invalid class index")
    previous = self.selected.clone()
    can_switch = self.dwell_decisions >= self.min_dwell_decisions
    choose = (proposed < self.num_experts) & can_switch
    self.selected = torch.where(choose, proposed, self.selected)
    changed = self.selected != previous
    self.dwell_decisions = torch.where(
      changed,
      torch.zeros_like(self.dwell_decisions),
      self.dwell_decisions + 1,
    )
    return self.selected.clone(), changed

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    if env_ids is None:
      self.selected.fill_(self.initial_expert_index)
      self.dwell_decisions.fill_(self.min_dwell_decisions)
      return
    ids = env_ids.to(self.device, dtype=torch.long)
    self.selected[ids] = self.initial_expert_index
    self.dwell_decisions[ids] = self.min_dwell_decisions


class TorqueCrossfader:
  """Blend physical torque during an expert handover."""

  def __init__(
    self,
    num_envs: int,
    crossfade_steps: int,
    *,
    device: torch.device | str,
  ) -> None:
    if num_envs <= 0 or crossfade_steps <= 0:
      raise ValueError("crossfader dimensions must be positive")
    self.num_envs = num_envs
    self.crossfade_steps = crossfade_steps
    self.device = torch.device(device)
    self.last_torque = torch.zeros(num_envs, device=self.device)
    self.blend_from = torch.zeros_like(self.last_torque)
    self.blend_step = torch.full(
      (num_envs,), crossfade_steps, device=self.device, dtype=torch.long
    )

  def begin_switch(self, changed: torch.Tensor) -> None:
    if changed.shape != (self.num_envs,):
      raise ValueError("changed mask must have shape [num_envs]")
    changed = changed.to(self.device, dtype=torch.bool)
    self.blend_from[changed] = self.last_torque[changed]
    self.blend_step[changed] = 0

  def apply(self, target_torque: torch.Tensor) -> torch.Tensor:
    if target_torque.shape != (self.num_envs,):
      raise ValueError("target torque must have shape [num_envs]")
    if not torch.isfinite(target_torque).all():
      raise ValueError("target torque contains non-finite values")
    active = self.blend_step < self.crossfade_steps
    self.blend_step[active] += 1
    alpha = torch.clamp(
      self.blend_step.float() / self.crossfade_steps, 0.0, 1.0
    )
    output = torch.where(
      active,
      (1.0 - alpha) * self.blend_from + alpha * target_torque,
      target_torque,
    )
    self.last_torque.copy_(output)
    return output

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    if env_ids is None:
      self.last_torque.zero_()
      self.blend_from.zero_()
      self.blend_step.fill_(self.crossfade_steps)
      return
    ids = env_ids.to(self.device, dtype=torch.long)
    self.last_torque[ids] = 0.0
    self.blend_from[ids] = 0.0
    self.blend_step[ids] = self.crossfade_steps


def confidence_gated_action(
  probabilities: torch.Tensor,
  *,
  hold_index: int,
  confidence_threshold: float,
) -> torch.Tensor:
  if probabilities.ndim != 2:
    raise ValueError("meta probabilities must have shape [batch, classes]")
  if not torch.isfinite(probabilities).all():
    raise ValueError("meta probabilities contain non-finite values")
  if not 0.0 <= confidence_threshold <= 1.0:
    raise ValueError("confidence threshold must be in [0, 1]")
  confidence, action = torch.max(probabilities, dim=-1)
  hold = torch.full_like(action, hold_index)
  return torch.where(confidence >= confidence_threshold, action, hold)
