"""Soft Actor-Critic update, safeguards and checkpoint state."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable

import torch
from torch import nn

from .config import SACConfig
from .networks import Actor, Critic, RunningNormalizer
from .replay_buffer import TransitionBatch


def _require_finite(name: str, value: torch.Tensor) -> None:
  if not torch.isfinite(value).all():
    raise FloatingPointError(f"non-finite values detected in {name}")


class SACAgent:
  def __init__(self, obs_dim: int, action_dim: int, config: SACConfig, device: str):
    if min(obs_dim, action_dim) <= 0:
      raise ValueError("observation and action dimensions must be positive")
    self.obs_dim = obs_dim
    self.action_dim = action_dim
    self.config = config
    self.device = torch.device(device)
    self.normalizer = RunningNormalizer(
      obs_dim, clip=config.observation_clip
    ).to(self.device)
    self.actor = Actor(obs_dim, action_dim, config.hidden_dims).to(self.device)
    self.critics = nn.ModuleList(
      Critic(obs_dim, action_dim, config.hidden_dims).to(self.device) for _ in range(2)
    )
    self.targets = nn.ModuleList(
      Critic(obs_dim, action_dim, config.hidden_dims).to(self.device) for _ in range(2)
    )
    for target, critic in zip(self.targets, self.critics):
      target.load_state_dict(critic.state_dict())
      target.requires_grad_(False)

    self.log_alpha = (
      torch.tensor(config.init_alpha, device=self.device).log().requires_grad_()
    )
    critic_parameters = [
      parameter for model in self.critics for parameter in model.parameters()
    ]
    self.actor_optimizer = torch.optim.Adam(
      self.actor.parameters(), lr=config.learning_rate
    )
    self.critic_optimizer = torch.optim.Adam(
      critic_parameters, lr=config.learning_rate
    )
    self.alpha_optimizer = torch.optim.Adam(
      (self.log_alpha,), lr=config.learning_rate
    )
    self.schedulers = tuple(
      self._make_scheduler(optimizer)
      for optimizer in (
        self.actor_optimizer,
        self.critic_optimizer,
        self.alpha_optimizer,
      )
    )
    self.target_entropy = (
      config.target_entropy if config.target_entropy is not None else -float(action_dim)
    )
    self.update_count = 0

  def _make_scheduler(
    self, optimizer: torch.optim.Optimizer
  ) -> torch.optim.lr_scheduler.LambdaLR:
    if self.config.learning_rate_schedule == "constant":
      return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    floor = self.config.minimum_learning_rate / self.config.learning_rate

    def multiplier(step: int) -> float:
      progress = min(step / self.config.learning_rate_decay_steps, 1.0)
      return max(floor, 1.0 - progress * (1.0 - floor))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)

  @property
  def alpha(self) -> torch.Tensor:
    return self.log_alpha.exp()

  @torch.no_grad()
  def observe(self, observations: torch.Tensor) -> None:
    """Update running statistics from newly collected, raw observations."""

    observations = observations.to(self.device)
    if self.config.nan_guard:
      _require_finite("collected observations", observations)
    if self.config.normalize_observations:
      self.normalizer.update(observations)

  def normalize(self, observations: torch.Tensor) -> torch.Tensor:
    observations = observations.to(self.device)
    if not self.config.normalize_observations:
      return observations
    return self.normalizer(observations)

  @torch.no_grad()
  def act(
    self, observations: torch.Tensor, deterministic: bool = False
  ) -> torch.Tensor:
    actions = self.actor.sample(self.normalize(observations), deterministic)[0]
    if self.config.nan_guard:
      _require_finite("actions", actions)
    return actions

  @torch.no_grad()
  def compute_target(self, batch: TransitionBatch) -> torch.Tensor:
    """Compute the clipped-double-Q Bellman target used by the critics."""

    next_observations = self.normalize(batch.next_observations)
    next_actions, next_log_prob = self.actor.sample(next_observations)
    assert next_log_prob is not None
    target_q = torch.minimum(
      *(target(next_observations, next_actions) for target in self.targets)
    )
    target_q -= self.alpha.detach() * next_log_prob
    rewards = batch.rewards * self.config.reward_scale
    return rewards + self.config.gamma * (1.0 - batch.dones) * target_q

  def update(self, batch: TransitionBatch) -> dict[str, float]:
    if self.config.nan_guard:
      for name, value in zip(batch._fields, batch):
        _require_finite(name, value)

    observations = self.normalize(batch.observations)
    target = self.compute_target(batch)
    if self.config.nan_guard:
      _require_finite("critic target", target)

    q_values = [critic(observations, batch.actions) for critic in self.critics]
    critic_loss = sum(torch.nn.functional.mse_loss(q, target) for q in q_values)
    critic_grad_norm = self._optimize(
      critic_loss,
      self.critic_optimizer,
      self.critics.parameters(),
      "critic",
    )

    self.critics.requires_grad_(False)
    try:
      actions, log_prob = self.actor.sample(observations)
      assert log_prob is not None
      q = torch.minimum(*(critic(observations, actions) for critic in self.critics))
      actor_loss = (self.alpha.detach() * log_prob - q).mean()
      actor_grad_norm = self._optimize(
        actor_loss,
        self.actor_optimizer,
        self.actor.parameters(),
        "actor",
      )
    finally:
      self.critics.requires_grad_(True)

    alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
    alpha_grad_norm = self._optimize(
      alpha_loss,
      self.alpha_optimizer,
      (self.log_alpha,),
      "alpha",
    )
    self._update_targets()
    for scheduler in self.schedulers:
      scheduler.step()
    self.update_count += 1

    metrics = {
      "actor_loss": actor_loss.item(),
      "critic_loss": critic_loss.item(),
      "alpha_loss": alpha_loss.item(),
      "alpha": self.alpha.item(),
      "entropy": -log_prob.mean().item(),
      "q_mean": torch.minimum(*q_values).mean().item(),
      "actor_grad_norm": actor_grad_norm,
      "critic_grad_norm": critic_grad_norm,
      "alpha_grad_norm": alpha_grad_norm,
      "learning_rate": self.actor_optimizer.param_groups[0]["lr"],
    }
    metrics_are_finite = all(
      torch.isfinite(torch.tensor(value)) for value in metrics.values()
    )
    if self.config.nan_guard and not metrics_are_finite:
      raise FloatingPointError("non-finite SAC metrics")
    return metrics

  def _optimize(
    self,
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    parameters: Iterable[torch.Tensor],
    name: str,
  ) -> float:
    if self.config.nan_guard:
      _require_finite(f"{name} loss", loss)
    parameter_list = list(parameters)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
      parameter_list,
      self.config.max_grad_norm,
      error_if_nonfinite=self.config.nan_guard,
    )
    optimizer.step()
    return float(grad_norm.detach())

  @torch.no_grad()
  def _update_targets(self) -> None:
    for target, critic in zip(self.targets, self.critics):
      for target_parameter, parameter in zip(target.parameters(), critic.parameters()):
        target_parameter.lerp_(parameter, self.config.tau)

  def state_dict(self) -> dict[str, Any]:
    return {
      "format_version": 2,
      "obs_dim": self.obs_dim,
      "action_dim": self.action_dim,
      "config": asdict(self.config),
      "normalizer": self.normalizer.state_dict(),
      "actor": self.actor.state_dict(),
      "critics": [model.state_dict() for model in self.critics],
      "targets": [model.state_dict() for model in self.targets],
      "log_alpha": self.log_alpha.detach(),
      "optimizers": [
        self.actor_optimizer.state_dict(),
        self.critic_optimizer.state_dict(),
        self.alpha_optimizer.state_dict(),
      ],
      "schedulers": [scheduler.state_dict() for scheduler in self.schedulers],
      "update_count": self.update_count,
    }

  def load_state_dict(self, state: dict[str, Any]) -> None:
    if int(state.get("format_version", 0)) != 2:
      raise ValueError("unsupported SAC agent checkpoint format")
    if (state["obs_dim"], state["action_dim"]) != (self.obs_dim, self.action_dim):
      raise ValueError("checkpoint dimensions do not match")
    self.normalizer.load_state_dict(state["normalizer"])
    self.actor.load_state_dict(state["actor"])
    for models, weights in (
      (self.critics, state["critics"]),
      (self.targets, state["targets"]),
    ):
      if len(models) != len(weights):
        raise ValueError("checkpoint critic count does not match")
      for model, model_weights in zip(models, weights):
        model.load_state_dict(model_weights)
    self.log_alpha.data.copy_(state["log_alpha"].to(self.device))
    for optimizer, weights in zip(
      (self.actor_optimizer, self.critic_optimizer, self.alpha_optimizer),
      state["optimizers"],
    ):
      optimizer.load_state_dict(weights)
    for scheduler, weights in zip(self.schedulers, state["schedulers"]):
      scheduler.load_state_dict(weights)
    self.update_count = int(state["update_count"])
