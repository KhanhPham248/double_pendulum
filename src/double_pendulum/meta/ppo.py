"""Categorical PPO for the high-level full-task expert selector."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .config import MetaRlCfg
from .models import EnvironmentFactorEncoder, MetaActorCritic


class MetaRolloutBuffer:
  """Fixed-shape storage shared by privileged teacher and deploy PPO."""

  def __init__(
    self,
    cfg: MetaRlCfg,
    num_envs: int,
    *,
    device: torch.device | str,
  ) -> None:
    if num_envs <= 0:
      raise ValueError("num_envs must be positive")
    self.cfg = cfg
    self.num_envs = num_envs
    self.device = torch.device(device)
    shape = (cfg.ppo.rollout_steps, num_envs)
    self.observations = torch.zeros(
      *shape, cfg.model.observation_dim, device=self.device
    )
    self.factors = torch.zeros(*shape, cfg.model.factor_dim, device=self.device)
    self.latents = torch.zeros(*shape, cfg.model.latent_dim, device=self.device)
    self.previous_experts = torch.zeros(
      *shape, len(cfg.expert_names), device=self.device
    )
    self.actions = torch.zeros(*shape, device=self.device, dtype=torch.long)
    self.log_probs = torch.zeros(*shape, device=self.device)
    self.values = torch.zeros(*shape, device=self.device)
    self.rewards = torch.zeros(*shape, device=self.device)
    self.dones = torch.zeros(*shape, device=self.device, dtype=torch.bool)
    self.route_targets = torch.full(
      shape, -1, device=self.device, dtype=torch.long
    )
    self.advantages = torch.zeros(*shape, device=self.device)
    self.returns = torch.zeros(*shape, device=self.device)
    self.step = 0

  def add(
    self,
    *,
    observation: torch.Tensor,
    previous_expert: torch.Tensor,
    action: torch.Tensor,
    log_prob: torch.Tensor,
    value: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
    factor: torch.Tensor | None = None,
    latent: torch.Tensor | None = None,
    route_target: torch.Tensor | None = None,
  ) -> None:
    if self.step >= self.cfg.ppo.rollout_steps:
      raise RuntimeError("meta rollout buffer is already full")
    if (factor is None) == (latent is None):
      raise ValueError("exactly one of factor or latent must be supplied")
    index = self.step
    self.observations[index].copy_(observation)
    self.previous_experts[index].copy_(previous_expert)
    self.actions[index].copy_(action)
    self.log_probs[index].copy_(log_prob)
    self.values[index].copy_(value)
    self.rewards[index].copy_(reward)
    self.dones[index].copy_(done)
    if route_target is not None:
      if route_target.shape != (self.num_envs,):
        raise ValueError("route target must have shape [num_envs]")
      self.route_targets[index].copy_(route_target)
    if factor is not None:
      self.factors[index].copy_(factor)
    if latent is not None:
      self.latents[index].copy_(latent)
    self.step += 1

  def compute_returns(self, next_value: torch.Tensor) -> None:
    if self.step != self.cfg.ppo.rollout_steps:
      raise RuntimeError("cannot compute returns before rollout is full")
    if next_value.shape != (self.num_envs,):
      raise ValueError("next value must have shape [num_envs]")
    advantage = torch.zeros(self.num_envs, device=self.device)
    for index in reversed(range(self.step)):
      following_value = next_value if index == self.step - 1 else self.values[index + 1]
      not_done = (~self.dones[index]).float()
      delta = (
        self.rewards[index]
        + self.cfg.ppo.gamma * following_value * not_done
        - self.values[index]
      )
      advantage = (
        delta
        + self.cfg.ppo.gamma
        * self.cfg.ppo.gae_lambda
        * not_done
        * advantage
      )
      self.advantages[index] = advantage
    self.returns.copy_(self.advantages + self.values)

  def flattened(self) -> dict[str, torch.Tensor]:
    if self.step != self.cfg.ppo.rollout_steps:
      raise RuntimeError("cannot flatten an incomplete rollout")
    return {
      "observations": self.observations.flatten(0, 1),
      "factors": self.factors.flatten(0, 1),
      "latents": self.latents.flatten(0, 1),
      "previous_experts": self.previous_experts.flatten(0, 1),
      "actions": self.actions.flatten(0, 1),
      "log_probs": self.log_probs.flatten(0, 1),
      "values": self.values.flatten(0, 1),
      "advantages": self.advantages.flatten(0, 1),
      "returns": self.returns.flatten(0, 1),
      "route_targets": self.route_targets.flatten(0, 1),
    }


@dataclass(frozen=True)
class PpoUpdateMetrics:
  policy_loss: float
  value_loss: float
  entropy: float
  approximate_kl: float
  clip_fraction: float
  routing_loss: float
  routing_accuracy: float


class MetaPpoOptimizer:
  """Optimize teacher actor/critic and optionally its factor encoder."""

  def __init__(
    self,
    cfg: MetaRlCfg,
    actor_critic: MetaActorCritic,
    factor_encoder: EnvironmentFactorEncoder | None,
    *,
    learning_rate: float | None = None,
  ) -> None:
    self.cfg = cfg
    self.actor_critic = actor_critic
    self.factor_encoder = factor_encoder
    parameters: list[nn.Parameter] = list(actor_critic.parameters())
    if factor_encoder is not None:
      parameters.extend(factor_encoder.parameters())
    self.parameters = parameters
    self.optimizer = torch.optim.Adam(
      parameters,
      lr=cfg.ppo.learning_rate if learning_rate is None else learning_rate,
    )

  def update(
    self, buffer: MetaRolloutBuffer, *, use_estimated_latent: bool
  ) -> PpoUpdateMetrics:
    if use_estimated_latent == (self.factor_encoder is not None):
      raise ValueError(
        "teacher PPO needs a factor encoder; deploy PPO must omit it"
      )
    data = buffer.flattened()
    advantages = data["advantages"]
    advantages = (advantages - advantages.mean()) / (
      advantages.std(unbiased=False) + 1.0e-8
    )
    sample_count = len(advantages)
    mini_batch_size = sample_count // self.cfg.ppo.mini_batches
    if mini_batch_size <= 0 or sample_count % self.cfg.ppo.mini_batches:
      raise ValueError("rollout sample count must divide into mini-batches")

    totals = torch.zeros(7, dtype=torch.float64)
    update_count = 0
    stop_early = False
    for _ in range(self.cfg.ppo.learning_epochs):
      permutation = torch.randperm(sample_count, device=buffer.device)
      for start in range(0, sample_count, mini_batch_size):
        indices = permutation[start : start + mini_batch_size]
        if use_estimated_latent:
          latent = data["latents"][indices]
        else:
          assert self.factor_encoder is not None
          latent = self.factor_encoder(data["factors"][indices])
        logits, value = self.actor_critic(
          data["observations"][indices],
          latent,
          data["previous_experts"][indices],
        )
        distribution = torch.distributions.Categorical(logits=logits)
        new_log_prob = distribution.log_prob(data["actions"][indices])
        entropy_mean = distribution.entropy().mean()
        ratio = torch.exp(new_log_prob - data["log_probs"][indices])
        batch_advantage = advantages[indices]
        unclipped = ratio * batch_advantage
        clipped = torch.clamp(
          ratio,
          1.0 - self.cfg.ppo.clip_param,
          1.0 + self.cfg.ppo.clip_param,
        ) * batch_advantage
        policy_loss = -torch.minimum(unclipped, clipped).mean()

        old_value = data["values"][indices]
        clipped_value = old_value + torch.clamp(
          value - old_value,
          -self.cfg.ppo.clip_param,
          self.cfg.ppo.clip_param,
        )
        value_error = torch.square(value - data["returns"][indices])
        clipped_error = torch.square(
          clipped_value - data["returns"][indices]
        )
        value_loss = 0.5 * torch.maximum(value_error, clipped_error).mean()
        routing_loss = torch.zeros((), device=logits.device)
        routing_accuracy = torch.zeros((), device=logits.device)
        if not use_estimated_latent and torch.any(data["route_targets"] >= 0):
          route_targets = data["route_targets"][indices]
          if torch.any(route_targets < 0):
            raise ValueError("teacher rollout has no route targets")
          routing_loss = F.cross_entropy(logits, route_targets)
          routing_accuracy = (
            torch.argmax(logits, dim=-1) == route_targets
          ).float().mean()
        loss = (
          policy_loss
          + self.cfg.ppo.value_loss_coef * value_loss
          - self.cfg.ppo.entropy_coef * entropy_mean
          + self.cfg.ppo.routing_loss_coef * routing_loss
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters, self.cfg.ppo.max_grad_norm)
        self.optimizer.step()

        with torch.no_grad():
          log_ratio = new_log_prob - data["log_probs"][indices]
          approximate_kl = torch.mean((torch.exp(log_ratio) - 1.0) - log_ratio)
          clip_fraction = torch.mean(
            (torch.abs(ratio - 1.0) > self.cfg.ppo.clip_param).float()
          )
        totals += torch.tensor(
          (
            policy_loss.item(),
            value_loss.item(),
            entropy_mean.item(),
            approximate_kl.item(),
            clip_fraction.item(),
            routing_loss.item(),
            routing_accuracy.item(),
          ),
          dtype=torch.float64,
        )
        update_count += 1
        if (
          self.cfg.ppo.target_kl > 0.0
          and approximate_kl.item() > 1.5 * self.cfg.ppo.target_kl
        ):
          stop_early = True
          break
      if stop_early:
        break
    mean = totals / max(update_count, 1)
    return PpoUpdateMetrics(*[float(value) for value in mean])
