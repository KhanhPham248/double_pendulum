"""Rollout collection for double-pendulum teacher and deployment PPO."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .models import AdaptationModule, EnvironmentFactorEncoder, MetaActorCritic
from .ppo import MetaRolloutBuffer
from .session import MetaMujocoSession


@dataclass(frozen=True)
class RolloutMetrics:
  mean_meta_reward: float
  done_rate: float
  timeout_rate: float
  switch_rate: float
  class_fractions: tuple[float, ...]
  selected_expert_fractions: tuple[float, ...]


def _metrics(
  rewards: list[torch.Tensor],
  dones: list[torch.Tensor],
  timeouts: list[torch.Tensor],
  switches: list[torch.Tensor],
  proposals: list[torch.Tensor],
  selected: list[torch.Tensor],
  num_classes: int,
  num_experts: int,
) -> RolloutMetrics:
  reward = torch.cat(rewards)
  done = torch.cat(dones).float()
  timeout = torch.cat(timeouts).float()
  switched = torch.cat(switches).float()
  proposal = torch.cat(proposals)
  selected_expert = torch.cat(selected)
  return RolloutMetrics(
    mean_meta_reward=float(reward.mean()),
    done_rate=float(done.mean()),
    timeout_rate=float(timeout.mean()),
    switch_rate=float(switched.mean()),
    class_fractions=tuple(
      float((proposal == index).float().mean())
      for index in range(num_classes)
    ),
    selected_expert_fractions=tuple(
      float((selected_expert == index).float().mean())
      for index in range(num_experts)
    ),
  )


@torch.no_grad()
def collect_teacher_rollout(
  session: MetaMujocoSession,
  actor: MetaActorCritic,
  encoder: EnvironmentFactorEncoder,
) -> tuple[MetaRolloutBuffer, torch.Tensor, RolloutMetrics]:
  cfg = session.cfg
  buffer = MetaRolloutBuffer(cfg, session.num_envs, device=session.device)
  rewards: list[torch.Tensor] = []
  dones: list[torch.Tensor] = []
  timeouts: list[torch.Tensor] = []
  switches: list[torch.Tensor] = []
  proposals: list[torch.Tensor] = []
  selected: list[torch.Tensor] = []
  for _ in range(cfg.ppo.rollout_steps):
    observation = session.actor_observation
    previous = session.selection.one_hot
    factor = session.privileged_factors()
    latent = encoder(factor)
    action, log_prob, value, _ = actor.act(observation, latent, previous)
    result = session.step_meta(action)
    _, next_value = actor(
      session.actor_observation,
      encoder(session.privileged_factors()),
      session.selection.one_hot,
    )
    buffer.add(
      observation=observation,
      previous_expert=previous,
      action=action,
      log_prob=log_prob,
      value=value,
      reward=result.reward,
      done=result.done,
      factor=factor,
    )
    rewards.append(result.reward)
    dones.append(result.done)
    timeouts.append(result.timeout)
    switches.append(result.switched)
    proposals.append(action)
    selected.append(result.selected_expert)
  _, next_value = actor(
    session.actor_observation,
    encoder(session.privileged_factors()),
    session.selection.one_hot,
  )
  buffer.compute_returns(next_value)
  return (
    buffer,
    next_value,
    _metrics(
      rewards,
      dones,
      timeouts,
      switches,
      proposals,
      selected,
      len(cfg.class_names),
      len(cfg.expert_names),
    ),
  )


@torch.no_grad()
def collect_deploy_rollout(
  session: MetaMujocoSession,
  actor: MetaActorCritic,
  adaptation: AdaptationModule,
) -> tuple[MetaRolloutBuffer, torch.Tensor, RolloutMetrics]:
  cfg = session.cfg
  buffer = MetaRolloutBuffer(cfg, session.num_envs, device=session.device)
  rewards: list[torch.Tensor] = []
  dones: list[torch.Tensor] = []
  timeouts: list[torch.Tensor] = []
  switches: list[torch.Tensor] = []
  proposals: list[torch.Tensor] = []
  selected: list[torch.Tensor] = []
  for _ in range(cfg.ppo.rollout_steps):
    observation = session.actor_observation
    previous = session.selection.one_hot
    latent, _ = adaptation(session.history.buffer)
    action, log_prob, value, _ = actor.act(observation, latent, previous)
    result = session.step_meta(action)
    next_latent, _ = adaptation(session.history.buffer)
    _, next_value = actor(
      session.actor_observation,
      next_latent,
      session.selection.one_hot,
    )
    buffer.add(
      observation=observation,
      previous_expert=previous,
      action=action,
      log_prob=log_prob,
      value=value,
      reward=result.reward,
      done=result.done,
      latent=latent,
    )
    rewards.append(result.reward)
    dones.append(result.done)
    timeouts.append(result.timeout)
    switches.append(result.switched)
    proposals.append(action)
    selected.append(result.selected_expert)
  next_latent, _ = adaptation(session.history.buffer)
  _, next_value = actor(
    session.actor_observation,
    next_latent,
    session.selection.one_hot,
  )
  buffer.compute_returns(next_value)
  return (
    buffer,
    next_value,
    _metrics(
      rewards,
      dones,
      timeouts,
      switches,
      proposals,
      selected,
      len(cfg.class_names),
      len(cfg.expert_names),
    ),
  )
