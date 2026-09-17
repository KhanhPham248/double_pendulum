"""Teacher, adaptation and deployable selector neural modules."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.distributions import Categorical

from .config import MetaRlCfg, MetaRlModelCfg


def _mlp(
  input_dim: int,
  hidden_dims: Sequence[int],
  output_dim: int,
  *,
  output_activation: nn.Module | None = None,
) -> nn.Sequential:
  layers: list[nn.Module] = []
  previous = input_dim
  for hidden in hidden_dims:
    layers.extend((nn.Linear(previous, hidden), nn.ELU()))
    previous = hidden
  layers.append(nn.Linear(previous, output_dim))
  if output_activation is not None:
    layers.append(output_activation)
  return nn.Sequential(*layers)


class EnvironmentFactorEncoder(nn.Module):
  """Train-only mapping from privileged dynamics factors to a latent."""

  def __init__(self, cfg: MetaRlModelCfg):
    super().__init__()
    self.cfg = cfg
    self.network = _mlp(
      cfg.factor_dim,
      cfg.factor_hidden_dims,
      cfg.latent_dim,
      output_activation=nn.Tanh(),
    )

  def forward(self, factors: torch.Tensor) -> torch.Tensor:
    if factors.shape[-1] != self.cfg.factor_dim:
      raise ValueError("privileged factor dimension does not match config")
    if not torch.isfinite(factors).all():
      raise ValueError("privileged factors contain non-finite values")
    return self.network(factors)


class MetaActorCritic(nn.Module):
  """Categorical high-level policy over full-task experts plus HOLD."""

  def __init__(self, cfg: MetaRlCfg):
    super().__init__()
    self.cfg = cfg
    input_dim = (
      cfg.model.observation_dim
      + cfg.model.latent_dim
      + len(cfg.expert_names)
    )
    feature_dim = cfg.model.meta_hidden_dims[-1]
    self.backbone = _mlp(input_dim, cfg.model.meta_hidden_dims[:-1], feature_dim)
    self.actor_head = nn.Linear(feature_dim, len(cfg.class_names))
    self.value_head = nn.Linear(feature_dim, 1)

  def forward(
    self,
    observation: torch.Tensor,
    latent: torch.Tensor,
    previous_expert_one_hot: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    expected = (observation.shape[0], self.cfg.model.observation_dim)
    if observation.shape != expected:
      raise ValueError(f"meta observation must have shape {expected}")
    if latent.shape != (observation.shape[0], self.cfg.model.latent_dim):
      raise ValueError("meta latent shape does not match config")
    if previous_expert_one_hot.shape != (
      observation.shape[0], len(self.cfg.expert_names)
    ):
      raise ValueError("previous expert one-hot shape does not match registry")
    inputs = torch.cat((observation, latent, previous_expert_one_hot), dim=-1)
    if not torch.isfinite(inputs).all():
      raise ValueError("meta actor input contains non-finite values")
    features = self.backbone(inputs)
    return self.actor_head(features), self.value_head(features).squeeze(-1)

  def distribution(
    self,
    observation: torch.Tensor,
    latent: torch.Tensor,
    previous_expert_one_hot: torch.Tensor,
  ) -> tuple[Categorical, torch.Tensor]:
    logits, value = self.forward(observation, latent, previous_expert_one_hot)
    return Categorical(logits=logits), value

  def act(
    self,
    observation: torch.Tensor,
    latent: torch.Tensor,
    previous_expert_one_hot: torch.Tensor,
    *,
    deterministic: bool = False,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    distribution, value = self.distribution(
      observation, latent, previous_expert_one_hot
    )
    action = (
      torch.argmax(distribution.logits, dim=-1)
      if deterministic
      else distribution.sample()
    )
    return action, distribution.log_prob(action), value, distribution.probs

  def evaluate_actions(
    self,
    observation: torch.Tensor,
    latent: torch.Tensor,
    previous_expert_one_hot: torch.Tensor,
    actions: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    distribution, value = self.distribution(
      observation, latent, previous_expert_one_hot
    )
    return distribution.log_prob(actions), distribution.entropy(), value


class AdaptationModule(nn.Module):
  """Deployable recurrent estimator mapping observation history to latent."""

  def __init__(self, cfg: MetaRlModelCfg):
    super().__init__()
    self.cfg = cfg
    self.register_buffer("observation_mean", torch.zeros(cfg.observation_dim))
    self.register_buffer("observation_std", torch.ones(cfg.observation_dim))
    self.gru = nn.GRU(
      input_size=cfg.observation_dim,
      hidden_size=cfg.adaptation_hidden_dim,
      num_layers=cfg.adaptation_num_layers,
      batch_first=True,
    )
    self.head = nn.Sequential(
      nn.LayerNorm(cfg.adaptation_hidden_dim),
      nn.Linear(cfg.adaptation_hidden_dim, cfg.latent_dim),
      nn.Tanh(),
    )

  def set_observation_normalizer(
    self, mean: torch.Tensor, std: torch.Tensor
  ) -> None:
    expected = (self.cfg.observation_dim,)
    if mean.shape != expected or std.shape != expected:
      raise ValueError("adaptation normalizer shapes do not match config")
    self.observation_mean.copy_(mean)
    self.observation_std.copy_(torch.clamp(std, min=1.0e-6))

  def initial_hidden(
    self, batch_size: int, *, device: torch.device | str | None = None
  ) -> torch.Tensor:
    return torch.zeros(
      self.cfg.adaptation_num_layers,
      batch_size,
      self.cfg.adaptation_hidden_dim,
      device=device if device is not None else self.observation_mean.device,
      dtype=self.observation_mean.dtype,
    )

  def forward(
    self, history: torch.Tensor, hidden: torch.Tensor | None = None
  ) -> tuple[torch.Tensor, torch.Tensor]:
    expected_last = self.cfg.observation_dim
    if history.ndim != 3 or history.shape[-1] != expected_last:
      raise ValueError("adaptation history must have shape [batch, time, obs]")
    normalized = (history - self.observation_mean) / (
      self.observation_std + 1.0e-6
    )
    recurrent, next_hidden = self.gru(normalized, hidden)
    return self.head(recurrent[:, -1, :]), next_hidden


class DeployAdapter(nn.Module):
  def __init__(self, adaptation: AdaptationModule):
    super().__init__()
    self.adaptation = adaptation

  def forward(self, history: torch.Tensor) -> torch.Tensor:
    latent, _ = self.adaptation(history)
    return latent


class DeployMetaSelector(nn.Module):
  def __init__(self, actor_critic: MetaActorCritic):
    super().__init__()
    self.actor_critic = actor_critic

  def forward(
    self,
    observation: torch.Tensor,
    latent: torch.Tensor,
    previous_expert_one_hot: torch.Tensor,
  ) -> torch.Tensor:
    logits, _ = self.actor_critic(
      observation, latent, previous_expert_one_hot
    )
    return torch.softmax(logits, dim=-1)
