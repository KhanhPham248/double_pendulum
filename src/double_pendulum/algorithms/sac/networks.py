"""Neural networks and running observation normalization for SAC."""

from __future__ import annotations

import torch
from torch import nn


def mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int) -> nn.Sequential:
  layers: list[nn.Module] = []
  previous = input_dim
  for width in hidden_dims:
    layers.extend((nn.Linear(previous, width), nn.ReLU()))
    previous = width
  layers.append(nn.Linear(previous, output_dim))
  return nn.Sequential(*layers)


class RunningNormalizer(nn.Module):
  """Numerically stable running mean/variance stored in checkpoints and ONNX."""

  def __init__(self, size: int, *, clip: float = 10.0, epsilon: float = 1e-5):
    super().__init__()
    if size <= 0 or clip <= 0.0 or epsilon <= 0.0:
      raise ValueError("normalizer size, clip and epsilon must be positive")
    self.clip = clip
    self.epsilon = epsilon
    self.register_buffer("mean", torch.zeros(size))
    self.register_buffer("variance", torch.ones(size))
    self.register_buffer("count", torch.zeros(()))

  @torch.no_grad()
  def update(self, values: torch.Tensor) -> None:
    if values.ndim != 2 or values.shape[1] != self.mean.numel():
      raise ValueError("normalizer expects [batch, observation_dim]")
    if values.shape[0] == 0:
      return
    batch_mean = values.mean(dim=0)
    batch_variance = values.var(dim=0, unbiased=False)
    batch_count = torch.as_tensor(
      values.shape[0], device=values.device, dtype=values.dtype
    )
    if self.count.item() == 0.0:
      self.mean.copy_(batch_mean)
      self.variance.copy_(batch_variance)
      self.count.copy_(batch_count)
      return

    delta = batch_mean - self.mean
    total = self.count + batch_count
    new_mean = self.mean + delta * batch_count / total
    old_m2 = self.variance * self.count
    batch_m2 = batch_variance * batch_count
    new_m2 = old_m2 + batch_m2 + delta.square() * self.count * batch_count / total
    self.mean.copy_(new_mean)
    self.variance.copy_(new_m2 / total)
    self.count.copy_(total)

  def forward(self, values: torch.Tensor) -> torch.Tensor:
    normalized = (values - self.mean) / torch.sqrt(self.variance + self.epsilon)
    return normalized.clamp(-self.clip, self.clip)


class Actor(nn.Module):
  """Gaussian policy squashed to normalized actions in `[-1, 1]`."""

  def __init__(self, obs_dim: int, action_dim: int, hidden_dims: tuple[int, ...]):
    super().__init__()
    self.backbone = mlp(obs_dim, hidden_dims, hidden_dims[-1])
    self.mean = nn.Linear(hidden_dims[-1], action_dim)
    self.log_std = nn.Linear(hidden_dims[-1], action_dim)

  def sample(
    self, observations: torch.Tensor, deterministic: bool = False
  ) -> tuple[torch.Tensor, torch.Tensor | None]:
    features = self.backbone(observations)
    mean = self.mean(features)
    if deterministic:
      return torch.tanh(mean), None

    log_std = self.log_std(features).clamp(-20.0, 2.0)
    distribution = torch.distributions.Normal(mean, log_std.exp())
    unsquashed = distribution.rsample()
    action = torch.tanh(unsquashed)
    correction = torch.log(1.0 - action.square() + 1e-6)
    log_prob = (distribution.log_prob(unsquashed) - correction).sum(-1, keepdim=True)
    return action, log_prob


class Critic(nn.Module):
  def __init__(self, obs_dim: int, action_dim: int, hidden_dims: tuple[int, ...]):
    super().__init__()
    self.network = mlp(obs_dim + action_dim, hidden_dims, 1)

  def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    return self.network(torch.cat((observations, actions), dim=-1))
