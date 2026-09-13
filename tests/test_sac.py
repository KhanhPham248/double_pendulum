from __future__ import annotations

import importlib.util

import pytest

if importlib.util.find_spec("torch") is None:
  pytest.skip("PyTorch is not installed", allow_module_level=True)

import torch
from torch import nn

from double_pendulum.algorithms.sac import (
  ReplayBuffer,
  SACAgent,
  SACConfig,
  TransitionBatch,
)
from double_pendulum.algorithms.sac.config import SACTrainConfig


def test_replay_buffer_handles_batched_ring_wrap() -> None:
  replay = ReplayBuffer(capacity=5, obs_dim=2, action_dim=1, device="cpu")
  for offset in (0, 3):
    observations = torch.arange(offset, offset + 6, dtype=torch.float32).reshape(3, 2)
    replay.add(
      observations,
      torch.zeros(3, 1),
      torch.ones(3),
      observations + 1.0,
      torch.zeros(3),
    )
  assert len(replay) == 5
  batch = replay.sample(4)
  assert batch.observations.shape == (4, 2)
  assert batch.dones.shape == (4, 1)
  state = replay.state_dict()
  assert state["observations"].device.type == "cpu"
  restored = ReplayBuffer(capacity=5, obs_dim=2, action_dim=1, device="cpu")
  restored.load_state_dict(state)
  assert len(restored) == len(replay)
  assert restored.position == replay.position


def test_sac_update_produces_finite_metrics() -> None:
  torch.manual_seed(0)
  agent = SACAgent(
    6,
    1,
    SACConfig(hidden_dims=(32, 32), learning_rate_schedule="linear"),
    "cpu",
  )
  replay = ReplayBuffer(capacity=64, obs_dim=6, action_dim=1, device="cpu")
  obs = torch.randn(32, 6)
  replay.add(
    obs,
    torch.tanh(torch.randn(32, 1)),
    torch.randn(32),
    torch.randn(32, 6),
    torch.zeros(32),
  )
  agent.observe(obs)
  initial_lr = agent.actor_optimizer.param_groups[0]["lr"]
  metrics = agent.update(replay.sample(16))
  assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
  assert agent.actor_optimizer.param_groups[0]["lr"] < initial_lr


def test_checkpoint_round_trip_preserves_deterministic_actor() -> None:
  torch.manual_seed(4)
  config = SACConfig(hidden_dims=(16, 16))
  agent = SACAgent(6, 1, config, "cpu")
  agent.observe(torch.randn(32, 6))
  probe = torch.randn(8, 6)
  expected = agent.act(probe, deterministic=True)
  restored = SACAgent(6, 1, config, "cpu")
  restored.load_state_dict(agent.state_dict())
  assert torch.equal(restored.act(probe, deterministic=True), expected)
  assert restored.normalizer.count.item() == 32


class _FixedActor(nn.Module):
  def sample(self, observations, deterministic=False):
    actions = torch.zeros((observations.shape[0], 1))
    log_probability = torch.full_like(actions, -0.5)
    return actions, log_probability


class _ConstantCritic(nn.Module):
  def __init__(self, value: float):
    super().__init__()
    self.value = value

  def forward(self, observations, actions):
    return torch.full((observations.shape[0], 1), self.value)


def test_bellman_target_matches_sac_reference_equation() -> None:
  config = SACConfig(
    gamma=0.5,
    init_alpha=0.2,
    reward_scale=2.0,
    normalize_observations=False,
    hidden_dims=(8, 8),
  )
  agent = SACAgent(6, 1, config, "cpu")
  agent.actor = _FixedActor()
  agent.targets = nn.ModuleList((_ConstantCritic(2.0), _ConstantCritic(3.0)))
  batch = TransitionBatch(
    observations=torch.zeros(2, 6),
    actions=torch.zeros(2, 1),
    rewards=torch.tensor([[1.0], [1.0]]),
    next_observations=torch.zeros(2, 6),
    dones=torch.tensor([[0.0], [1.0]]),
  )
  # r*2 + gamma*(1-done)*(min(Q1,Q2) - alpha*log_pi)
  expected = torch.tensor([[3.05], [2.0]])
  assert torch.allclose(agent.compute_target(batch), expected)


def test_nan_guard_rejects_corrupt_replay_batch() -> None:
  agent = SACAgent(6, 1, SACConfig(hidden_dims=(8, 8)), "cpu")
  batch = TransitionBatch(
    observations=torch.full((2, 6), float("nan")),
    actions=torch.zeros(2, 1),
    rewards=torch.zeros(2, 1),
    next_observations=torch.zeros(2, 6),
    dones=torch.zeros(2, 1),
  )
  with pytest.raises(FloatingPointError):
    agent.update(batch)


def test_update_budget_is_independent_of_vectorized_collection_size() -> None:
  from double_pendulum.algorithms.sac import UpdateBudget

  small_batches = UpdateBudget(0.25)
  large_batches = UpdateBudget(0.25)
  small_updates = sum(small_batches.add(64) for _ in range(8))
  large_updates = large_batches.add(512)
  assert small_updates == large_updates == 128
  assert small_batches.actual_ratio == large_batches.actual_ratio == 0.25


def test_sac_training_defaults_use_transition_based_updates() -> None:
  config = SACTrainConfig()
  assert config.batch_size == 256
  assert config.utd_ratio == 0.25
  assert config.agent.learning_rate_schedule == "constant"
  assert config.agent.learning_rate_decay_steps == 100_000
  assert config.evaluation_episodes == 10
