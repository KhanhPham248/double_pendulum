"""CPU contract tests for the isolated double-pendulum RMA-meta core."""

from __future__ import annotations

from pathlib import Path
import unittest

import torch

from double_pendulum.meta.config import (
  MetaRlCfg,
  MetaRlPpoCfg,
  load_meta_rl_cfg,
)
from double_pendulum.meta.expert_registry import load_expert_registry
from double_pendulum.meta.factors import build_privileged_factors
from double_pendulum.meta.history import ObservationHistory
from double_pendulum.meta.models import (
  AdaptationModule,
  EnvironmentFactorEncoder,
  MetaActorCritic,
)
from double_pendulum.meta.ppo import MetaPpoOptimizer, MetaRolloutBuffer
from double_pendulum.meta.runtime import (
  ExpertSelectionState,
  TorqueCrossfader,
  confidence_gated_action,
)


ROOT = Path(__file__).resolve().parents[1]


class MetaRlCoreTest(unittest.TestCase):
  def setUp(self) -> None:
    torch.manual_seed(11)
    self.cfg = MetaRlCfg(
      ppo=MetaRlPpoCfg(rollout_steps=8, learning_epochs=1, mini_batches=2)
    )

  def test_config_and_registry_are_selected_without_touching_combined_task(self) -> None:
    loaded = load_meta_rl_cfg(ROOT / "configs/meta_rl/double_pendulum_rma_v1.yaml")
    self.assertEqual(loaded.expert_names, ("policy_sac", "policy_ppo"))
    self.assertEqual(loaded.model.observation_dim, 6)
    self.assertEqual(loaded.model.latent_dim, 4)
    registry = load_expert_registry(
      ROOT / "configs/meta_rl/double_pendulum_expert_registry_v1.yaml"
    )
    self.assertEqual(registry.expert_names, loaded.expert_names)
    self.assertEqual(loaded.task_id, "combined")

  def test_models_have_teacher_and_deploy_shapes(self) -> None:
    batch = 4
    actor = MetaActorCritic(self.cfg)
    encoder = EnvironmentFactorEncoder(self.cfg.model)
    adaptation = AdaptationModule(self.cfg.model)
    observation = torch.randn(batch, 6)
    factors = torch.randn(batch, 6)
    previous = torch.nn.functional.one_hot(
      torch.zeros(batch, dtype=torch.long), len(self.cfg.expert_names)
    ).float()
    latent = encoder(factors)
    logits, values = actor(observation, latent, previous)
    estimated, hidden = adaptation(torch.randn(batch, 50, 6))
    self.assertEqual(logits.shape, (batch, 3))
    self.assertEqual(values.shape, (batch,))
    self.assertEqual(estimated.shape, (batch, 4))
    self.assertEqual(hidden.shape, (1, batch, 64))

  def test_factor_layout_and_history_reset(self) -> None:
    factors = build_privileged_factors(
      torch.ones(3),
      torch.ones(3),
      torch.ones(3),
      torch.ones(3),
      torch.zeros(3),
      torch.zeros(3),
    )
    self.assertEqual(factors.shape, (3, 6))
    torch.testing.assert_close(factors, torch.zeros(3, 6))

    history = ObservationHistory(2, 3, 6, device="cpu")
    history.append(torch.ones(2, 6))
    history.reset(torch.tensor([1]))
    self.assertEqual(history.valid_steps.tolist(), [1, 0])
    torch.testing.assert_close(history.buffer[1], torch.zeros(3, 6))

  def test_selector_hold_dwell_and_torque_crossfade(self) -> None:
    selection = ExpertSelectionState(
      2, 2, 0, min_dwell_decisions=2, device="cpu"
    )
    selected, changed = selection.resolve(torch.tensor([1, 1]))
    torch.testing.assert_close(selected, torch.tensor([1, 1]))
    torch.testing.assert_close(changed, torch.tensor([True, True]))
    selected, changed = selection.resolve(torch.tensor([0, 0]))
    torch.testing.assert_close(selected, torch.tensor([1, 1]))
    torch.testing.assert_close(changed, torch.tensor([False, False]))

    crossfade = TorqueCrossfader(2, 2, device="cpu")
    crossfade.begin_switch(torch.tensor([True, False]))
    output = crossfade.apply(torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(output, torch.tensor([0.5, 1.0]))
    output = crossfade.apply(torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(output, torch.ones(2))

    probabilities = torch.tensor([[0.4, 0.3, 0.3], [0.7, 0.2, 0.1]])
    torch.testing.assert_close(
      confidence_gated_action(
        probabilities, hold_index=2, confidence_threshold=0.6
      ),
      torch.tensor([2, 0]),
    )

  def test_teacher_ppo_update_is_finite(self) -> None:
    envs = 2
    actor = MetaActorCritic(self.cfg)
    encoder = EnvironmentFactorEncoder(self.cfg.model)
    buffer = MetaRolloutBuffer(self.cfg, envs, device="cpu")
    for _ in range(self.cfg.ppo.rollout_steps):
      observation = torch.randn(envs, 6)
      factors = torch.randn(envs, 6)
      previous = torch.nn.functional.one_hot(
        torch.randint(0, 2, (envs,)), 2
      ).float()
      with torch.no_grad():
        latent = encoder(factors)
        action, log_prob, value, _ = actor.act(observation, latent, previous)
      buffer.add(
        observation=observation,
        previous_expert=previous,
        action=action,
        log_prob=log_prob,
        value=value,
        reward=torch.randn(envs),
        done=torch.zeros(envs, dtype=torch.bool),
        factor=factors,
        route_target=torch.tensor([0, 1]),
      )
    buffer.compute_returns(torch.zeros(envs))
    metrics = MetaPpoOptimizer(self.cfg, actor, encoder).update(
      buffer, use_estimated_latent=False
    )
    self.assertTrue(torch.isfinite(torch.tensor(list(vars(metrics).values()))).all())


if __name__ == "__main__":
  unittest.main()
