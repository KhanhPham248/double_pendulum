from __future__ import annotations

import importlib.util

import pytest

if importlib.util.find_spec("torch") is None:
  pytest.skip("PyTorch is not installed", allow_module_level=True)

import torch

from double_pendulum.common import DEFAULT_CONTRACT
from double_pendulum.tasks.combined import CombinedTaskCfg
from double_pendulum.tasks.combined import mdp


class _Data:
  joint_pos = torch.tensor([[0.0, torch.pi / 2]])
  joint_vel = torch.tensor([[1.5, -2.0]])


class _Entity:
  data = _Data()


class _Env:
  scene = {"pendulum": _Entity()}


def test_observation_order_is_part_of_the_policy_contract() -> None:
  actual = mdp.policy_state(_Env())
  expected = torch.tensor([[0.0, 1.0, 1.0, 0.0, 1.5, -2.0]])
  assert torch.allclose(actual, expected, atol=1e-6)
  assert DEFAULT_CONTRACT.observation_names == (
    "sin_q1",
    "cos_q1",
    "sin_q2",
    "cos_q2",
    "qdot1",
    "qdot2",
  )


def test_combined_success_duration_is_five_seconds() -> None:
  config = CombinedTaskCfg()
  assert config.episode_length_s == 20.0
  assert config.success_hold_s == 5.0
