from __future__ import annotations

import math

import numpy as np

from double_pendulum.common import DEFAULT_COMBINED_REWARD
from double_pendulum.common.rewards import (
  absolute_link_alignment,
  absolute_link_velocity_l2,
  action_rate_l2,
  capture_weighted_upright_velocity_l2,
  upright_capture,
)
from double_pendulum.evaluation.metrics import EpisodeAccumulator


def test_absolute_link_alignment_orders_key_configurations() -> None:
  hanging = np.array([0.0, 0.0])
  horizontal = np.array([math.pi / 2.0, 0.0])
  upright = np.array([math.pi, 0.0])

  assert np.isclose(absolute_link_alignment(hanging, np), -1.0)
  assert np.isclose(absolute_link_alignment(horizontal, np), 0.0, atol=1e-12)
  assert np.isclose(absolute_link_alignment(upright, np), 1.0)


def test_second_link_reward_uses_its_world_angle() -> None:
  qpos = np.array([math.pi / 2.0, -math.pi / 2.0])
  # Link 1 is horizontal and link 2 points down in world coordinates.
  assert np.isclose(absolute_link_alignment(qpos, np), -0.5, atol=1e-12)


def test_alignment_is_symmetric_around_upright() -> None:
  left = absolute_link_alignment(np.array([math.pi - 0.4, 0.1]), np)
  right = absolute_link_alignment(np.array([math.pi + 0.4, -0.1]), np)
  assert np.isclose(left, right)


def _capture(qpos: np.ndarray) -> float:
  reward = DEFAULT_COMBINED_REWARD
  return float(
    upright_capture(
      qpos,
      np,
      base_sigma_rad=reward.capture_base_sigma_rad,
      elbow_sigma_rad=reward.capture_elbow_sigma_rad,
    )
  )


def test_upright_capture_is_smooth_and_goal_centered() -> None:
  upright = _capture(np.array([math.pi, 0.0]))
  nearby = _capture(np.array([math.pi + 0.2, -0.1]))
  hanging = _capture(np.array([0.0, 0.0]))
  assert np.isclose(upright, 1.0)
  assert upright > nearby > hanging


def test_upright_capture_uses_relative_elbow_error() -> None:
  correct_pose = _capture(np.array([math.pi, 0.0]))
  bent_elbow = _capture(np.array([math.pi, 0.45]))
  assert np.isclose(bent_elbow, math.exp(-1.0))
  assert correct_pose > bent_elbow


def test_upright_velocity_cost_uses_absolute_link_velocities() -> None:
  qpos = np.array([math.pi, 0.0])
  reward = DEFAULT_COMBINED_REWARD
  kwargs = {
    "base_sigma_rad": reward.capture_base_sigma_rad,
    "elbow_sigma_rad": reward.capture_elbow_sigma_rad,
  }
  stationary = capture_weighted_upright_velocity_l2(
    qpos, np.zeros(2), np, **kwargs
  )
  moving = capture_weighted_upright_velocity_l2(
    qpos, np.array([1.0, -1.0]), np, **kwargs
  )
  assert np.isclose(stationary, 0.0)
  assert np.isclose(moving, 0.5)


def test_global_velocity_cost_applies_away_from_upright() -> None:
  stationary = absolute_link_velocity_l2(np.zeros(2), np)
  spinning = absolute_link_velocity_l2(np.array([6.0, 6.0]), np)
  assert np.isclose(stationary, 0.0)
  assert spinning > 0.0


def _single_step_return(action: float, qvel: np.ndarray | None = None) -> float:
  accumulator = EpisodeAccumulator(
    control_dt=0.02,
    torque_limit_nm=6.0,
    required_hold_s=5.0,
    angle_threshold_rad=0.21,
    velocity_threshold_rad_s=1.0,
    reward_spec=DEFAULT_COMBINED_REWARD,
  )
  accumulator.add(
    np.array([math.pi, 0.0]),
    np.zeros(2) if qvel is None else qvel,
    action,
  )
  return accumulator.episode_return


def test_stationary_zero_torque_upright_has_highest_local_reward() -> None:
  stationary = _single_step_return(0.0)
  full_torque = _single_step_return(1.0)
  moving = _single_step_return(0.0, np.array([1.0, 0.0]))
  assert stationary > full_torque
  assert stationary > moving


def test_action_rate_cost_detects_a_control_jump() -> None:
  steady = action_rate_l2(np.array([0.4]), np.array([0.4]), np)
  jump = action_rate_l2(np.array([0.4]), np.array([-0.4]), np)
  assert np.isclose(steady, 0.0)
  assert jump > steady


def test_straight_hanging_state_is_not_rewarded() -> None:
  accumulator = EpisodeAccumulator(
    control_dt=0.02,
    torque_limit_nm=6.0,
    required_hold_s=5.0,
    angle_threshold_rad=0.21,
    velocity_threshold_rad_s=1.0,
    reward_spec=DEFAULT_COMBINED_REWARD,
  )
  accumulator.add(np.array([0.0, 0.0]), np.zeros(2), 0.0)
  assert accumulator.episode_return < 0.0


def test_fast_hanging_spin_is_penalized_more_than_quiet_hanging() -> None:
  quiet = EpisodeAccumulator(
    control_dt=0.02,
    torque_limit_nm=6.0,
    required_hold_s=5.0,
    angle_threshold_rad=0.21,
    velocity_threshold_rad_s=1.0,
    reward_spec=DEFAULT_COMBINED_REWARD,
  )
  spinning = EpisodeAccumulator(
    control_dt=0.02,
    torque_limit_nm=6.0,
    required_hold_s=5.0,
    angle_threshold_rad=0.21,
    velocity_threshold_rad_s=1.0,
    reward_spec=DEFAULT_COMBINED_REWARD,
  )
  quiet.add(np.array([0.0, 0.0]), np.zeros(2), 0.0)
  spinning.add(np.array([0.0, 0.0]), np.array([6.0, 6.0]), 0.0)
  assert spinning.episode_return < quiet.episode_return
