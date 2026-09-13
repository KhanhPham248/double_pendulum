"""Per-episode behavior and control metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from double_pendulum.common import DEFAULT_COMBINED_REWARD, CombinedRewardSpec
from double_pendulum.common.observations import wrapped_angle_error
from double_pendulum.common.rewards import (
  absolute_link_alignment,
  absolute_link_velocity_l2,
  capture_weighted_upright_velocity_l2,
  upright_capture,
  upright_proximity,
  upright_velocity_l2,
)


@dataclass
class EpisodeAccumulator:
  control_dt: float
  torque_limit_nm: float
  required_hold_s: float
  angle_threshold_rad: float
  velocity_threshold_rad_s: float
  reward_spec: CombinedRewardSpec = DEFAULT_COMBINED_REWARD
  steps: int = 0
  stable_steps: int = 0
  stable_streak: int = 0
  best_streak: int = 0
  first_upright_step: int | None = None
  success_step: int | None = None
  post_success_steps: int = 0
  post_success_stable_steps: int = 0
  post_success_angle_error_sum: float = 0.0
  post_success_angular_velocity_sum: float = 0.0
  post_success_abs_torque_sum: float = 0.0
  post_success_saturation_steps: int = 0
  post_success_escape_count: int = 0
  escape_count: int = 0
  was_stable: bool = False
  episode_return: float = 0.0
  angle_error_sum: float = 0.0
  angular_velocity_sum: float = 0.0
  abs_torque_sum: float = 0.0
  torque_square_integral: float = 0.0
  action_rate_square_sum: float = 0.0
  saturation_steps: int = 0
  previous_action: float = 0.0
  final_angle_error: float = 0.0

  @property
  def required_hold_steps(self) -> int:
    return math.ceil(self.required_hold_s / self.control_dt)

  def add(self, qpos: np.ndarray, qvel: np.ndarray, action: float) -> None:
    q1_error = abs(float(wrapped_angle_error(qpos[0], math.pi, np)))
    q2_error = abs(float(wrapped_angle_error(qpos[1], 0.0, np)))
    angle_error = max(q1_error, q2_error)
    angular_velocity = float(np.max(np.abs(qvel)))
    stable = (
      angle_error < self.angle_threshold_rad
      and angular_velocity < self.velocity_threshold_rad_s
    )
    if stable:
      self.stable_steps += 1
      self.stable_streak += 1
      if self.first_upright_step is None:
        self.first_upright_step = self.steps
    else:
      self.stable_streak = 0
      if self.was_stable:
        self.escape_count += 1
        if self.success_step is not None:
          self.post_success_escape_count += 1
    self.best_streak = max(self.best_streak, self.stable_streak)
    self.was_stable = stable

    torque = action * self.torque_limit_nm
    reward = self.reward_spec
    if reward.formula_version == 1:
      alignment = 0.5 * (math.cos(qpos[0] - math.pi) + math.cos(qpos[1]))
      capture = 0.0
      error_sq = q1_error**2 + q2_error**2
      velocity_cost = math.exp(
        -error_sq / reward.capture_angle_sigma_rad**2
      ) * float(np.square(qvel).sum())
    elif reward.formula_version == 2:
      alignment = float(absolute_link_alignment(qpos, np))
      capture = float(
        upright_proximity(
          qpos,
          np,
          sigma_rad=reward.capture_angle_sigma_rad,
        )
      )
      velocity_cost = float(
        upright_velocity_l2(
          qpos,
          qvel,
          np,
          sigma_rad=reward.capture_angle_sigma_rad,
        )
      )
    else:
      alignment = float(absolute_link_alignment(qpos, np))
      capture = float(
        upright_capture(
          qpos,
          np,
          base_sigma_rad=reward.capture_base_sigma_rad,
          elbow_sigma_rad=reward.capture_elbow_sigma_rad,
        )
      )
      velocity_cost = float(
        capture_weighted_upright_velocity_l2(
          qpos,
          qvel,
          np,
          base_sigma_rad=reward.capture_base_sigma_rad,
          elbow_sigma_rad=reward.capture_elbow_sigma_rad,
        )
      )
    global_velocity_cost = 0.0
    if reward.formula_version >= 4:
      global_velocity_cost = float(absolute_link_velocity_l2(qvel, np))
    action_rate = (action - self.previous_action) ** 2
    self.episode_return += (
      reward.link_alignment_weight * alignment
      + reward.upright_capture_weight * capture
      + reward.balancing_bonus_weight * float(stable)
      + reward.global_velocity_weight * global_velocity_cost
      + reward.upright_velocity_weight * velocity_cost
      + reward.torque_weight * action**2
      + reward.action_rate_weight * action_rate
    )
    self.angle_error_sum += angle_error
    self.angular_velocity_sum += angular_velocity
    self.abs_torque_sum += abs(torque)
    self.torque_square_integral += torque**2 * self.control_dt
    self.action_rate_square_sum += action_rate
    self.saturation_steps += int(abs(action) > 0.99)
    self.previous_action = action
    self.final_angle_error = angle_error
    self.steps += 1

    was_successful_before_step = self.success_step is not None
    if self.success_step is None and self.stable_streak >= self.required_hold_steps:
      self.success_step = self.steps - 1
    if was_successful_before_step:
      self.post_success_steps += 1
      self.post_success_stable_steps += int(stable)
      self.post_success_angle_error_sum += angle_error
      self.post_success_angular_velocity_sum += angular_velocity
      self.post_success_abs_torque_sum += abs(torque)
      self.post_success_saturation_steps += int(abs(action) > 0.99)

  def result(self) -> dict[str, float | None]:
    count = max(self.steps, 1)
    first = self.first_upright_step
    post_count = self.post_success_steps
    return {
      "success": float(self.best_streak * self.control_dt >= self.required_hold_s),
      "episode_return": self.episode_return,
      "swing_up_time_s": None if first is None else first * self.control_dt,
      "time_to_success_s": (
        None if self.success_step is None else (self.success_step + 1) * self.control_dt
      ),
      "longest_hold_s": self.best_streak * self.control_dt,
      "upright_fraction": self.stable_steps / count,
      "post_success_duration_s": post_count * self.control_dt,
      "post_success_stable_fraction": (
        None if post_count == 0 else self.post_success_stable_steps / post_count
      ),
      "post_success_escape_count": (
        None if self.success_step is None else float(self.post_success_escape_count)
      ),
      "post_success_mean_angle_error_rad": (
        None
        if post_count == 0
        else self.post_success_angle_error_sum / post_count
      ),
      "post_success_mean_angular_velocity_rad_s": (
        None
        if post_count == 0
        else self.post_success_angular_velocity_sum / post_count
      ),
      "post_success_mean_abs_torque_nm": (
        None
        if post_count == 0
        else self.post_success_abs_torque_sum / post_count
      ),
      "post_success_action_saturation_fraction": (
        None
        if post_count == 0
        else self.post_success_saturation_steps / post_count
      ),
      "final_angle_error_rad": self.final_angle_error,
      "mean_angle_error_rad": self.angle_error_sum / count,
      "mean_angular_velocity_rad_s": self.angular_velocity_sum / count,
      "mean_abs_torque_nm": self.abs_torque_sum / count,
      "rms_torque_nm": math.sqrt(
        self.torque_square_integral / (count * self.control_dt)
      ),
      "torque_squared_integral": self.torque_square_integral,
      "action_rate_l2": self.action_rate_square_sum / count,
      "action_saturation_fraction": self.saturation_steps / count,
      "escape_count": float(self.escape_count),
    }
