"""Per-episode behavior and control metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from double_pendulum.common.observations import wrapped_angle_error


@dataclass
class EpisodeAccumulator:
  control_dt: float
  torque_limit_nm: float
  required_hold_s: float
  angle_threshold_rad: float
  velocity_threshold_rad_s: float
  steps: int = 0
  stable_steps: int = 0
  stable_streak: int = 0
  best_streak: int = 0
  first_upright_step: int | None = None
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
    self.best_streak = max(self.best_streak, self.stable_streak)
    self.was_stable = stable

    torque = action * self.torque_limit_nm
    upright_reward = 0.5 * (math.cos(qpos[0] - math.pi) + math.cos(qpos[1]))
    error_sq = q1_error**2 + q2_error**2
    velocity_cost = math.exp(-error_sq / 0.35**2) * float(np.square(qvel).sum())
    self.episode_return += (
      upright_reward + 0.5 * float(stable) - 0.03 * velocity_cost - 0.002 * action**2
    )
    self.angle_error_sum += angle_error
    self.angular_velocity_sum += angular_velocity
    self.abs_torque_sum += abs(torque)
    self.torque_square_integral += torque**2 * self.control_dt
    self.action_rate_square_sum += (action - self.previous_action) ** 2
    self.saturation_steps += int(abs(action) > 0.99)
    self.previous_action = action
    self.final_angle_error = angle_error
    self.steps += 1

  def result(self) -> dict[str, float | None]:
    count = max(self.steps, 1)
    first = self.first_upright_step
    return {
      "success": float(self.best_streak * self.control_dt >= self.required_hold_s),
      "episode_return": self.episode_return,
      "swing_up_time_s": None if first is None else first * self.control_dt,
      "longest_hold_s": self.best_streak * self.control_dt,
      "upright_fraction": self.stable_steps / count,
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
