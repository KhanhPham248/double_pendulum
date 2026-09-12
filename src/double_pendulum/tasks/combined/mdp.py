"""Reset, observation, reward and termination terms for the combined task."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from double_pendulum.common.observations import policy_observation

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def state(env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor]:
  asset = env.scene["pendulum"]
  return asset.data.joint_pos[:, :2], asset.data.joint_vel[:, :2]


def angle_error(angle: torch.Tensor, target: float) -> torch.Tensor:
  difference = angle - target
  return torch.atan2(torch.sin(difference), torch.cos(difference))


def reset_pendulum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  *,
  near_upright_probability: float,
  hanging_probability: float,
  near_upright_angle_rad: float,
  hanging_angle_rad: float,
  random_velocity_rad_s: float,
) -> None:
  if not 0.0 <= near_upright_probability <= 1.0:
    raise ValueError("near_upright_probability must lie in [0, 1]")
  if not 0.0 <= hanging_probability <= 1.0:
    raise ValueError("hanging_probability must lie in [0, 1]")
  if near_upright_probability + hanging_probability > 1.0:
    raise ValueError("reset probabilities must sum to at most one")
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

  count = len(env_ids)
  qpos = torch.empty((count, 2), device=env.device).uniform_(-math.pi, math.pi)
  qvel = torch.empty((count, 2), device=env.device).uniform_(
    -random_velocity_rad_s, random_velocity_rad_s
  )
  regime = torch.rand(count, device=env.device)
  near_upright = regime < near_upright_probability
  hanging = (regime >= near_upright_probability) & (
    regime < near_upright_probability + hanging_probability
  )

  if torch.any(near_upright):
    size = int(near_upright.sum())
    qpos[near_upright, 0] = math.pi + torch.empty(size, device=env.device).uniform_(
      -near_upright_angle_rad, near_upright_angle_rad
    )
    qpos[near_upright, 1] = torch.empty(size, device=env.device).uniform_(
      -near_upright_angle_rad, near_upright_angle_rad
    )
    qvel[near_upright] *= 0.25
  if torch.any(hanging):
    qpos[hanging] = torch.empty(
      (int(hanging.sum()), 2), device=env.device
    ).uniform_(-hanging_angle_rad, hanging_angle_rad)
    qvel[hanging] *= 0.5

  env.scene["pendulum"].write_joint_state_to_sim(qpos, qvel, env_ids=env_ids)


def policy_state(env: ManagerBasedRlEnv) -> torch.Tensor:
  qpos, qvel = state(env)
  return policy_observation(qpos, qvel, torch)


def upright_alignment(env: ManagerBasedRlEnv) -> torch.Tensor:
  qpos, _ = state(env)
  return 0.5 * (torch.cos(qpos[:, 0] - math.pi) + torch.cos(qpos[:, 1]))


def stable_mask(
  qpos: torch.Tensor,
  qvel: torch.Tensor,
  *,
  angle_threshold_rad: float,
  velocity_threshold_rad_s: float,
) -> torch.Tensor:
  angles_ok = (angle_error(qpos[:, 0], math.pi).abs() < angle_threshold_rad) & (
    angle_error(qpos[:, 1], 0.0).abs() < angle_threshold_rad
  )
  return angles_ok & (qvel.abs().amax(dim=1) < velocity_threshold_rad_s)


def balancing_bonus(
  env: ManagerBasedRlEnv,
  *,
  angle_threshold_rad: float,
  velocity_threshold_rad_s: float,
) -> torch.Tensor:
  return stable_mask(
    *state(env),
    angle_threshold_rad=angle_threshold_rad,
    velocity_threshold_rad_s=velocity_threshold_rad_s,
  ).float()


def near_goal_velocity_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  qpos, qvel = state(env)
  error_sq = torch.square(angle_error(qpos[:, 0], math.pi)) + torch.square(
    angle_error(qpos[:, 1], 0.0)
  )
  return torch.exp(-error_sq / (0.35**2)) * torch.sum(torch.square(qvel), dim=1)


def torque_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  action = env.action_manager.get_term("base_torque").raw_action
  return torch.sum(torch.square(action), dim=1)


def time_out(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.episode_length_buf >= env.max_episode_length


def numerical_failure(env: ManagerBasedRlEnv) -> torch.Tensor:
  qpos, qvel = state(env)
  return ~(torch.isfinite(qpos).all(dim=1) & torch.isfinite(qvel).all(dim=1))
