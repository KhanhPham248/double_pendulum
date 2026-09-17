"""CPU MuJoCo session for RMA-meta training over full-task experts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import torch

from double_pendulum.common import DEFAULT_COMBINED_REWARD, PolicyContract
from double_pendulum.common.observations import wrapped_angle_error
from double_pendulum.common.rewards import (
  absolute_link_alignment,
  absolute_link_velocity_l2,
  capture_weighted_upright_velocity_l2,
  upright_capture,
)
from double_pendulum.runtime.mujoco_sim import MujocoDoublePendulum

from .config import MetaRlCfg
from .expert_registry import load_frozen_experts
from .factors import build_privileged_factors
from .history import ObservationHistory
from .runtime import ExpertSelectionState, TorqueCrossfader


@dataclass(frozen=True)
class MetaStepResult:
  observation: torch.Tensor
  reward: torch.Tensor
  done: torch.Tensor
  timeout: torch.Tensor
  proposed_action: torch.Tensor
  selected_expert: torch.Tensor
  switched: torch.Tensor


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


class MetaMujocoSession:
  """Run the selected ONNX experts in a small vector of independent MuJoCo envs.

  This is intentionally separate from the legacy mJLab `combined` trainer. It
  gives the meta stages one closed-loop path while keeping the original SAC and
  PPO task registry unchanged. The first implementation randomizes mass,
  damping and observation noise; torque/delay factors remain nominal until
  their simulator randomizers are added.
  """

  def __init__(
    self,
    *,
    cfg: MetaRlCfg,
    project_root: str | Path,
    model_path: str | Path,
    num_envs: int,
    device: torch.device | str,
    seed: int,
    mass_range: tuple[float, float] = (0.85, 1.15),
    damping_range: tuple[float, float] = (0.75, 1.25),
    observation_noise_range: tuple[float, float] = (0.0, 0.02),
  ) -> None:
    if num_envs <= 0:
      raise ValueError("num_envs must be positive")
    if seed < 0:
      raise ValueError("seed must be non-negative")
    for name, bounds in (
      ("mass_range", mass_range),
      ("damping_range", damping_range),
      ("observation_noise_range", observation_noise_range),
    ):
      if (
        len(bounds) != 2
        or not np.isfinite(bounds[0])
        or not np.isfinite(bounds[1])
        or bounds[1] < bounds[0]
      ):
        raise ValueError(f"invalid {name}")
      if name != "observation_noise_range" and bounds[0] <= 0.0:
        raise ValueError(f"invalid {name}")
    if observation_noise_range[0] < 0.0:
      raise ValueError("observation noise must be non-negative")
    self.cfg = cfg
    self.project_root = Path(project_root).resolve()
    self.model_path = Path(model_path)
    if not self.model_path.is_absolute():
      self.model_path = self.project_root / self.model_path
    self.model_path = self.model_path.resolve()
    self.num_envs = num_envs
    self.device = torch.device(device)
    self.rng = np.random.default_rng(seed)
    self.mass_range = mass_range
    self.damping_range = damping_range
    self.observation_noise_range = observation_noise_range

    registry_path = Path(cfg.expert_registry)
    if not registry_path.is_absolute():
      registry_path = self.project_root / registry_path
    self.registry_path = registry_path.resolve()
    self.registry, self.experts = load_frozen_experts(
      self.project_root, self.registry_path
    )
    if self.registry.expert_names != cfg.expert_names:
      raise ValueError("meta config and expert registry order differ")
    first_policy = self.experts[cfg.expert_names[0]]
    self.contract: PolicyContract = first_policy.manifest.contract
    if self.contract.observation_dim != cfg.model.observation_dim:
      raise ValueError("expert observation dimension differs from meta config")
    if self.contract.action_dim != 1:
      raise ValueError("double-pendulum meta session needs one torque action")

    self.episode_length_steps = round(
      20.0 / self.contract.control_timestep_s
    )
    self.simulators: list[MujocoDoublePendulum] = []
    self.observations_np = np.zeros(
      (num_envs, cfg.model.observation_dim), dtype=np.float32
    )
    self.episode_steps = np.zeros(num_envs, dtype=np.int64)
    self.episode_generation = np.zeros(num_envs, dtype=np.int64)
    self.mass_scale = np.ones(num_envs, dtype=np.float32)
    self.damping_scale = np.ones(num_envs, dtype=np.float32)
    self.observation_noise_std = np.zeros(num_envs, dtype=np.float32)
    self.selection = ExpertSelectionState(
      num_envs,
      len(cfg.expert_names),
      cfg.initial_expert_index,
      min_dwell_decisions=cfg.runtime.min_dwell_decisions,
      device=self.device,
    )
    self.crossfader = TorqueCrossfader(
      num_envs,
      cfg.runtime.crossfade_steps,
      device=self.device,
    )
    self.history = ObservationHistory(
      num_envs,
      cfg.model.history_steps,
      cfg.model.observation_dim,
      device=self.device,
    )

  @property
  def registry_sha256(self) -> str:
    return _sha256(self.registry_path)

  @property
  def episode_ids(self) -> torch.Tensor:
    indices = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
    generation = torch.as_tensor(
      self.episode_generation, device=self.device, dtype=torch.long
    )
    return generation * self.num_envs + indices

  @property
  def actor_observation(self) -> torch.Tensor:
    observation = torch.as_tensor(
      self.observations_np, device=self.device, dtype=torch.float32
    )
    if not torch.isfinite(observation).all():
      raise RuntimeError("meta actor observation contains non-finite values")
    return observation

  def _sample_dynamics(self, ids: np.ndarray) -> None:
    self.mass_scale[ids] = self.rng.uniform(
      *self.mass_range, size=len(ids)
    ).astype(np.float32)
    self.damping_scale[ids] = self.rng.uniform(
      *self.damping_range, size=len(ids)
    ).astype(np.float32)
    self.observation_noise_std[ids] = self.rng.uniform(
      *self.observation_noise_range, size=len(ids)
    ).astype(np.float32)

  def _observe(self, index: int, raw: np.ndarray) -> np.ndarray:
    noise = self.rng.normal(
      0.0, float(self.observation_noise_std[index]), size=raw.shape
    )
    observation = np.asarray(raw + noise, dtype=np.float32)
    if not np.isfinite(observation).all():
      raise FloatingPointError("meta session produced a non-finite observation")
    return observation

  def _reset_ids(self, ids: np.ndarray, *, increment_generation: bool) -> None:
    self._sample_dynamics(ids)
    for index in ids:
      self.simulators[int(index)] = MujocoDoublePendulum(
        self.model_path,
        self.contract,
        mass_scale=float(self.mass_scale[index]),
        damping_scale=float(self.damping_scale[index]),
      )
      raw = self.simulators[int(index)].reset(
        seed=int(self.rng.integers(0, 2**31 - 1)), mode="hanging"
      )
      self.observations_np[index] = self._observe(int(index), raw)
      self.episode_steps[index] = 0
      if increment_generation:
        self.episode_generation[index] += 1
    ids_tensor = torch.as_tensor(ids, device=self.device, dtype=torch.long)
    self.selection.reset(ids_tensor)
    self.crossfader.reset(ids_tensor)
    self.history.reset(ids_tensor)

  def reset(self) -> torch.Tensor:
    self.simulators = [None] * self.num_envs  # type: ignore[list-item]
    self.episode_generation.fill(0)
    ids = np.arange(self.num_envs, dtype=np.int64)
    self._reset_ids(ids, increment_generation=False)
    self.selection.reset()
    self.crossfader.reset()
    self.history.reset()
    self.history.append(self.actor_observation)
    return self.actor_observation

  def privileged_factors(self) -> torch.Tensor:
    ones = np.ones(self.num_envs, dtype=np.float32)
    zeros = np.zeros(self.num_envs, dtype=np.float32)
    return build_privileged_factors(
      torch.as_tensor(self.mass_scale, device=self.device),
      torch.as_tensor(self.damping_scale, device=self.device),
      torch.as_tensor(self.damping_scale, device=self.device),
      torch.as_tensor(ones, device=self.device),
      torch.as_tensor(zeros, device=self.device),
      torch.as_tensor(self.observation_noise_std, device=self.device),
    )

  def _expert_actions(self) -> np.ndarray:
    actions = []
    for name in self.cfg.expert_names:
      result = self.experts[name].act(self.observations_np)
      actions.append(np.asarray(result[:, 0], dtype=np.float32))
    return np.stack(actions, axis=1)

  def _reward(self, index: int, normalized_action: float) -> float:
    qpos = self.simulators[index].qpos
    qvel = self.simulators[index].qvel
    reward = DEFAULT_COMBINED_REWARD
    alignment = float(absolute_link_alignment(qpos, np))
    capture = float(
      upright_capture(
        qpos,
        np,
        base_sigma_rad=reward.capture_base_sigma_rad,
        elbow_sigma_rad=reward.capture_elbow_sigma_rad,
      )
    )
    q1_error = abs(float(wrapped_angle_error(qpos[0], np.pi, np)))
    q2_error = abs(float(wrapped_angle_error(qpos[1], 0.0, np)))
    stable = float(
      q1_error < 0.21 and q2_error < 0.21 and np.max(np.abs(qvel)) < 1.0
    )
    global_velocity = float(absolute_link_velocity_l2(qvel, np))
    upright_velocity = float(
      capture_weighted_upright_velocity_l2(
        qpos,
        qvel,
        np,
        base_sigma_rad=reward.capture_base_sigma_rad,
        elbow_sigma_rad=reward.capture_elbow_sigma_rad,
      )
    )
    return float(
      reward.link_alignment_weight * alignment
      + reward.upright_capture_weight * capture
      + reward.balancing_bonus_weight * stable
      + reward.global_velocity_weight * global_velocity
      + reward.upright_velocity_weight * upright_velocity
      + reward.torque_weight * normalized_action**2
    )

  def step_meta(self, proposed_action: torch.Tensor) -> MetaStepResult:
    if proposed_action.shape != (self.num_envs,):
      raise ValueError("meta action must have shape [num_envs]")
    proposed_action = proposed_action.to(self.device, dtype=torch.long)
    selected, switched = self.selection.resolve(proposed_action)
    self.crossfader.begin_switch(switched)
    reward_sum = torch.zeros(self.num_envs, device=self.device)
    done_any = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
    timeout_any = torch.zeros_like(done_any)
    active = torch.ones_like(done_any)

    for _ in range(self.cfg.ppo.meta_decimation):
      expert_actions = self._expert_actions()
      selected_np = selected.detach().cpu().numpy()
      target_normalized = expert_actions[
        np.arange(self.num_envs), selected_np
      ]
      target_torque = torch.as_tensor(
        target_normalized * self.contract.torque_limit_nm,
        device=self.device,
      )
      target_torque = torch.where(
        active,
        target_torque,
        self.crossfader.last_torque,
      )
      torque = self.crossfader.apply(target_torque)
      normalized = (
        torque / self.contract.torque_limit_nm
      ).detach().cpu().numpy()
      for index in np.flatnonzero(active.detach().cpu().numpy()):
        raw = self.simulators[int(index)].step(float(normalized[index]))
        self.observations_np[index] = self._observe(int(index), raw)
        reward_sum[index] += self._reward(int(index), float(normalized[index]))
        self.episode_steps[index] += 1
      timeout = torch.as_tensor(
        self.episode_steps >= self.episode_length_steps,
        device=self.device,
        dtype=torch.bool,
      ) & active
      newly_done = timeout
      timeout_any |= timeout
      done_any |= newly_done
      active &= ~newly_done
      done_ids = np.flatnonzero(newly_done.detach().cpu().numpy())
      if len(done_ids):
        self._reset_ids(done_ids, increment_generation=True)
      self.history.append(self.actor_observation)

    reward_sum -= self.cfg.ppo.switch_penalty * switched.float()
    return MetaStepResult(
      observation=self.actor_observation,
      reward=reward_sum,
      done=done_any,
      timeout=timeout_any,
      proposed_action=proposed_action,
      selected_expert=selected,
      switched=switched,
    )

  def close(self) -> None:
    self.simulators.clear()
