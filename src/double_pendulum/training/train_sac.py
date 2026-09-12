"""Off-policy SAC training loop for a registered mJLab task."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv

from double_pendulum.algorithms.sac import ReplayBuffer, SACAgent
from double_pendulum.algorithms.sac.config import SACTrainConfig
from double_pendulum.common import DEFAULT_CONTRACT
from double_pendulum.export.checkpoints import (
  save_torch_checkpoint,
  update_latest_checkpoint,
)
from double_pendulum.export.sac_exporter import export_sac_policy
from double_pendulum.tasks import get_task, list_tasks
from double_pendulum.tasks.combined import mdp

from .common import (
  MetricsLogger,
  capture_rng_state,
  prepare_run_dir,
  restore_rng_state,
)

CHECKPOINT_FORMAT = "double_pendulum_sac_v2"


def _actor_observation(observations: dict[str, torch.Tensor]) -> torch.Tensor:
  actor = observations.get("actor")
  expected = DEFAULT_CONTRACT.observation_dim
  if actor is None or actor.ndim != 2 or actor.shape[1] != expected:
    raise RuntimeError(f"actor observation contract must be [num_envs, {expected}]")
  return actor


def _save(
  run_dir: Path,
  *,
  agent: SACAgent,
  replay: ReplayBuffer,
  config: SACTrainConfig,
  transitions: int,
  collects: int,
  evaluation,
) -> Path:
  checkpoint = run_dir / "checkpoints" / f"step_{transitions:012d}.pt"
  state = {
    "format": CHECKPOINT_FORMAT,
    "task": config.task,
    "train_config": asdict(config),
    "agent": agent.state_dict(),
    "replay": replay.state_dict(),
    "transitions": transitions,
    "collects": collects,
    "rng": capture_rng_state(),
  }
  save_torch_checkpoint(state, checkpoint)
  update_latest_checkpoint(checkpoint)
  try:
    parity = export_sac_policy(
      agent,
      run_dir,
      task=config.task,
      checkpoint=str(checkpoint.relative_to(run_dir)),
      evaluation=evaluation,
    )
    print(
      "ONNX_UPDATED "
      f"abs_error={parity.max_abs_error:.3e} rel_error={parity.max_rel_error:.3e}",
      flush=True,
    )
  except Exception as error:
    print(f"ONNX_EXPORT_FAILED checkpoint={checkpoint}: {error}", flush=True)
    if config.export_fail_fast:
      raise
  return checkpoint


def train(config: SACTrainConfig) -> Path:
  if config.device.startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError("CUDA was requested but PyTorch cannot see a CUDA device")
  task = get_task(config.task)
  random.seed(config.seed)
  torch.manual_seed(config.seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.seed)

  run_dir = prepare_run_dir(
    task=config.task,
    algorithm="sac",
    requested=config.run_dir,
    resume=config.resume,
  )
  logger = MetricsLogger(run_dir)
  env = ManagerBasedRlEnv(cfg=task.make_env_cfg(config.num_envs), device=config.device)
  started = time.monotonic()
  try:
    observations, _ = env.reset(seed=config.seed)
    actor_obs = _actor_observation(observations)
    action_dim = env.action_manager.total_action_dim
    if action_dim != DEFAULT_CONTRACT.action_dim:
      raise RuntimeError(f"expected one base-motor action, got {action_dim}")
    agent = SACAgent(actor_obs.shape[1], action_dim, config.agent, config.device)
    replay = ReplayBuffer(
      config.replay_capacity, actor_obs.shape[1], action_dim, config.device
    )
    transitions = 0
    collects = 0
    if config.resume is not None:
      state = torch.load(config.resume, map_location=config.device, weights_only=False)
      if state.get("format") != CHECKPOINT_FORMAT or state.get("task") != config.task:
        raise ValueError("checkpoint format or task does not match this SAC run")
      if state["agent"]["config"] != asdict(config.agent):
        raise ValueError("SAC agent config differs from the checkpoint")
      agent.load_state_dict(state["agent"])
      replay.load_state_dict(state["replay"])
      transitions = int(state["transitions"])
      collects = int(state["collects"])
      restore_rng_state(state["rng"])
    else:
      agent.observe(actor_obs)

    next_checkpoint = (
      transitions // config.checkpoint_interval + 1
    ) * config.checkpoint_interval
    stable_streak = torch.zeros(env.num_envs, dtype=torch.long, device=config.device)
    best_streak = torch.zeros_like(stable_streak)
    last_saved_at = -1
    while transitions < config.total_transitions:
      if transitions < config.learning_starts:
        actions = torch.empty(
          (env.num_envs, action_dim), device=agent.device
        ).uniform_(-1.0, 1.0)
      else:
        actions = agent.act(actor_obs)

      next_observations, rewards, terminated, truncated, _ = env.step(actions)
      next_actor_obs = _actor_observation(next_observations)
      dones = terminated | truncated
      replay.add(actor_obs, actions, rewards, next_actor_obs, dones)
      agent.observe(next_actor_obs)
      actor_obs = next_actor_obs
      transitions += env.num_envs
      collects += 1

      qpos, qvel = mdp.state(env)
      stable = mdp.stable_mask(
        qpos,
        qvel,
        angle_threshold_rad=task.config.stable_angle_rad,
        velocity_threshold_rad_s=task.config.stable_velocity_rad_s,
      )
      stable_streak = torch.where(
        stable, stable_streak + 1, torch.zeros_like(stable_streak)
      )
      best_streak = torch.maximum(best_streak, stable_streak)
      stable_streak[dones] = 0

      update_metrics: dict[str, float] = {}
      if transitions >= config.learning_starts and len(replay) >= config.batch_size:
        for _ in range(config.updates_per_collect):
          update_metrics = agent.update(replay.sample(config.batch_size))

      if collects % config.log_interval == 0:
        elapsed = max(time.monotonic() - started, 1e-6)
        values: dict[str, float | int] = {
          "train/collect": collects,
          "train/mean_step_reward": float(rewards.mean()),
          "train/replay_size": len(replay),
          "train/done_fraction": float(dones.float().mean()),
          "train/stable_fraction": float(stable.float().mean()),
          "train/longest_hold_s": float(best_streak.float().max())
          * DEFAULT_CONTRACT.control_timestep_s,
          "control/mean_abs_action": float(actions.abs().mean()),
          "control/action_saturation_fraction": float(
            (actions.abs() > 0.99).float().mean()
          ),
          "performance/transitions_per_second": transitions / elapsed,
          **{f"sac/{name}": value for name, value in update_metrics.items()},
        }
        logger.write(transitions, values)
        print(
          json.dumps({"transitions": transitions, **values}, sort_keys=True),
          flush=True,
        )

      if transitions >= next_checkpoint:
        _save(
          run_dir,
          agent=agent,
          replay=replay,
          config=config,
          transitions=transitions,
          collects=collects,
          evaluation=task.evaluation,
        )
        last_saved_at = transitions
        while next_checkpoint <= transitions:
          next_checkpoint += config.checkpoint_interval

    if last_saved_at != transitions:
      _save(
        run_dir,
        agent=agent,
        replay=replay,
        config=config,
        transitions=transitions,
        collects=collects,
        evaluation=task.evaluation,
      )
  finally:
    logger.close()
    env.close()
  return run_dir


def parse_args(argv: list[str] | None = None) -> SACTrainConfig:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", choices=list_tasks(), default="combined")
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--num-envs", type=int, default=512)
  parser.add_argument("--total-transitions", type=int, default=2_000_000)
  parser.add_argument("--replay-capacity", type=int, default=1_000_000)
  parser.add_argument("--batch-size", type=int, default=1024)
  parser.add_argument("--learning-starts", type=int, default=25_000)
  parser.add_argument("--updates-per-collect", type=int, default=1)
  parser.add_argument("--checkpoint-interval", type=int, default=200_000)
  parser.add_argument("--log-interval", type=int, default=25)
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--run-dir")
  parser.add_argument("--resume")
  parser.add_argument("--export-fail-fast", action="store_true")
  return SACTrainConfig(**vars(parser.parse_args(argv)))


def main(argv: list[str] | None = None) -> None:
  run_dir = train(parse_args(argv))
  print(f"TRAIN_COMPLETE run_dir={run_dir}")
