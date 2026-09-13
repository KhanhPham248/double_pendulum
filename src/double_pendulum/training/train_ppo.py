"""On-policy PPO training through the mJLab/RSL-RL integration."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict

import torch
import yaml

from double_pendulum.algorithms.ppo import (
  DoublePendulumPpoRunner,
  PPOTrainConfig,
  make_ppo_runner_cfg,
)
from double_pendulum.tasks import get_task, list_tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends

from .common import WandbSession, prepare_run_dir


def _dump_config(path, values) -> None:
  serializable = json.loads(json.dumps(values))
  with path.open("w", encoding="utf-8") as output:
    yaml.safe_dump(serializable, output, sort_keys=False)


def train(config: PPOTrainConfig):
  if config.device.startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError("CUDA was requested but PyTorch cannot see a CUDA device")
  random.seed(config.seed)
  torch.manual_seed(config.seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.seed)
  configure_torch_backends()

  task = get_task(config.task)
  run_dir = prepare_run_dir(
    task=config.task,
    algorithm="ppo",
    requested=config.run_dir,
    resume=config.resume,
  )
  runner_cfg = make_ppo_runner_cfg(config)
  if config.resume is None:
    _dump_config(run_dir / "train_config.yaml", asdict(config))
    _dump_config(run_dir / "task_config.yaml", asdict(task.config))
    _dump_config(run_dir / "ppo_config.yaml", asdict(runner_cfg))
  wandb_session = WandbSession(
    run_dir,
    enabled=config.use_wandb,
    project=config.wandb_project,
    entity=config.wandb_entity,
    name=config.wandb_run_name,
    config={"train": asdict(config), "task": asdict(task.config)},
  )
  if wandb_session.url is not None:
    print(f"WANDB_RUN={wandb_session.url}", flush=True)
  env = None
  wrapped = None
  try:
    env = ManagerBasedRlEnv(
      cfg=task.make_env_cfg(config.num_envs),
      device=config.device,
    )
    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
    runner = DoublePendulumPpoRunner(
      wrapped,
      asdict(runner_cfg),
      str(run_dir),
      config.device,
      task_name=config.task,
      evaluation_spec=task.evaluation,
      reward_spec=task.config.reward,
      export_fail_fast=config.export_fail_fast,
      evaluation_episodes=config.evaluation_episodes,
    )
    if config.resume is not None:
      runner.load(config.resume, map_location=config.device)
    runner.learn(
      num_learning_iterations=config.max_iterations,
      init_at_random_ep_len=True,
    )
  finally:
    if wrapped is not None:
      wrapped.close()
    elif env is not None:
      env.close()
    wandb_session.finish()
  return run_dir


def parse_args(argv: list[str] | None = None) -> PPOTrainConfig:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", choices=list_tasks(), default="combined")
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--num-envs", type=int, default=512)
  parser.add_argument("--max-iterations", type=int, default=2_000)
  parser.add_argument("--save-interval", type=int, default=50)
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--run-dir")
  parser.add_argument("--resume")
  parser.add_argument("--export-fail-fast", action="store_true")
  parser.add_argument("--evaluation-episodes", type=int, default=10)
  parser.add_argument("--wandb", action="store_true", dest="use_wandb")
  parser.add_argument("--wandb-project", default="double-pendulum")
  parser.add_argument("--wandb-entity")
  parser.add_argument("--wandb-run-name")
  return PPOTrainConfig(**vars(parser.parse_args(argv)))


def main(argv: list[str] | None = None) -> None:
  run_dir = train(parse_args(argv))
  print(f"TRAIN_COMPLETE run_dir={run_dir}")
