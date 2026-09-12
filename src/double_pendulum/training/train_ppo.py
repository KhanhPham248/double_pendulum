"""On-policy PPO training through the mJLab/RSL-RL integration."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict

import torch
import yaml

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends

from double_pendulum.algorithms.ppo import (
  DoublePendulumPpoRunner,
  PPOTrainConfig,
  make_ppo_runner_cfg,
)
from double_pendulum.tasks import get_task, list_tasks

from .common import prepare_run_dir


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
  env = ManagerBasedRlEnv(cfg=task.make_env_cfg(config.num_envs), device=config.device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
  runner_cfg = make_ppo_runner_cfg(config)
  _dump_config(run_dir / "train_config.yaml", asdict(config))
  _dump_config(run_dir / "ppo_config.yaml", asdict(runner_cfg))
  runner = DoublePendulumPpoRunner(
    wrapped,
    asdict(runner_cfg),
    str(run_dir),
    config.device,
    task_name=config.task,
    evaluation_spec=task.evaluation,
    export_fail_fast=config.export_fail_fast,
  )
  try:
    if config.resume is not None:
      runner.load(config.resume, map_location=config.device)
    runner.learn(
      num_learning_iterations=config.max_iterations,
      init_at_random_ep_len=True,
    )
  finally:
    wrapped.close()
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
  return PPOTrainConfig(**vars(parser.parse_args(argv)))


def main(argv: list[str] | None = None) -> None:
  run_dir = train(parse_args(argv))
  print(f"TRAIN_COMPLETE run_dir={run_dir}")
