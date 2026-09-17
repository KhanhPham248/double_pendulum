#!/usr/bin/env python3
"""Train privileged categorical PPO over the two frozen full-task experts."""

from __future__ import annotations

import argparse
from pathlib import Path
import random

import numpy as np
import torch

from double_pendulum.meta.checkpoint import (
  make_checkpoint,
  save_checkpoint_atomic,
)
from double_pendulum.meta.config import load_meta_rl_cfg
from double_pendulum.meta.engine import collect_teacher_rollout
from double_pendulum.meta.models import EnvironmentFactorEncoder, MetaActorCritic
from double_pendulum.meta.ppo import MetaPpoOptimizer
from double_pendulum.meta.session import MetaMujocoSession


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--config", type=Path, default=Path("configs/meta_rl/double_pendulum_rma_v1.yaml")
  )
  parser.add_argument("--model", type=Path, default=Path("assets/double_pendulum.xml"))
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=16)
  parser.add_argument("--iterations", type=int)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args(argv)
  if min(args.num_envs, args.iterations or 1) <= 0:
    raise ValueError("num-envs and iterations must be positive")
  if args.device.startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError("CUDA was requested but PyTorch cannot see a CUDA device")

  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
  root = Path(__file__).resolve().parents[1]
  cfg = load_meta_rl_cfg(root / args.config)
  iterations = args.iterations or cfg.ppo.max_iterations
  device = torch.device(args.device)
  session = MetaMujocoSession(
    cfg=cfg,
    project_root=root,
    model_path=args.model,
    num_envs=args.num_envs,
    device=device,
    seed=args.seed,
  )
  actor = MetaActorCritic(cfg).to(device)
  encoder = EnvironmentFactorEncoder(cfg.model).to(device)
  optimizer = MetaPpoOptimizer(cfg, actor, encoder)
  args.output_dir.mkdir(parents=True, exist_ok=True)
  session.reset()
  try:
    for iteration in range(1, iterations + 1):
      actor.eval()
      encoder.eval()
      buffer, _, rollout = collect_teacher_rollout(session, actor, encoder)
      actor.train()
      encoder.train()
      update = optimizer.update(buffer, use_estimated_latent=False)
      print(
        f"[teacher {iteration}/{iterations}] "
        f"reward={rollout.mean_meta_reward:.5f} "
        f"switch={rollout.switch_rate:.5f} "
        f"sac={rollout.selected_expert_fractions[0]:.3f} "
        f"ppo={rollout.selected_expert_fractions[1]:.3f} "
        f"kl={update.approximate_kl:.6f}",
        flush=True,
      )
      if iteration % cfg.ppo.save_interval == 0 or iteration == iterations:
        payload = make_checkpoint(
          stage="teacher",
          cfg=cfg,
          actor=actor,
          encoder=encoder,
          adaptation=None,
          iteration=iteration,
          registry_sha256=session.registry_sha256,
          optimizer_state_dict=optimizer.optimizer.state_dict(),
          metrics={"rollout": vars(rollout), "ppo": vars(update)},
        )
        save_checkpoint_atomic(
          payload, args.output_dir / f"teacher_{iteration:06d}.pt"
        )
        save_checkpoint_atomic(payload, args.output_dir / "teacher_latest.pt")
  finally:
    session.close()


if __name__ == "__main__":
  main()
