#!/usr/bin/env python3
"""Fine-tune only the deploy-aware meta selector with frozen adapter/experts."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from double_pendulum.meta.checkpoint import (
  load_checkpoint_models,
  make_checkpoint,
  save_checkpoint_atomic,
)
from double_pendulum.meta.config import load_meta_rl_cfg
from double_pendulum.meta.engine import collect_deploy_rollout
from double_pendulum.meta.models import MetaActorCritic
from double_pendulum.meta.ppo import MetaPpoOptimizer
from double_pendulum.meta.session import MetaMujocoSession


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--config", type=Path, default=Path("configs/meta_rl/double_pendulum_rma_v1.yaml")
  )
  parser.add_argument("--adaptation-checkpoint", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--model", type=Path, default=Path("assets/double_pendulum.xml"))
  parser.add_argument("--num-envs", type=int, default=16)
  parser.add_argument("--iterations", type=int)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--seed", type=int, default=45)
  args = parser.parse_args(argv)
  root = Path(__file__).resolve().parents[1]
  cfg = load_meta_rl_cfg(root / args.config)
  device = torch.device(args.device)
  payload, source_cfg, actor, encoder, adaptation = load_checkpoint_models(
    args.adaptation_checkpoint,
    map_location=device,
    expected_stage="adaptation",
    expected_expert_names=cfg.expert_names,
  )
  if source_cfg != cfg or adaptation is None:
    raise ValueError("adaptation checkpoint is incompatible with fine-tuning config")
  for parameter in encoder.parameters():
    parameter.requires_grad_(False)
  for parameter in adaptation.parameters():
    parameter.requires_grad_(False)
  encoder.eval()
  adaptation.eval()
  optimizer = MetaPpoOptimizer(
    cfg,
    actor,
    factor_encoder=None,
    learning_rate=cfg.ppo.finetune_learning_rate,
  )
  session = MetaMujocoSession(
    cfg=cfg,
    project_root=root,
    model_path=args.model,
    num_envs=args.num_envs,
    device=device,
    seed=args.seed,
  )
  iterations = args.iterations or cfg.ppo.finetune_iterations
  args.output_dir.mkdir(parents=True, exist_ok=True)
  session.reset()
  try:
    for iteration in range(1, iterations + 1):
      actor.eval()
      buffer, _, rollout = collect_deploy_rollout(session, actor, adaptation)
      actor.train()
      update = optimizer.update(buffer, use_estimated_latent=True)
      print(
        f"[finetune {iteration}/{iterations}] "
        f"reward={rollout.mean_meta_reward:.5f} "
        f"switch={rollout.switch_rate:.5f} "
        f"kl={update.approximate_kl:.6f}",
        flush=True,
      )
      if iteration % cfg.ppo.save_interval == 0 or iteration == iterations:
        checkpoint = make_checkpoint(
          stage="deployment_finetune",
          cfg=cfg,
          actor=actor,
          encoder=encoder,
          adaptation=adaptation,
          iteration=iteration,
          registry_sha256=session.registry_sha256,
          optimizer_state_dict=optimizer.optimizer.state_dict(),
          metrics={"rollout": vars(rollout), "ppo": vars(update)},
        )
        save_checkpoint_atomic(
          checkpoint, args.output_dir / f"finetune_{iteration:06d}.pt"
        )
        save_checkpoint_atomic(checkpoint, args.output_dir / "finetune_latest.pt")
  finally:
    session.close()


if __name__ == "__main__":
  main()
