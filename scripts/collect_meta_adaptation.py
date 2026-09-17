#!/usr/bin/env python3
"""Collect privileged teacher latents paired with deployable histories."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from double_pendulum.meta.checkpoint import load_checkpoint_models, sha256_file
from double_pendulum.meta.config import load_meta_rl_cfg
from double_pendulum.meta.dataset import save_adaptation_dataset
from double_pendulum.meta.session import MetaMujocoSession


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--config", type=Path, default=Path("configs/meta_rl/double_pendulum_rma_v1.yaml")
  )
  parser.add_argument("--teacher-checkpoint", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--model", type=Path, default=Path("assets/double_pendulum.xml"))
  parser.add_argument("--num-envs", type=int, default=16)
  parser.add_argument("--meta-steps", type=int, default=4000)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--seed", type=int, default=43)
  args = parser.parse_args(argv)
  if min(args.num_envs, args.meta_steps) <= 0:
    raise ValueError("num-envs and meta-steps must be positive")
  root = Path(__file__).resolve().parents[1]
  cfg = load_meta_rl_cfg(root / args.config)
  device = torch.device(args.device)
  payload, source_cfg, actor, encoder, adaptation = load_checkpoint_models(
    args.teacher_checkpoint,
    map_location=device,
    expected_stage="teacher",
    expected_expert_names=cfg.expert_names,
  )
  if source_cfg != cfg or adaptation is not None:
    raise ValueError("teacher checkpoint is incompatible with collection config")
  session = MetaMujocoSession(
    cfg=cfg,
    project_root=root,
    model_path=args.model,
    num_envs=args.num_envs,
    device=device,
    seed=args.seed,
  )
  histories: list[torch.Tensor] = []
  latents: list[torch.Tensor] = []
  factors: list[torch.Tensor] = []
  episode_ids: list[torch.Tensor] = []
  valid_steps: list[torch.Tensor] = []
  session.reset()
  actor.eval()
  encoder.eval()
  try:
    with torch.no_grad():
      for step in range(args.meta_steps):
        factor = session.privileged_factors()
        latent = encoder(factor)
        histories.append(session.history.buffer.detach().cpu())
        latents.append(latent.detach().cpu())
        factors.append(factor.detach().cpu())
        episode_ids.append(session.episode_ids.detach().cpu())
        valid_steps.append(session.history.valid_steps.detach().cpu())
        action, _, _, _ = actor.act(
          session.actor_observation,
          latent,
          session.selection.one_hot,
        )
        session.step_meta(action)
        if (step + 1) % 100 == 0:
          print(f"[collect {step + 1}/{args.meta_steps}]", flush=True)
  finally:
    session.close()
  if args.output.exists():
    raise FileExistsError(args.output)
  save_adaptation_dataset(
    args.output,
    histories=torch.cat(histories),
    latents=torch.cat(latents),
    factors=torch.cat(factors),
    episode_ids=torch.cat(episode_ids),
    valid_steps=torch.cat(valid_steps),
  )
  manifest = {
    "dataset_version": 1,
    "config": cfg.to_dict(),
    "teacher_checkpoint": str(args.teacher_checkpoint.resolve()),
    "teacher_checkpoint_sha256": sha256_file(args.teacher_checkpoint),
    "registry_sha256": payload["registry_sha256"],
    "samples": int(torch.cat(latents).shape[0]),
    "history_shape": list(torch.cat(histories).shape[1:]),
  }
  manifest_path = args.output.with_suffix(".yaml")
  manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
  print(f"ADAPTATION_DATASET={args.output.resolve()}")
  print(f"ADAPTATION_MANIFEST={manifest_path.resolve()}")


if __name__ == "__main__":
  main()
