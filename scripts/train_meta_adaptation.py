#!/usr/bin/env python3
"""Train the deployable GRU history-to-latent adaptation module."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from double_pendulum.meta.checkpoint import (
  load_checkpoint_models,
  make_checkpoint,
  save_checkpoint_atomic,
  sha256_file,
)
from double_pendulum.meta.config import load_meta_rl_cfg
from double_pendulum.meta.dataset import (
  episode_validation_mask,
  load_adaptation_dataset,
)
from double_pendulum.meta.models import AdaptationModule


def _normalizer(
  histories: torch.Tensor,
  valid_steps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  time = torch.arange(histories.shape[1])[None, :]
  valid = time >= (histories.shape[1] - valid_steps[:, None])
  values = histories[valid]
  mean = values.mean(dim=0)
  std = values.std(dim=0, unbiased=False).clamp_min(1.0e-6)
  return mean, std


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--config", type=Path, default=Path("configs/meta_rl/double_pendulum_rma_v1.yaml")
  )
  parser.add_argument("--teacher-checkpoint", type=Path, required=True)
  parser.add_argument("--dataset", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--epochs", type=int)
  parser.add_argument("--batch-size", type=int, default=256)
  parser.add_argument("--device", default="cpu")
  args = parser.parse_args(argv)
  root = Path(__file__).resolve().parents[1]
  cfg = load_meta_rl_cfg(root / args.config)
  device = torch.device(args.device)
  teacher_payload, source_cfg, actor, encoder, old_adaptation = (
    load_checkpoint_models(
      args.teacher_checkpoint,
      map_location=device,
      expected_stage="teacher",
      expected_expert_names=cfg.expert_names,
    )
  )
  if source_cfg != cfg or old_adaptation is not None:
    raise ValueError("teacher checkpoint is incompatible with adaptation config")
  data = load_adaptation_dataset(args.dataset)
  histories = data["histories"].float()
  latents = data["latents"].float()
  factors = data["factors"].float()
  episode_ids = data["episode_ids"].long()
  valid_steps = data["valid_steps"].long()
  if histories.shape[1:] != (cfg.model.history_steps, cfg.model.observation_dim):
    raise ValueError("adaptation history shape differs from config")
  if latents.shape[1] != cfg.model.latent_dim or factors.shape[1] != cfg.model.factor_dim:
    raise ValueError("adaptation target shape differs from config")
  validation = episode_validation_mask(episode_ids, 0.1)
  training = ~validation
  if not torch.any(training) or not torch.any(validation):
    raise ValueError("episode split produced an empty partition")

  adaptation = AdaptationModule(cfg.model).to(device)
  mean, std = _normalizer(histories[training], valid_steps[training])
  adaptation.set_observation_normalizer(mean.to(device), std.to(device))
  optimizer = torch.optim.AdamW(
    adaptation.parameters(),
    lr=3.0e-4,
    weight_decay=1.0e-4,
  )
  epochs = args.epochs or 100
  args.output_dir.mkdir(parents=True, exist_ok=True)
  train_data = torch.utils.data.TensorDataset(histories[training], latents[training])
  validation_data = torch.utils.data.TensorDataset(
    histories[validation], latents[validation]
  )
  train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size, shuffle=True
  )
  validation_loader = torch.utils.data.DataLoader(
    validation_data, batch_size=args.batch_size, shuffle=False
  )
  for epoch in range(1, epochs + 1):
    adaptation.train()
    train_loss = 0.0
    train_count = 0
    for history, target in train_loader:
      history, target = history.to(device), target.to(device)
      prediction, _ = adaptation(history)
      loss = nn.functional.mse_loss(prediction, target)
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      nn.utils.clip_grad_norm_(adaptation.parameters(), 1.0)
      optimizer.step()
      train_loss += loss.item() * len(target)
      train_count += len(target)
    adaptation.eval()
    validation_loss = 0.0
    validation_count = 0
    with torch.no_grad():
      for history, target in validation_loader:
        prediction, _ = adaptation(history.to(device))
        validation_loss += nn.functional.mse_loss(
          prediction, target.to(device), reduction="sum"
        ).item()
        validation_count += len(target)
    train_mean = train_loss / max(train_count, 1)
    validation_mean = validation_loss / max(validation_count * cfg.model.latent_dim, 1)
    if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
      print(
        f"[adaptation {epoch}/{epochs}] "
        f"train_mse={train_mean:.7f} val_mse={validation_mean:.7f}",
        flush=True,
      )
  payload = make_checkpoint(
    stage="adaptation",
    cfg=cfg,
    actor=actor,
    encoder=encoder,
    adaptation=adaptation,
    iteration=epochs,
    registry_sha256=teacher_payload["registry_sha256"],
    optimizer_state_dict=optimizer.state_dict(),
    metrics={"validation_mse": validation_mean},
  )
  path = save_checkpoint_atomic(payload, args.output_dir / "adaptation_latest.pt")
  print(f"ADAPTATION_CHECKPOINT={path.resolve()}")
  print(f"TEACHER_SHA256={sha256_file(args.teacher_checkpoint)}")


if __name__ == "__main__":
  main()
