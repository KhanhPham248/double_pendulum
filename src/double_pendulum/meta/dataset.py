"""History-to-latent dataset helpers with episode-level validation splits."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


DATASET_VERSION = 1


def save_adaptation_dataset(
  path: str | Path,
  *,
  histories: torch.Tensor,
  latents: torch.Tensor,
  factors: torch.Tensor,
  episode_ids: torch.Tensor,
  valid_steps: torch.Tensor,
) -> Path:
  if histories.ndim != 3:
    raise ValueError("histories must have shape [samples, time, observation]")
  samples = histories.shape[0]
  if latents.shape[0] != samples or factors.shape[0] != samples:
    raise ValueError("dataset sample counts differ")
  if episode_ids.shape != (samples,) or valid_steps.shape != (samples,):
    raise ValueError("episode_ids and valid_steps must have shape [samples]")
  if not all(
    torch.isfinite(value).all()
    for value in (histories, latents, factors)
  ):
    raise ValueError("adaptation dataset contains non-finite values")
  destination = Path(path)
  destination.parent.mkdir(parents=True, exist_ok=True)
  torch.save(
    {
      "dataset_version": DATASET_VERSION,
      "histories": histories.cpu(),
      "latents": latents.cpu(),
      "factors": factors.cpu(),
      "episode_ids": episode_ids.cpu(),
      "valid_steps": valid_steps.cpu(),
    },
    destination,
  )
  return destination


def load_adaptation_dataset(path: str | Path) -> dict[str, torch.Tensor | int]:
  payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
  if int(payload.get("dataset_version", 0)) != DATASET_VERSION:
    raise ValueError("unsupported adaptation dataset version")
  required = ("histories", "latents", "factors", "episode_ids", "valid_steps")
  if any(key not in payload for key in required):
    raise ValueError("adaptation dataset is missing required tensors")
  samples = payload["histories"].shape[0]
  if any(payload[key].shape[0] != samples for key in required):
    raise ValueError("adaptation dataset tensor counts differ")
  return payload


def episode_validation_mask(
  episode_ids: torch.Tensor,
  validation_fraction: float,
) -> torch.Tensor:
  if episode_ids.ndim != 1:
    raise ValueError("episode_ids must be rank one")
  if not 0.0 < validation_fraction < 1.0:
    raise ValueError("validation fraction must be in (0, 1)")
  unique = torch.unique(episode_ids, sorted=True)
  validation_count = max(1, round(len(unique) * validation_fraction))
  validation_ids = unique[-validation_count:]
  return torch.isin(episode_ids, validation_ids)
