"""Atomic native checkpoint helpers used by SAC and PPO."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import torch


def save_torch_checkpoint(state: dict[str, Any], path: Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_name(path.name + ".tmp")
  try:
    torch.save(state, temporary)
    os.replace(temporary, path)
  finally:
    temporary.unlink(missing_ok=True)


def update_latest_checkpoint(checkpoint: Path) -> Path:
  latest = checkpoint.parent / "latest.pt"
  temporary = checkpoint.parent / ".latest.pt.tmp"
  temporary.unlink(missing_ok=True)
  try:
    try:
      os.link(checkpoint, temporary)
    except OSError:
      shutil.copy2(checkpoint, temporary)
    os.replace(temporary, latest)
  finally:
    temporary.unlink(missing_ok=True)
  return latest
