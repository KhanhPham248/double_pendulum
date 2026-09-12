"""Small logging, run-directory and checkpoint helpers shared by trainers."""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
import torch


class MetricsLogger:
  def __init__(self, run_dir: Path):
    from torch.utils.tensorboard import SummaryWriter

    self._jsonl = (run_dir / "metrics.jsonl").open("a", encoding="utf-8")
    self._tensorboard = SummaryWriter(run_dir / "tensorboard")

  def write(self, step: int, values: dict[str, float | int]) -> None:
    record = {"step": step, **values}
    self._jsonl.write(json.dumps(record, sort_keys=True) + "\n")
    self._jsonl.flush()
    for name, value in values.items():
      self._tensorboard.add_scalar(name, value, step)

  def close(self) -> None:
    self._jsonl.close()
    self._tensorboard.close()


def prepare_run_dir(
  *,
  task: str,
  algorithm: str,
  requested: str | None = None,
  resume: str | None = None,
) -> Path:
  if resume is not None:
    checkpoint = Path(resume).expanduser().resolve()
    if checkpoint.parent.name != "checkpoints":
      raise ValueError("resume checkpoint must be inside a checkpoints directory")
    run_dir = checkpoint.parent.parent
    if not run_dir.is_dir():
      raise FileNotFoundError(run_dir)
    return run_dir
  if requested is not None:
    run_dir = Path(requested).expanduser().resolve()
  else:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path("runs") / task / algorithm / stamp
  run_dir.mkdir(parents=True, exist_ok=False)
  (run_dir / "checkpoints").mkdir()
  (run_dir / "videos").mkdir()
  return run_dir.resolve()


def capture_rng_state() -> dict[str, Any]:
  state: dict[str, Any] = {
    "python": random.getstate(),
    "torch_cpu": torch.get_rng_state(),
  }
  if torch.cuda.is_available():
    state["torch_cuda"] = torch.cuda.get_rng_state_all()
  return state


def restore_rng_state(state: dict[str, Any]) -> None:
  random.setstate(state["python"])
  torch.set_rng_state(state["torch_cpu"].cpu())
  if "torch_cuda" in state and torch.cuda.is_available():
    torch.cuda.set_rng_state_all(state["torch_cuda"])
