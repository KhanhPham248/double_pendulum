"""Deterministic checkpoint evaluation and best-policy retention."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from double_pendulum.evaluation import EvaluationConfig, evaluate_run


def _score(report: dict[str, Any]) -> tuple[float, ...]:
  aggregate = report["aggregate"]

  def value(name: str, default: float = float("-inf")) -> float:
    result = aggregate.get(name)
    return default if result is None else float(result)

  return (
    value("success_rate"),
    value("longest_hold_s"),
    value("upright_fraction"),
    value("episode_return"),
    -value("mean_abs_torque_nm", default=float("inf")),
  )


def _atomic_copy(source: Path, destination: Path) -> None:
  temporary = destination.with_name(destination.name + ".tmp")
  try:
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)
  finally:
    temporary.unlink(missing_ok=True)


class CheckpointMonitor:
  """Evaluate exported checkpoints and preserve the strongest policy."""

  def __init__(
    self,
    run_dir: Path,
    *,
    episodes: int,
    reset_mode: str = "hanging",
  ) -> None:
    if episodes < 0:
      raise ValueError("evaluation episodes cannot be negative")
    self.run_dir = Path(run_dir)
    self.episodes = episodes
    self.reset_mode = reset_mode
    self.model_path = Path(__file__).resolve().parents[3] / "assets/double_pendulum.xml"
    self.best_dir = self.run_dir / "best"
    best_report = self.best_dir / "evaluation.json"
    self.best_score: tuple[float, ...] | None = None
    if best_report.is_file():
      with best_report.open(encoding="utf-8") as source:
        self.best_score = _score(json.load(source))

  def evaluate(self, step: int, checkpoint: Path) -> dict[str, float]:
    if self.episodes == 0:
      return {}
    output = self.run_dir / "evaluations" / f"step_{step:012d}.json"
    report = evaluate_run(
      self.run_dir,
      self.model_path,
      EvaluationConfig(
        episodes=self.episodes,
        reset_mode=self.reset_mode,
      ),
      output_path=output,
    )
    aggregate = report["aggregate"]
    metrics = {
      f"eval/{self.reset_mode}/{name}": float(value)
      for name, value in aggregate.items()
      if value is not None
    }
    score = _score(report)
    is_best = self.best_score is None or score > self.best_score
    metrics[f"eval/{self.reset_mode}/is_best"] = float(is_best)
    if is_best:
      self.best_score = score
      self.best_dir.mkdir(parents=True, exist_ok=True)
      _atomic_copy(self.run_dir / "policy.onnx", self.best_dir / "policy.onnx")
      _atomic_copy(self.run_dir / "policy.yaml", self.best_dir / "policy.yaml")
      _atomic_copy(output, self.best_dir / "evaluation.json")
      reference = self.best_dir / "checkpoint.txt"
      temporary = reference.with_name(reference.name + ".tmp")
      try:
        temporary.write_text(
          str(checkpoint.relative_to(self.run_dir)) + "\n",
          encoding="utf-8",
        )
        os.replace(temporary, reference)
      finally:
        temporary.unlink(missing_ok=True)
    metrics[f"eval/{self.reset_mode}/best_success_rate"] = self.best_score[0]
    return metrics
