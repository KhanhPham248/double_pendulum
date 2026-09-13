"""Deterministic checkpoint evaluation and best-policy retention."""

from __future__ import annotations

import json
import math
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from double_pendulum.evaluation import EvaluationConfig, evaluate_run

_NEGATIVE_SENTINEL = -1.0e12
_POSITIVE_SENTINEL = 1.0e12


def _score(report: dict[str, Any]) -> tuple[float, ...]:
  aggregate = report["aggregate"]

  def value(name: str, default: float = _NEGATIVE_SENTINEL) -> float:
    result = aggregate.get(name)
    if result is None:
      return default
    converted = float(result)
    return default if not math.isfinite(converted) else converted

  return (
    value("success_rate"),
    value("post_success_stable_fraction"),
    -value("post_success_escape_count", default=_POSITIVE_SENTINEL),
    -value("time_to_success_s", default=_POSITIVE_SENTINEL),
    -value("post_success_mean_angle_error_rad", default=_POSITIVE_SENTINEL),
    -value("post_success_mean_angular_velocity_rad_s", default=_POSITIVE_SENTINEL),
    -value("post_success_mean_abs_torque_nm", default=_POSITIVE_SENTINEL),
    -value("post_success_action_saturation_fraction", default=_POSITIVE_SENTINEL),
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
      self._publish_best(
        checkpoint=checkpoint,
        evaluation_path=output,
        report=report,
        score=score,
      )
      self.best_score = score
    metrics[f"eval/{self.reset_mode}/best_success_rate"] = self.best_score[0]
    return metrics

  def _publish_best(
    self,
    *,
    checkpoint: Path,
    evaluation_path: Path,
    report: dict[str, Any],
    score: tuple[float, ...],
  ) -> None:
    bundle_name = f".best-{uuid.uuid4().hex}"
    staged = self.run_dir / bundle_name
    staged.mkdir()
    try:
      _atomic_copy(self.run_dir / "policy.onnx", staged / "policy.onnx")
      _atomic_copy(self.run_dir / "policy.yaml", staged / "policy.yaml")
      _atomic_copy(evaluation_path, staged / "evaluation.json")
      (staged / "checkpoint.txt").write_text(
        str(checkpoint.relative_to(self.run_dir)) + "\n", encoding="utf-8"
      )
      selection = {
        "checkpoint": str(checkpoint.relative_to(self.run_dir)),
        "reset_mode": self.reset_mode,
        "episodes": self.episodes,
        "score": list(score),
        "aggregate": report["aggregate"],
      }
      (staged / "selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
      )

      published = self.run_dir / f"best-{uuid.uuid4().hex}"
      os.replace(staged, published)
      link = self.run_dir / ".best-link"
      link.unlink(missing_ok=True)
      link.symlink_to(published.name)
      if self.best_dir.is_symlink():
        os.replace(link, self.best_dir)
      elif self.best_dir.exists():
        legacy = self.run_dir / f"best-legacy-{uuid.uuid4().hex}"
        os.replace(self.best_dir, legacy)
        os.replace(link, self.best_dir)
      else:
        os.replace(link, self.best_dir)
    except Exception:
      shutil.rmtree(staged, ignore_errors=True)
      raise
