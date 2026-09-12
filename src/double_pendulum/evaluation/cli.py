"""Command-line interface for common ONNX evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluator import EvaluationConfig, evaluate_run


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--run", required=True, type=Path)
  parser.add_argument(
    "--model",
    type=Path,
    default=Path("assets/double_pendulum.xml"),
  )
  parser.add_argument("--episodes", type=int, default=100)
  parser.add_argument("--duration", type=float)
  parser.add_argument("--seed", type=int, default=10_001)
  parser.add_argument(
    "--reset-mode", choices=("hanging", "upright", "random"), default="hanging"
  )
  parser.add_argument("--mass-scale", type=float, default=1.0)
  parser.add_argument("--damping-scale", type=float, default=1.0)
  parser.add_argument("--velocity-kick", type=float, default=0.0)
  parser.add_argument("--kick-time", type=float, default=5.0)
  args = parser.parse_args(argv)
  report = evaluate_run(
    args.run,
    args.model,
    EvaluationConfig(
      episodes=args.episodes,
      duration_s=args.duration,
      seed=args.seed,
      reset_mode=args.reset_mode,
      mass_scale=args.mass_scale,
      damping_scale=args.damping_scale,
      velocity_kick_rad_s=args.velocity_kick,
      kick_time_s=args.kick_time,
    ),
  )
  print(json.dumps(report["aggregate"], indent=2, sort_keys=True))
