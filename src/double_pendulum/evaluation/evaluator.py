"""Evaluate SAC or PPO through the same ONNX/MuJoCo path."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from double_pendulum.runtime import MujocoDoublePendulum, OnnxTorquePolicy

from .metrics import EpisodeAccumulator


@dataclass(frozen=True)
class EvaluationConfig:
  episodes: int = 100
  duration_s: float | None = None
  seed: int = 10_001
  reset_mode: str = "hanging"
  mass_scale: float = 1.0
  damping_scale: float = 1.0
  velocity_kick_rad_s: float = 0.0
  kick_time_s: float = 5.0

  def __post_init__(self) -> None:
    if self.episodes <= 0:
      raise ValueError("episodes must be positive")
    if self.duration_s is not None and self.duration_s <= 0.0:
      raise ValueError("duration must be positive")


def evaluate_run(
  run_dir: Path,
  model_path: Path,
  config: EvaluationConfig,
  *,
  output_path: Path | None = None,
) -> dict[str, object]:
  policy = OnnxTorquePolicy(run_dir)
  contract = policy.manifest.contract
  evaluation = policy.manifest.evaluation
  duration_s = config.duration_s or evaluation.default_duration_s
  steps = round(duration_s / contract.control_timestep_s)
  kick_step = round(config.kick_time_s / contract.control_timestep_s)
  episodes: list[dict[str, float | None]] = []

  for episode in range(config.episodes):
    simulator = MujocoDoublePendulum(
      model_path,
      contract,
      mass_scale=config.mass_scale,
      damping_scale=config.damping_scale,
    )
    observation = simulator.reset(seed=config.seed + episode, mode=config.reset_mode)
    accumulator = EpisodeAccumulator(
      control_dt=contract.control_timestep_s,
      torque_limit_nm=contract.torque_limit_nm,
      required_hold_s=evaluation.success_hold_s,
      angle_threshold_rad=evaluation.angle_threshold_rad,
      velocity_threshold_rad_s=evaluation.velocity_threshold_rad_s,
      reward_spec=policy.manifest.reward,
    )
    for step in range(steps):
      if config.velocity_kick_rad_s != 0.0 and step == kick_step:
        simulator.apply_velocity_kick(config.velocity_kick_rad_s)
        observation = simulator.observation()
      action = float(policy.act(observation)[0, 0])
      observation = simulator.step(action)
      accumulator.add(simulator.qpos, simulator.qvel, action)
    episodes.append(accumulator.result())

  aggregate: dict[str, float | None] = {}
  for name in episodes[0]:
    values = [episode[name] for episode in episodes if episode[name] is not None]
    aggregate[name if name != "success" else "success_rate"] = (
      float(np.mean(values)) if values else None
    )
  report: dict[str, object] = {
    "policy": {
      "task": policy.manifest.task,
      "algorithm": policy.manifest.algorithm,
      "sha256": policy.manifest.policy_sha256,
    },
    "config": {**asdict(config), "duration_s": duration_s},
    "aggregate": aggregate,
    "episodes": episodes,
  }
  output = output_path or Path(run_dir) / "evaluation.json"
  output.parent.mkdir(parents=True, exist_ok=True)
  with output.open("w", encoding="utf-8") as destination:
    json.dump(report, destination, indent=2, sort_keys=True)
  return report
