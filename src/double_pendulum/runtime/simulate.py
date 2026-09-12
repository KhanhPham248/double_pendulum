"""Interactive or recorded Python MuJoCo simulation of one ONNX policy."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco

from double_pendulum.evaluation.metrics import EpisodeAccumulator
from double_pendulum.runtime import MujocoDoublePendulum, OnnxTorquePolicy


def simulate(
  *,
  run_dir: Path,
  model_path: Path,
  duration_s: float | None,
  seed: int,
  reset_mode: str,
  headless: bool,
  video_path: Path | None,
) -> dict[str, float | None]:
  policy = OnnxTorquePolicy(run_dir)
  evaluation = policy.manifest.evaluation
  duration_s = duration_s or evaluation.default_duration_s
  simulator = MujocoDoublePendulum(model_path, policy.manifest.contract)
  observation = simulator.reset(seed=seed, mode=reset_mode)
  accumulator = EpisodeAccumulator(
    control_dt=policy.manifest.contract.control_timestep_s,
    torque_limit_nm=policy.manifest.contract.torque_limit_nm,
    required_hold_s=evaluation.success_hold_s,
    angle_threshold_rad=evaluation.angle_threshold_rad,
    velocity_threshold_rad_s=evaluation.velocity_threshold_rad_s,
  )
  steps = round(duration_s / policy.manifest.contract.control_timestep_s)
  frames = []
  renderer = (
    mujoco.Renderer(simulator.model, height=720, width=1280)
    if video_path
    else None
  )

  def control_step() -> None:
    nonlocal observation
    action = float(policy.act(observation)[0, 0])
    observation = simulator.step(action)
    accumulator.add(simulator.qpos, simulator.qvel, action)
    if renderer is not None:
      renderer.update_scene(simulator.data, camera="overview")
      frames.append(renderer.render())

  try:
    if headless:
      for _ in range(steps):
        control_step()
    else:
      import mujoco.viewer

      with mujoco.viewer.launch_passive(simulator.model, simulator.data) as viewer:
        for _ in range(steps):
          started = time.monotonic()
          control_step()
          viewer.sync()
          remaining = policy.manifest.contract.control_timestep_s - (
            time.monotonic() - started
          )
          if remaining > 0.0:
            time.sleep(remaining)
    if video_path is not None:
      from double_pendulum.evaluation.video import write_video

      write_video(
        video_path,
        frames,
        fps=policy.manifest.contract.control_frequency_hz,
      )
  finally:
    if renderer is not None:
      renderer.close()
  return accumulator.result()


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--run", required=True, type=Path)
  parser.add_argument(
    "--model",
    type=Path,
    default=Path("assets/double_pendulum.xml"),
  )
  parser.add_argument("--duration", type=float)
  parser.add_argument("--seed", type=int, default=10_001)
  parser.add_argument(
    "--reset-mode", choices=("hanging", "upright", "random"), default="hanging"
  )
  parser.add_argument("--headless", action="store_true")
  parser.add_argument("--video", type=Path)
  args = parser.parse_args(argv)
  result = simulate(
    run_dir=args.run,
    model_path=args.model,
    duration_s=args.duration,
    seed=args.seed,
    reset_mode=args.reset_mode,
    headless=args.headless,
    video_path=args.video,
  )
  print(json.dumps(result, indent=2, sort_keys=True))
