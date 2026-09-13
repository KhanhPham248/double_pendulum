from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from double_pendulum.common import DEFAULT_CONTRACT
from double_pendulum.common.observations import policy_observation
from double_pendulum.evaluation.metrics import EpisodeAccumulator
from double_pendulum.runtime.mujoco_sim import MujocoDoublePendulum


def test_numpy_runtime_observation_matches_contract() -> None:
  observation = policy_observation(
    np.array([0.0, math.pi / 2]),
    np.array([1.5, -2.0]),
    np,
  )
  assert np.allclose(observation, (0.0, 1.0, 1.0, 0.0, 1.5, -2.0))


def test_runtime_steps_at_declared_control_rate() -> None:
  simulator = MujocoDoublePendulum(
    model_path=Path("assets/double_pendulum.xml"),
    contract=DEFAULT_CONTRACT,
  )
  simulator.reset(seed=1)
  initial_time = simulator.data.time
  simulator.step(0.0)
  assert math.isclose(
    simulator.data.time - initial_time,
    DEFAULT_CONTRACT.control_timestep_s,
    abs_tol=1e-12,
  )


def test_five_second_hold_is_success() -> None:
  accumulator = EpisodeAccumulator(
    control_dt=0.02,
    torque_limit_nm=6.0,
    required_hold_s=5.0,
    angle_threshold_rad=0.21,
    velocity_threshold_rad_s=1.0,
  )
  for _ in range(250):
    accumulator.add(np.array([math.pi, 0.0]), np.zeros(2), 0.0)
  result = accumulator.result()
  assert result["success"] == 1.0
  assert result["longest_hold_s"] == 5.0


def test_post_success_metrics_start_after_required_hold() -> None:
  accumulator = EpisodeAccumulator(
    control_dt=0.02,
    torque_limit_nm=6.0,
    required_hold_s=5.0,
    angle_threshold_rad=0.21,
    velocity_threshold_rad_s=1.0,
  )
  for _ in range(250):
    accumulator.add(np.array([math.pi, 0.0]), np.zeros(2), 0.0)
  for _ in range(50):
    accumulator.add(np.zeros(2), np.zeros(2), 0.0)

  result = accumulator.result()
  assert result["success"] == 1.0
  assert result["time_to_success_s"] == 5.0
  assert result["post_success_duration_s"] == 1.0
  assert math.isclose(result["post_success_stable_fraction"], 0.0)
  assert result["post_success_escape_count"] == 1.0
