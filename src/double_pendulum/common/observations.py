"""Backend-agnostic observation math used by mJLab and NumPy runtime."""

from __future__ import annotations

from typing import Any


def policy_observation(qpos: Any, qvel: Any, array_api: Any) -> Any:
  """Return `[sin(q1), cos(q1), sin(q2), cos(q2), qdot1, qdot2]`."""

  return array_api.stack(
    (
      array_api.sin(qpos[..., 0]),
      array_api.cos(qpos[..., 0]),
      array_api.sin(qpos[..., 1]),
      array_api.cos(qpos[..., 1]),
      qvel[..., 0],
      qvel[..., 1],
    ),
    -1,
  )


def wrapped_angle_error(angle: Any, target: float, array_api: Any) -> Any:
  difference = angle - target
  return array_api.arctan2(array_api.sin(difference), array_api.cos(difference))
