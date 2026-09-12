"""Small explicit registry; algorithms never import a concrete task."""

from __future__ import annotations

from typing import Any, Protocol

from double_pendulum.common import EvaluationSpec

from .combined import CombinedTask

_TASKS = {"combined": CombinedTask}


class RegisteredTask(Protocol):
  name: str
  config: Any

  @property
  def evaluation(self) -> EvaluationSpec: ...

  def make_env_cfg(self, num_envs: int = 512) -> Any: ...


def list_tasks() -> tuple[str, ...]:
  return tuple(_TASKS)


def get_task(name: str) -> RegisteredTask:
  try:
    task_type = _TASKS[name]
  except KeyError as error:
    choices = ", ".join(list_tasks())
    raise ValueError(f"unknown task {name!r}; choose one of: {choices}") from error
  return task_type()
