"""Transition-based update scheduling for vectorized SAC collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class UpdateBudget:
  ratio: float
  remainder: float = 0.0
  eligible_transitions: int = 0
  total_updates: int = 0

  def __post_init__(self) -> None:
    if self.ratio <= 0.0:
      raise ValueError("update-to-data ratio must be positive")
    if not 0.0 <= self.remainder < 1.0:
      raise ValueError("update budget remainder must lie in [0, 1)")
    if min(self.eligible_transitions, self.total_updates) < 0:
      raise ValueError("update counters must be non-negative")

  def add(self, transitions: int) -> int:
    if transitions <= 0:
      raise ValueError("collected transitions must be positive")
    self.eligible_transitions += transitions
    available = self.remainder + transitions * self.ratio
    updates = int(available)
    self.remainder = available - updates
    self.total_updates += updates
    return updates

  @property
  def actual_ratio(self) -> float:
    return self.total_updates / max(self.eligible_transitions, 1)

  def state_dict(self) -> dict[str, float | int]:
    return {
      "ratio": self.ratio,
      "remainder": self.remainder,
      "eligible_transitions": self.eligible_transitions,
      "total_updates": self.total_updates,
    }

  @classmethod
  def from_state_dict(
    cls,
    state: dict[str, Any],
    *,
    expected_ratio: float,
  ) -> "UpdateBudget":
    ratio = float(state["ratio"])
    if ratio != expected_ratio:
      raise ValueError(
        f"checkpoint UTD ratio {ratio} differs from configured {expected_ratio}"
      )
    return cls(
      ratio=ratio,
      remainder=float(state["remainder"]),
      eligible_transitions=int(state["eligible_transitions"]),
      total_updates=int(state["total_updates"]),
    )
