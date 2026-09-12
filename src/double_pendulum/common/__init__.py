"""Physical model and policy contract shared by every task."""

from .contract import (
  DEFAULT_CONTRACT,
  DEFAULT_EVALUATION,
  EvaluationSpec,
  PolicyContract,
)
from .model import DoublePendulumModelCfg, build_double_pendulum_spec
from .rewards import (
  DEFAULT_COMBINED_REWARD,
  LEGACY_COMBINED_REWARD,
  CombinedRewardSpec,
)

__all__ = [
  "DEFAULT_CONTRACT",
  "DEFAULT_COMBINED_REWARD",
  "DEFAULT_EVALUATION",
  "LEGACY_COMBINED_REWARD",
  "CombinedRewardSpec",
  "DoublePendulumModelCfg",
  "EvaluationSpec",
  "PolicyContract",
  "build_double_pendulum_spec",
]
