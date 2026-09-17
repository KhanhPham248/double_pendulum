"""RMA-meta components for selecting between full-task pendulum policies."""

from .config import (
  HOLD_NAME,
  MetaRlCfg,
  MetaRlModelCfg,
  MetaRlPpoCfg,
  MetaRlRuntimeCfg,
  load_meta_rl_cfg,
)
from .factors import FACTOR_NAMES, build_privileged_factors
from .history import ObservationHistory
from .models import (
  AdaptationModule,
  DeployAdapter,
  DeployMetaSelector,
  EnvironmentFactorEncoder,
  MetaActorCritic,
)
from .runtime import (
  ExpertSelectionState,
  TorqueCrossfader,
  confidence_gated_action,
)
from .ppo import MetaPpoOptimizer, MetaRolloutBuffer, PpoUpdateMetrics

__all__ = [
  "AdaptationModule",
  "DeployAdapter",
  "DeployMetaSelector",
  "EnvironmentFactorEncoder",
  "ExpertSelectionState",
  "FACTOR_NAMES",
  "HOLD_NAME",
  "MetaActorCritic",
  "MetaPpoOptimizer",
  "MetaRolloutBuffer",
  "MetaRlCfg",
  "MetaRlModelCfg",
  "MetaRlPpoCfg",
  "MetaRlRuntimeCfg",
  "ObservationHistory",
  "PpoUpdateMetrics",
  "TorqueCrossfader",
  "build_privileged_factors",
  "confidence_gated_action",
  "load_meta_rl_cfg",
]
