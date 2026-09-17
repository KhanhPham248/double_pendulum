#!/usr/bin/env python3
"""Validate the selected full-task experts and RMA-meta configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

from double_pendulum.meta.config import load_meta_rl_cfg
from double_pendulum.meta.expert_registry import (
  load_frozen_experts,
  validate_expert_manifests,
)


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--config",
    type=Path,
    default=Path("configs/meta_rl/double_pendulum_rma_v1.yaml"),
  )
  parser.add_argument("--project-root", type=Path, default=Path("."))
  parser.add_argument(
    "--contract-only",
    action="store_true",
    help="validate hashes/contracts without importing ONNX Runtime",
  )
  args = parser.parse_args(argv)

  project_root = args.project_root.resolve()
  cfg = load_meta_rl_cfg(project_root / args.config)
  if cfg.model.observation_dim != 6 or cfg.model.latent_dim != 4:
    raise ValueError("unexpected double-pendulum meta model dimensions")
  if args.contract_only:
    registry, manifests = validate_expert_manifests(
      project_root, project_root / cfg.expert_registry
    )
    policies = {}
  else:
    registry, policies = load_frozen_experts(
      project_root,
      project_root / cfg.expert_registry,
    )
  if registry.expert_names != cfg.expert_names:
    raise ValueError("config expert order differs from registry expert order")
  for name in registry.expert_names:
    manifest = manifests[name] if args.contract_only else policies[name].manifest
    print(
      f"OK {name} algorithm={manifest.algorithm} "
      f"sha256={manifest.policy_sha256} "
      f"obs={manifest.contract.observation_dim} "
      f"action={manifest.contract.action_dim}"
    )
  print(
    f"OK classes={','.join(cfg.class_names)} "
    f"low_level_hz={cfg.runtime.low_level_hz:g} "
    f"meta_hz={cfg.runtime.meta_hz:g} "
    f"history={cfg.model.history_steps}x{cfg.model.observation_dim}"
  )
  print("META_CONTRACT=PASS" if args.contract_only else "META_PREFLIGHT=PASS")


if __name__ == "__main__":
  main()
