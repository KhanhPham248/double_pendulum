"""Versioned registry and contract checks for frozen full-task experts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from double_pendulum.common import PolicyContract
from double_pendulum.export.manifest import PolicyManifest, load_manifest
from double_pendulum.runtime.onnx_policy import OnnxTorquePolicy


@dataclass(frozen=True)
class FullTaskExpertSpec:
  name: str
  algorithm: str
  run_dir: str
  model: str = "policy.onnx"
  policy_sha256: str = ""


@dataclass(frozen=True)
class FullTaskExpertRegistry:
  experts: tuple[FullTaskExpertSpec, ...]

  def __post_init__(self) -> None:
    if len(self.experts) < 2:
      raise ValueError("RMA-meta needs at least two full-task experts")
    names = tuple(expert.name for expert in self.experts)
    if any(not name for name in names) or len(set(names)) != len(names):
      raise ValueError("expert names must be non-empty and unique")

  @property
  def expert_names(self) -> tuple[str, ...]:
    return tuple(expert.name for expert in self.experts)


def load_expert_registry(path: str | Path) -> FullTaskExpertRegistry:
  registry_path = Path(path)
  payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
  if not isinstance(payload, dict) or int(payload.get("registry_version", 0)) != 1:
    raise ValueError("expert registry_version must be 1")
  raw_experts = payload.get("experts")
  if not isinstance(raw_experts, list) or len(raw_experts) < 2:
    raise ValueError("expert registry must contain at least two experts")
  return FullTaskExpertRegistry(
    experts=tuple(
      FullTaskExpertSpec(
        name=str(item["name"]),
        algorithm=str(item["algorithm"]),
        run_dir=str(item["run_dir"]),
        model=str(item.get("model", "policy.onnx")),
        policy_sha256=str(item.get("policy_sha256", "")),
      )
      for item in raw_experts
    )
  )


def load_frozen_experts(
  project_root: str | Path,
  registry_path: str | Path,
) -> tuple[FullTaskExpertRegistry, dict[str, OnnxTorquePolicy]]:
  root = Path(project_root).resolve()
  registry_file = Path(registry_path)
  if not registry_file.is_absolute():
    registry_file = root / registry_file
  registry, manifests = validate_expert_manifests(root, registry_file)
  policies: dict[str, OnnxTorquePolicy] = {}
  for expert in registry.experts:
    run_dir = Path(expert.run_dir)
    if not run_dir.is_absolute():
      run_dir = root / run_dir
    policy = OnnxTorquePolicy(run_dir)
    if policy.manifest != manifests[expert.name]:
      raise ValueError(f"expert {expert.name} changed while loading")
    policies[expert.name] = policy
  return registry, policies


def validate_expert_manifests(
  project_root: str | Path,
  registry_path: str | Path,
) -> tuple[FullTaskExpertRegistry, dict[str, PolicyManifest]]:
  """Validate hashes and contracts without requiring ONNX Runtime."""
  root = Path(project_root).resolve()
  registry_file = Path(registry_path)
  if not registry_file.is_absolute():
    registry_file = root / registry_file
  registry = load_expert_registry(registry_file)
  manifests: dict[str, PolicyManifest] = {}
  reference_contract: PolicyContract | None = None
  for expert in registry.experts:
    run_dir = Path(expert.run_dir)
    if not run_dir.is_absolute():
      run_dir = root / run_dir
    if expert.model != "policy.onnx":
      raise ValueError(
        "frozen expert model must be policy.onnx so its policy.yaml hash is checked"
      )
    model_path = run_dir / expert.model
    manifest_path = run_dir / "policy.yaml"
    if not model_path.is_file() or not manifest_path.is_file():
      raise FileNotFoundError(
        f"expert {expert.name} requires {model_path} and {manifest_path}"
      )
    manifest = load_manifest(manifest_path)
    if manifest.algorithm != expert.algorithm:
      raise ValueError(
        f"expert {expert.name} algorithm does not match policy manifest"
      )
    if expert.policy_sha256 and manifest.policy_sha256 != expert.policy_sha256:
      raise ValueError(f"expert {expert.name} policy hash differs from registry")
    if reference_contract is None:
      reference_contract = manifest.contract
    elif manifest.contract != reference_contract:
      raise ValueError("full-task experts do not share the same policy contract")
    manifests[expert.name] = manifest
  return registry, manifests
