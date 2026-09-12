"""Portable, fail-closed policy metadata for Python MuJoCo runtime."""

from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from double_pendulum.common import (
  DEFAULT_COMBINED_REWARD,
  DEFAULT_CONTRACT,
  DEFAULT_EVALUATION,
  LEGACY_COMBINED_REWARD,
  CombinedRewardSpec,
  EvaluationSpec,
  PolicyContract,
)


@dataclass(frozen=True)
class PolicyManifest:
  task: str
  algorithm: str
  checkpoint: str
  policy_sha256: str
  contract: PolicyContract = DEFAULT_CONTRACT
  evaluation: EvaluationSpec = DEFAULT_EVALUATION
  reward: CombinedRewardSpec = DEFAULT_COMBINED_REWARD
  format_version: int = 1
  policy_type: str = "torque"
  created_at_utc: str = ""

  def __post_init__(self) -> None:
    if self.format_version != 1 or self.policy_type != "torque":
      raise ValueError("unsupported policy manifest")
    if not self.task or not self.algorithm:
      raise ValueError("task and algorithm are required")
    if Path(self.checkpoint).is_absolute():
      raise ValueError("checkpoint path in manifest must be relative")
    if len(self.policy_sha256) != 64:
      raise ValueError("policy_sha256 must contain a SHA-256 digest")

  def as_dict(self) -> dict[str, Any]:
    values = asdict(self)
    values["contract"] = self.contract.as_dict()
    values["reward"] = self.reward.as_dict()
    return values

  @classmethod
  def create(
    cls,
    *,
    task: str,
    algorithm: str,
    checkpoint: str,
    policy_path: Path,
    contract: PolicyContract = DEFAULT_CONTRACT,
    evaluation: EvaluationSpec = DEFAULT_EVALUATION,
    reward: CombinedRewardSpec = DEFAULT_COMBINED_REWARD,
  ) -> "PolicyManifest":
    timestamp = datetime.now(timezone.utc).isoformat()
    return cls(
      task=task,
      algorithm=algorithm,
      checkpoint=checkpoint,
      policy_sha256=file_sha256(policy_path),
      contract=contract,
      evaluation=evaluation,
      reward=reward,
      created_at_utc=timestamp,
    )

  @classmethod
  def from_dict(cls, values: dict[str, Any]) -> "PolicyManifest":
    data = dict(values)
    data["contract"] = PolicyContract.from_dict(data["contract"])
    data["evaluation"] = EvaluationSpec(**data["evaluation"])
    if "reward" in data:
      data["reward"] = CombinedRewardSpec(**data["reward"])
    else:
      data["reward"] = LEGACY_COMBINED_REWARD
    return cls(**data)


def file_sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def write_manifest(path: Path, manifest: PolicyManifest) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_name(path.name + ".tmp")
  try:
    with temporary.open("w", encoding="utf-8") as output:
      yaml.safe_dump(manifest.as_dict(), output, sort_keys=False)
    os.replace(temporary, path)
  finally:
    temporary.unlink(missing_ok=True)


def load_manifest(path: Path, *, verify_policy: bool = True) -> PolicyManifest:
  with path.open(encoding="utf-8") as source:
    values = yaml.safe_load(source)
  if not isinstance(values, dict):
    raise ValueError("policy manifest must contain a mapping")
  manifest = PolicyManifest.from_dict(values)
  if verify_policy:
    policy_path = path.parent / "policy.onnx"
    if file_sha256(policy_path) != manifest.policy_sha256:
      raise ValueError("policy.onnx hash does not match policy.yaml")
  return manifest
