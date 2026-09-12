from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from double_pendulum.common import LEGACY_COMBINED_REWARD
from double_pendulum.export.manifest import (
  PolicyManifest,
  load_manifest,
  write_manifest,
)
from double_pendulum.export.validation import ParityResult, export_atomically


def test_manifest_detects_policy_tampering(tmp_path: Path) -> None:
  policy = tmp_path / "policy.onnx"
  policy.write_bytes(b"valid-policy")
  write_manifest(
    tmp_path / "policy.yaml",
    PolicyManifest.create(
      task="combined",
      algorithm="sac",
      checkpoint="checkpoints/latest.pt",
      policy_path=policy,
    ),
  )
  loaded = load_manifest(tmp_path / "policy.yaml")
  assert loaded.algorithm == "sac"
  assert loaded.reward.formula_version == 3
  policy.write_bytes(b"tampered")
  with pytest.raises(ValueError, match="hash"):
    load_manifest(tmp_path / "policy.yaml")


def test_failed_export_keeps_previous_policy(tmp_path: Path) -> None:
  destination = tmp_path / "policy.onnx"
  destination.write_bytes(b"previous")

  def export_to(path: Path) -> None:
    path.write_bytes(b"invalid-new-policy")

  def reject(path: Path) -> ParityResult:
    raise ValueError("parity failed")

  with pytest.raises(ValueError, match="parity"):
    export_atomically(destination, export_to, reject)
  assert destination.read_bytes() == b"previous"
  assert not (tmp_path / "policy.onnx.tmp").exists()


def test_manifest_without_reward_metadata_loads_as_legacy(tmp_path: Path) -> None:
  policy = tmp_path / "policy.onnx"
  policy.write_bytes(b"legacy-policy")
  manifest = PolicyManifest.create(
    task="combined",
    algorithm="ppo",
    checkpoint="checkpoints/latest.pt",
    policy_path=policy,
  ).as_dict()
  manifest.pop("reward")
  (tmp_path / "policy.yaml").write_text(
    yaml.safe_dump(manifest, sort_keys=False),
    encoding="utf-8",
  )
  loaded = load_manifest(tmp_path / "policy.yaml")
  assert loaded.reward == LEGACY_COMBINED_REWARD
