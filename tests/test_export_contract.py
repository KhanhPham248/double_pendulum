from __future__ import annotations

from pathlib import Path

import pytest

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
  assert load_manifest(tmp_path / "policy.yaml").algorithm == "sac"
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
