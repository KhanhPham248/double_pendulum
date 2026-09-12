"""Deterministic SAC Actor export with normalization embedded in ONNX."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn

from double_pendulum.algorithms.sac import SACAgent
from double_pendulum.common import DEFAULT_CONTRACT, DEFAULT_EVALUATION, EvaluationSpec

from .manifest import PolicyManifest, write_manifest
from .validation import ParityResult, export_atomically, validate_onnx


class SacInferenceModule(nn.Module):
  def __init__(self, agent: SACAgent):
    super().__init__()
    self.normalizer = copy.deepcopy(agent.normalizer)
    self.actor = copy.deepcopy(agent.actor)
    self.normalize_observations = agent.config.normalize_observations

  def forward(self, observations: torch.Tensor) -> torch.Tensor:
    if self.normalize_observations:
      observations = self.normalizer(observations)
    return self.actor.sample(observations, deterministic=True)[0]


def export_sac_policy(
  agent: SACAgent,
  run_dir: Path,
  *,
  task: str,
  checkpoint: str,
  evaluation: EvaluationSpec = DEFAULT_EVALUATION,
) -> ParityResult:
  module = SacInferenceModule(agent).cpu().eval()
  generator = torch.Generator(device="cpu").manual_seed(17)
  samples = torch.randn(
    (32, DEFAULT_CONTRACT.observation_dim), generator=generator
  ).numpy().astype(np.float32)
  destination = run_dir / "policy.onnx"

  def export_to(path: Path) -> None:
    dummy = torch.zeros((1, DEFAULT_CONTRACT.observation_dim), dtype=torch.float32)
    torch.onnx.export(
      module,
      dummy,
      str(path),
      opset_version=18,
      input_names=["observations"],
      output_names=["actions"],
      dynamic_axes={"observations": {0: "batch"}, "actions": {0: "batch"}},
      dynamo=False,
    )

  def native(values: np.ndarray) -> np.ndarray:
    with torch.no_grad():
      return module(torch.from_numpy(values)).numpy()

  result = export_atomically(
    destination,
    export_to,
    lambda path: validate_onnx(path, samples, native),
  )
  manifest = PolicyManifest.create(
    task=task,
    algorithm="sac",
    checkpoint=checkpoint,
    policy_path=destination,
    evaluation=evaluation,
  )
  write_manifest(run_dir / "policy.yaml", manifest)
  return result
