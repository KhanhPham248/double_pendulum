"""ONNX contract/parity validation and atomic replacement."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from double_pendulum.common import DEFAULT_CONTRACT, PolicyContract


@dataclass(frozen=True)
class ParityResult:
  max_abs_error: float
  max_rel_error: float


def validate_onnx(
  path: Path,
  samples: np.ndarray,
  native_inference: Callable[[np.ndarray], np.ndarray],
  *,
  contract: PolicyContract = DEFAULT_CONTRACT,
  absolute_tolerance: float = 1e-5,
  relative_tolerance: float = 1e-4,
) -> ParityResult:
  try:
    import onnxruntime as ort
  except ImportError as error:
    raise RuntimeError(
      "onnxruntime is required to validate exported policies"
    ) from error

  if samples.dtype != np.float32 or samples.shape[1:] != (contract.observation_dim,):
    raise ValueError("parity samples must be float32 [batch, observation_dim]")
  session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
  inputs, outputs = session.get_inputs(), session.get_outputs()
  if len(inputs) != 1 or len(outputs) != 1:
    raise ValueError("torque policy must have exactly one input and one output")
  if inputs[0].type != "tensor(float)" or outputs[0].type != "tensor(float)":
    raise ValueError("policy input and output must be float32 tensors")
  if inputs[0].shape[-1] != contract.observation_dim:
    raise ValueError("ONNX observation dimension does not match policy contract")
  if outputs[0].shape[-1] != contract.action_dim:
    raise ValueError("ONNX action dimension does not match policy contract")

  actual = session.run([outputs[0].name], {inputs[0].name: samples})[0]
  expected = np.asarray(native_inference(samples), dtype=np.float32)
  if actual.shape != expected.shape:
    raise ValueError(
      f"native/ONNX output shapes differ: {expected.shape} != {actual.shape}"
    )
  absolute = np.abs(actual - expected)
  relative = absolute / np.maximum(np.abs(expected), 1e-6)
  result = ParityResult(float(absolute.max()), float(relative.max()))
  parity_failed = (
    result.max_abs_error > absolute_tolerance
    and result.max_rel_error > relative_tolerance
  )
  if parity_failed:
    raise ValueError(
      "ONNX parity failed: "
      f"abs={result.max_abs_error:.3e}, rel={result.max_rel_error:.3e}"
    )
  return result


def export_atomically(
  destination: Path,
  export_to: Callable[[Path], None],
  validate: Callable[[Path], ParityResult],
) -> ParityResult:
  destination.parent.mkdir(parents=True, exist_ok=True)
  temporary = destination.with_name(destination.name + ".tmp")
  try:
    export_to(temporary)
    result = validate(temporary)
    os.replace(temporary, destination)
    return result
  finally:
    temporary.unlink(missing_ok=True)
