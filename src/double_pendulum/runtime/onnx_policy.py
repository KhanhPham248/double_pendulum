"""Fail-closed ONNX Runtime policy loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from double_pendulum.export.manifest import PolicyManifest, load_manifest


class OnnxTorquePolicy:
  def __init__(self, run_dir: Path):
    try:
      import onnxruntime as ort
    except ImportError as error:
      raise RuntimeError("install onnxruntime to run exported policies") from error

    self.run_dir = Path(run_dir).expanduser().resolve()
    self.manifest: PolicyManifest = load_manifest(self.run_dir / "policy.yaml")
    self.session = ort.InferenceSession(
      str(self.run_dir / "policy.onnx"), providers=["CPUExecutionProvider"]
    )
    inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
      raise ValueError("torque policy must have one input and one output")
    self.input_name = inputs[0].name
    self.output_name = outputs[0].name
    if inputs[0].type != "tensor(float)" or outputs[0].type != "tensor(float)":
      raise ValueError("ONNX policy tensors must be float32")
    if inputs[0].shape[-1] != self.manifest.contract.observation_dim:
      raise ValueError("ONNX input does not match policy manifest")
    if outputs[0].shape[-1] != self.manifest.contract.action_dim:
      raise ValueError("ONNX output does not match policy manifest")

  def act(self, observation: np.ndarray) -> np.ndarray:
    values = np.asarray(observation, dtype=np.float32)
    if values.shape == (self.manifest.contract.observation_dim,):
      values = values[None, :]
    expected = (None, self.manifest.contract.observation_dim)
    if values.ndim != 2 or values.shape[1] != expected[1]:
      raise ValueError(f"observation must have shape [batch, {expected[1]}]")
    if not np.isfinite(values).all():
      raise FloatingPointError("non-finite policy observation")
    actions = self.session.run([self.output_name], {self.input_name: values})[0]
    if actions.shape != (values.shape[0], self.manifest.contract.action_dim):
      raise ValueError("ONNX returned an invalid action shape")
    if not np.isfinite(actions).all():
      raise FloatingPointError("ONNX returned non-finite actions")
    return np.clip(actions, -1.0, 1.0)
