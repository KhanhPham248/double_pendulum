"""RSL-RL runner that maintains one validated latest ONNX policy."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from mjlab.rl import MjlabOnPolicyRunner

from double_pendulum.common import (
  DEFAULT_CONTRACT,
  CombinedRewardSpec,
  EvaluationSpec,
)
from double_pendulum.export.checkpoints import update_latest_checkpoint
from double_pendulum.export.manifest import PolicyManifest, write_manifest
from double_pendulum.export.validation import export_atomically, validate_onnx


class DoublePendulumPpoRunner(MjlabOnPolicyRunner):
  def __init__(
    self,
    *args,
    task_name: str,
    evaluation_spec: EvaluationSpec,
    reward_spec: CombinedRewardSpec,
    export_fail_fast: bool = False,
    **kwargs,
  ) -> None:
    self.task_name = task_name
    self.evaluation_spec = evaluation_spec
    self.reward_spec = reward_spec
    self.export_fail_fast = export_fail_fast
    self._last_saved_iteration = -1
    super().__init__(*args, **kwargs)

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    infos = super().load(path, load_cfg, strict, map_location)
    restores_iteration = load_cfg is None or load_cfg.get("iteration", False)
    if restores_iteration:
      # RSL-RL stores the iteration that has already completed. Continue from
      # the next number so the resumed run cannot overwrite that checkpoint.
      self.current_learning_iteration += 1
    self._last_saved_iteration = -1
    return infos

  def save(self, path: str, infos=None) -> None:
    if self.current_learning_iteration == self._last_saved_iteration:
      return
    run_dir = Path(self.logger.log_dir)
    checkpoint = (
      run_dir
      / "checkpoints"
      / f"iteration_{self.current_learning_iteration:08d}.pt"
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
      **(infos or {}),
      "double_pendulum": {
        "format_version": 1,
        "task": self.task_name,
        "algorithm": "ppo",
        "contract": DEFAULT_CONTRACT.as_dict(),
        "reward": self.reward_spec.as_dict(),
      },
    }
    temporary = checkpoint.with_name(checkpoint.name + ".tmp")
    try:
      super().save(str(temporary), metadata)
      os.replace(temporary, checkpoint)
    finally:
      temporary.unlink(missing_ok=True)
    update_latest_checkpoint(checkpoint)
    self._last_saved_iteration = self.current_learning_iteration
    try:
      self._export_latest(run_dir, checkpoint)
    except Exception as error:
      print(f"[WARN] PPO ONNX export failed; previous policy kept: {error}")
      if self.export_fail_fast:
        raise

  def _export_latest(self, run_dir: Path, checkpoint: Path) -> None:
    module = self.alg.get_policy().as_onnx(verbose=False).cpu().eval()
    generator = torch.Generator(device="cpu").manual_seed(23)
    samples = torch.randn(
      (32, DEFAULT_CONTRACT.observation_dim), generator=generator
    ).numpy().astype(np.float32)

    def export_to(path: Path) -> None:
      self.export_policy_to_onnx(str(path.parent), path.name)

    def native(values: np.ndarray) -> np.ndarray:
      with torch.no_grad():
        return module(torch.from_numpy(values)).numpy()

    result = export_atomically(
      run_dir / "policy.onnx",
      export_to,
      lambda path: validate_onnx(path, samples, native),
    )
    manifest = PolicyManifest.create(
      task=self.task_name,
      algorithm="ppo",
      checkpoint=str(checkpoint.relative_to(run_dir)),
      policy_path=run_dir / "policy.onnx",
      evaluation=self.evaluation_spec,
      reward=self.reward_spec,
    )
    write_manifest(run_dir / "policy.yaml", manifest)
    print(
      "PPO_ONNX_UPDATED "
      f"abs_error={result.max_abs_error:.3e} rel_error={result.max_rel_error:.3e}",
      flush=True,
    )
