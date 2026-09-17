#!/usr/bin/env python3
"""Export the trained RMA-meta adapter and selector to ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from double_pendulum.meta.checkpoint import load_checkpoint_models, sha256_file
from double_pendulum.meta.config import load_meta_rl_cfg
from double_pendulum.meta.models import DeployAdapter, DeployMetaSelector


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--config", type=Path, default=Path("configs/meta_rl/double_pendulum_rma_v1.yaml")
  )
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  args = parser.parse_args(argv)
  root = Path(__file__).resolve().parents[1]
  cfg = load_meta_rl_cfg(root / args.config)
  payload, source_cfg, actor, encoder, adaptation = load_checkpoint_models(
    args.checkpoint,
    map_location="cpu",
    expected_expert_names=cfg.expert_names,
  )
  if source_cfg != cfg or adaptation is None:
    raise ValueError("checkpoint has no deployable adaptation module")
  adaptation.eval()
  actor.eval()
  output_dir = args.output_dir.resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  adapter_path = output_dir / "adapter.onnx"
  selector_path = output_dir / "selector.onnx"
  adapter = DeployAdapter(adaptation).eval()
  selector = DeployMetaSelector(actor).eval()
  history = torch.zeros(1, cfg.model.history_steps, cfg.model.observation_dim)
  observation = torch.zeros(1, cfg.model.observation_dim)
  latent = torch.zeros(1, cfg.model.latent_dim)
  previous = torch.zeros(1, len(cfg.expert_names))
  with torch.no_grad():
    torch.onnx.export(
      adapter,
      history,
      adapter_path,
      input_names=["history"],
      output_names=["latent"],
      dynamic_axes={"history": {0: "batch"}, "latent": {0: "batch"}},
      opset_version=17,
    )
    torch.onnx.export(
      selector,
      (observation, latent, previous),
      selector_path,
      input_names=["observation", "latent", "previous_expert"],
      output_names=["probabilities"],
      dynamic_axes={
        "observation": {0: "batch"},
        "latent": {0: "batch"},
        "previous_expert": {0: "batch"},
        "probabilities": {0: "batch"},
      },
      opset_version=17,
    )
  manifest = {
    "format_version": 1,
    "stage": payload["stage"],
    "checkpoint": str(args.checkpoint.resolve()),
    "checkpoint_sha256": sha256_file(args.checkpoint),
    "registry_sha256": payload["registry_sha256"],
    "expert_names": list(cfg.expert_names),
    "class_names": list(cfg.class_names),
    "observation_dim": cfg.model.observation_dim,
    "latent_dim": cfg.model.latent_dim,
    "history_shape": [cfg.model.history_steps, cfg.model.observation_dim],
    "adapter": adapter_path.name,
    "selector": selector_path.name,
  }
  manifest_path = output_dir / "meta_manifest.yaml"
  manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
  print(f"ADAPTER_ONNX={adapter_path}")
  print(f"SELECTOR_ONNX={selector_path}")
  print(f"META_MANIFEST={manifest_path}")


if __name__ == "__main__":
  main()
