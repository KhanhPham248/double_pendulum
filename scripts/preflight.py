#!/usr/bin/env python3
"""Check that this standalone project has a usable mJLab training runtime."""

from __future__ import annotations

import importlib
import importlib.metadata
import sys


def main() -> int:
  required = (
    "torch",
    "mujoco",
    "mujoco_warp",
    "warp",
    "rsl_rl",
    "onnx",
    "onnxruntime",
    "yaml",
    "tensorboard",
    "tyro",
    "mediapy",
  )
  missing: list[str] = []
  for module_name in required:
    try:
      module = importlib.import_module(module_name)
    except Exception as error:  # Import errors should be reported, not hidden.
      missing.append(f"{module_name}: {error}")
    else:
      print(f"OK {module_name}={getattr(module, '__version__', 'installed')}")
  if missing:
    print("FAIL missing or broken dependencies:")
    print("\n".join(missing))
    return 1

  rsl_version = importlib.metadata.version("rsl-rl-lib")
  if rsl_version != "5.0.1":
    print(f"FAIL rsl-rl-lib={rsl_version}; mJLab 1.2.0 requires 5.0.1")
    return 1

  import torch

  if not torch.cuda.is_available():
    print("FAIL CUDA is unavailable; mJLab training needs an NVIDIA CUDA runtime")
    return 1
  index = torch.cuda.current_device()
  properties = torch.cuda.get_device_properties(index)
  free_bytes, total_bytes = torch.cuda.mem_get_info(index)
  print(
    "OK cuda="
    f"{index} name={properties.name!r} "
    f"capability={properties.major}.{properties.minor} "
    f"free_gib={free_bytes / 2**30:.2f} total_gib={total_bytes / 2**30:.2f}"
  )
  if total_bytes < 4 * 2**30:
    print("WARN less than 4 GiB VRAM: start train with --num-envs 64")
  print("PREFLIGHT=PASS")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
