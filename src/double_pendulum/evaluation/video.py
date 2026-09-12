"""Small video writer kept outside training and simulation logic."""

from __future__ import annotations

from pathlib import Path

import mediapy
import numpy as np


def write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
  if not frames:
    raise ValueError("cannot write an empty video")
  path.parent.mkdir(parents=True, exist_ok=True)
  mediapy.write_video(path, frames, fps=fps)
