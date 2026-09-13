from __future__ import annotations

import json
from pathlib import Path

from double_pendulum.algorithms.ppo import PPOTrainConfig, make_ppo_runner_cfg
from double_pendulum.training.common import WandbSession
from double_pendulum.training.monitoring import CheckpointMonitor


def _report(success: float, hold: float) -> dict[str, object]:
  return {
    "aggregate": {
      "success_rate": success,
      "longest_hold_s": hold,
      "upright_fraction": hold / 20.0,
      "episode_return": hold * 100.0,
      "mean_abs_torque_nm": 1.0,
    }
  }


def test_checkpoint_monitor_keeps_best_export(monkeypatch, tmp_path: Path) -> None:
  reports = iter((_report(1.0, 15.0), _report(0.0, 1.0)))

  def fake_evaluate(*args, output_path: Path, **kwargs):
    report = next(reports)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report), encoding="utf-8")
    return report

  monkeypatch.setattr(
    "double_pendulum.training.monitoring.evaluate_run",
    fake_evaluate,
  )
  (tmp_path / "policy.onnx").write_bytes(b"first")
  (tmp_path / "policy.yaml").write_text("first", encoding="utf-8")
  checkpoint = tmp_path / "checkpoints" / "step.pt"
  checkpoint.parent.mkdir()
  checkpoint.write_bytes(b"checkpoint")
  monitor = CheckpointMonitor(tmp_path, episodes=2)

  first = monitor.evaluate(1, checkpoint)
  (tmp_path / "policy.onnx").write_bytes(b"second")
  (tmp_path / "policy.yaml").write_text("second", encoding="utf-8")
  second = monitor.evaluate(2, checkpoint)

  assert first["eval/hanging/is_best"] == 1.0
  assert second["eval/hanging/is_best"] == 0.0
  assert (tmp_path / "best" / "policy.onnx").read_bytes() == b"first"
  assert (tmp_path / "best" / "checkpoint.txt").read_text().strip() == (
    "checkpoints/step.pt"
  )


def test_wandb_session_can_be_disabled_without_initializing(tmp_path: Path) -> None:
  session = WandbSession(
    tmp_path,
    enabled=False,
    project="test",
    entity=None,
    name=None,
    config={},
  )
  assert session.url is None
  session.finish()


def test_ppo_defaults_limit_late_training_updates() -> None:
  config = PPOTrainConfig()
  runner = make_ppo_runner_cfg(config)
  assert runner.algorithm.learning_rate == 3e-4
  assert runner.algorithm.schedule == "fixed"
  assert config.evaluation_episodes == 10
