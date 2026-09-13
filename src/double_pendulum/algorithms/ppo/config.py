"""PPO hyperparameters kept separate from task configuration."""

from __future__ import annotations

from dataclasses import dataclass

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@dataclass(frozen=True)
class PPOTrainConfig:
  task: str = "combined"
  device: str = "cuda:0"
  num_envs: int = 512
  max_iterations: int = 2_000
  save_interval: int = 50
  seed: int = 1
  run_dir: str | None = None
  resume: str | None = None
  export_fail_fast: bool = False
  evaluation_episodes: int = 20
  use_wandb: bool = False
  wandb_project: str = "double-pendulum"
  wandb_entity: str | None = None
  wandb_run_name: str | None = None

  def __post_init__(self) -> None:
    if min(self.num_envs, self.max_iterations, self.save_interval) <= 0:
      raise ValueError("environment, iteration and save counts must be positive")
    if self.evaluation_episodes < 0:
      raise ValueError("evaluation episodes cannot be negative")
    if self.use_wandb and not self.wandb_project:
      raise ValueError("W&B project cannot be empty")
    if self.run_dir is not None and self.resume is not None:
      raise ValueError("run_dir and resume cannot be used together")


def make_ppo_runner_cfg(config: PPOTrainConfig) -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    seed=config.seed,
    actor=RslRlModelCfg(
      hidden_dims=(128, 128, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(128, 128, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=3e-4,
      schedule="fixed",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    num_steps_per_env=24,
    max_iterations=config.max_iterations,
    save_interval=config.save_interval,
    experiment_name=f"double_pendulum_{config.task}",
    run_name=config.wandb_run_name or "",
    logger="tensorboard",
    wandb_project=config.wandb_project,
    upload_model=False,
    clip_actions=1.0,
  )
