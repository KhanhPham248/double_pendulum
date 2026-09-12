"""mJLab configuration for one policy that learns swing-up and balancing."""

from __future__ import annotations

from dataclasses import dataclass

from mjlab.actuator import BuiltinMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointEffortActionCfg
from mjlab.managers import (
  EventTermCfg,
  ObservationGroupCfg,
  ObservationTermCfg,
  RewardTermCfg,
  TerminationTermCfg,
)
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig

from double_pendulum.common import (
  DEFAULT_CONTRACT,
  DoublePendulumModelCfg,
  EvaluationSpec,
  build_double_pendulum_spec,
)

from . import mdp


@dataclass(frozen=True)
class CombinedTaskCfg:
  episode_length_s: float = 20.0
  success_hold_s: float = 5.0
  stable_angle_rad: float = 0.21
  stable_velocity_rad_s: float = 1.0
  near_upright_probability: float = 0.20
  hanging_probability: float = 0.35
  near_upright_angle_rad: float = 0.20
  hanging_angle_rad: float = 0.25
  random_velocity_rad_s: float = 1.0

  def __post_init__(self) -> None:
    if min(self.episode_length_s, self.success_hold_s) <= 0.0:
      raise ValueError("episode and hold durations must be positive")
    if self.success_hold_s >= self.episode_length_s:
      raise ValueError("success hold must be shorter than the episode")


@dataclass(frozen=True)
class CombinedTask:
  name: str = "combined"
  config: CombinedTaskCfg = CombinedTaskCfg()

  @property
  def evaluation(self) -> EvaluationSpec:
    return EvaluationSpec(
      angle_threshold_rad=self.config.stable_angle_rad,
      velocity_threshold_rad_s=self.config.stable_velocity_rad_s,
      success_hold_s=self.config.success_hold_s,
      default_duration_s=self.config.episode_length_s,
    )

  def make_env_cfg(self, num_envs: int = 512) -> ManagerBasedRlEnvCfg:
    return make_env_cfg(num_envs=num_envs, task_cfg=self.config)


def make_env_cfg(
  *,
  num_envs: int = 512,
  task_cfg: CombinedTaskCfg = CombinedTaskCfg(),
  model_cfg: DoublePendulumModelCfg = DoublePendulumModelCfg(),
) -> ManagerBasedRlEnvCfg:
  if num_envs <= 0:
    raise ValueError("num_envs must be positive")

  pendulum_cfg = EntityCfg(
    spec_fn=lambda: build_double_pendulum_spec(model_cfg),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        BuiltinMotorActuatorCfg(
          target_names_expr=(DEFAULT_CONTRACT.actuated_joint,),
          effort_limit=DEFAULT_CONTRACT.torque_limit_nm,
          armature=0.002,
        ),
      )
    ),
  )
  observations = {
    name: ObservationGroupCfg(
      {"state": ObservationTermCfg(func=mdp.policy_state)},
      enable_corruption=False,
    )
    for name in ("actor", "critic")
  }
  actions = {
    "base_torque": JointEffortActionCfg(
      entity_name="pendulum",
      actuator_names=(DEFAULT_CONTRACT.actuated_joint,),
      scale=DEFAULT_CONTRACT.torque_limit_nm,
    )
  }
  events = {
    "reset_pendulum": EventTermCfg(
      func=mdp.reset_pendulum,
      mode="reset",
      params={
        "near_upright_probability": task_cfg.near_upright_probability,
        "hanging_probability": task_cfg.hanging_probability,
        "near_upright_angle_rad": task_cfg.near_upright_angle_rad,
        "hanging_angle_rad": task_cfg.hanging_angle_rad,
        "random_velocity_rad_s": task_cfg.random_velocity_rad_s,
      },
    )
  }
  rewards = {
    "upright": RewardTermCfg(func=mdp.upright_alignment, weight=1.0),
    "quiet_upright": RewardTermCfg(
      func=mdp.balancing_bonus,
      weight=0.5,
      params={
        "angle_threshold_rad": task_cfg.stable_angle_rad,
        "velocity_threshold_rad_s": task_cfg.stable_velocity_rad_s,
      },
    ),
    "near_goal_velocity": RewardTermCfg(
      func=mdp.near_goal_velocity_l2,
      weight=-0.03,
    ),
    "torque": RewardTermCfg(func=mdp.torque_l2, weight=-0.002),
  }
  terminations = {
    "time_limit": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "numerical_failure": TerminationTermCfg(func=mdp.numerical_failure),
  }
  return ManagerBasedRlEnvCfg(
    decimation=DEFAULT_CONTRACT.decimation,
    scene=SceneCfg(
      num_envs=num_envs,
      env_spacing=1.5,
      entities={"pendulum": pendulum_cfg},
    ),
    observations=observations,
    actions=actions,
    events=events,
    rewards=rewards,
    terminations=terminations,
    sim=SimulationCfg(
      mujoco=MujocoCfg(timestep=DEFAULT_CONTRACT.physics_timestep_s)
    ),
    viewer=ViewerConfig(lookat=(0.0, 0.0, -0.5), distance=2.0, elevation=-10.0),
    episode_length_s=task_cfg.episode_length_s,
    is_finite_horizon=True,
    scale_rewards_by_dt=False,
  )
