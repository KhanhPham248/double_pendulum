from __future__ import annotations

from double_pendulum.common import DEFAULT_CONTRACT
from double_pendulum.tasks import get_task, list_tasks


def test_combined_task_has_one_shared_contract() -> None:
  assert list_tasks() == ("combined",)
  task = get_task("combined")
  env = task.make_env_cfg(num_envs=4)
  assert env.scene.num_envs == 4
  assert env.decimation == DEFAULT_CONTRACT.decimation
  assert env.sim.mujoco.timestep == DEFAULT_CONTRACT.physics_timestep_s
  assert tuple(env.actions) == ("base_torque",)
  assert tuple(env.rewards) == (
    "link_alignment",
    "upright_capture",
    "quiet_upright",
    "global_velocity",
    "upright_velocity",
    "torque",
    "action_rate",
  )
  assert task.evaluation.success_hold_s == 5.0
