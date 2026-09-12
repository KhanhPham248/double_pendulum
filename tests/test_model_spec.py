from __future__ import annotations

import math

import mujoco
import numpy as np

from double_pendulum.common import DEFAULT_CONTRACT, build_double_pendulum_spec


def test_raw_model_has_two_hinges_and_no_hidden_actuator() -> None:
  model = build_double_pendulum_spec().compile()
  assert model.nq == 2
  assert model.nv == 2
  assert model.nu == 0
  assert [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
    for index in range(model.njnt)
  ] == ["base_joint", "elbow_joint"]
  assert np.allclose(model.dof_damping, (0.025, 0.015))


def test_upright_target_places_tip_above_the_base() -> None:
  model = build_double_pendulum_spec().compile()
  data = mujoco.MjData(model)
  data.qpos[:] = (math.pi, 0.0)
  mujoco.mj_forward(model, data)
  tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip")
  assert data.site_xpos[tip_id, 2] > 0.95


def test_runtime_xml_matches_physical_contract() -> None:
  model = mujoco.MjModel.from_xml_path("assets/double_pendulum.xml")
  assert (model.nq, model.nv, model.nu) == (2, 2, 1)
  assert model.opt.timestep == DEFAULT_CONTRACT.physics_timestep_s
  assert np.allclose(
    model.actuator_ctrlrange[0],
    (-DEFAULT_CONTRACT.torque_limit_nm, DEFAULT_CONTRACT.torque_limit_nm),
  )
  assert np.allclose(model.body_mass[2:4], (0.5, 0.5))
  assert np.allclose(model.dof_damping, (0.025, 0.015))
