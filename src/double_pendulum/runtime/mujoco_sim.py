"""Independent MuJoCo simulation driven only by ONNX actions."""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from double_pendulum.common import PolicyContract
from double_pendulum.common.observations import policy_observation


class MujocoDoublePendulum:
  def __init__(
    self,
    model_path: Path,
    contract: PolicyContract,
    *,
    mass_scale: float = 1.0,
    damping_scale: float = 1.0,
  ) -> None:
    if mass_scale <= 0.0 or damping_scale < 0.0:
      raise ValueError("mass scale must be positive and damping non-negative")
    self.contract = contract
    self.model = mujoco.MjModel.from_xml_path(str(model_path.expanduser().resolve()))
    self.data = mujoco.MjData(self.model)
    self._validate_model()
    for body_name in ("link1", "link2"):
      body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
      if body_id < 0:
        raise ValueError(f"runtime model is missing body {body_name!r}")
      self.model.body_mass[body_id] *= mass_scale
    self.model.dof_damping[:] *= damping_scale
    self._qpos_indices = tuple(
      int(self.model.jnt_qposadr[self._joint_id(name)]) for name in contract.joint_names
    )
    self._qvel_indices = tuple(
      int(self.model.jnt_dofadr[self._joint_id(name)]) for name in contract.joint_names
    )

  def _joint_id(self, name: str) -> int:
    identifier = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if identifier < 0:
      raise ValueError(f"model is missing joint {name!r}")
    return identifier

  def _validate_model(self) -> None:
    if (self.model.nq, self.model.nv, self.model.nu) != (2, 2, 1):
      raise ValueError("runtime model must have two hinges and one actuator")
    if abs(self.model.opt.timestep - self.contract.physics_timestep_s) > 1e-12:
      raise ValueError("MuJoCo timestep does not match policy contract")
    actuator_joint = int(self.model.actuator_trnid[0, 0])
    if actuator_joint != self._joint_id(self.contract.actuated_joint):
      raise ValueError("runtime actuator is not attached to the contracted joint")
    expected_range = np.array(
      [-self.contract.torque_limit_nm, self.contract.torque_limit_nm]
    )
    if not np.allclose(self.model.actuator_ctrlrange[0], expected_range):
      raise ValueError("runtime actuator range does not match policy contract")

  @property
  def qpos(self) -> np.ndarray:
    return self.data.qpos[list(self._qpos_indices)].copy()

  @property
  def qvel(self) -> np.ndarray:
    return self.data.qvel[list(self._qvel_indices)].copy()

  def observation(self) -> np.ndarray:
    observation = policy_observation(self.qpos, self.qvel, np).astype(np.float32)
    if not np.isfinite(observation).all():
      raise FloatingPointError("MuJoCo produced a non-finite observation")
    return observation

  def reset(self, *, seed: int, mode: str = "hanging") -> np.ndarray:
    rng = np.random.default_rng(seed)
    mujoco.mj_resetData(self.model, self.data)
    if mode == "hanging":
      self.data.qpos[:] = rng.uniform(-0.25, 0.25, size=2)
      self.data.qvel[:] = rng.uniform(-0.5, 0.5, size=2)
    elif mode == "upright":
      self.data.qpos[:] = (math.pi, 0.0) + rng.uniform(-0.20, 0.20, size=2)
      self.data.qvel[:] = rng.uniform(-0.25, 0.25, size=2)
    elif mode == "random":
      self.data.qpos[:] = rng.uniform(-math.pi, math.pi, size=2)
      self.data.qvel[:] = rng.uniform(-1.0, 1.0, size=2)
    else:
      raise ValueError("reset mode must be hanging, upright or random")
    mujoco.mj_forward(self.model, self.data)
    return self.observation()

  def apply_velocity_kick(self, magnitude_rad_s: float) -> None:
    self.data.qvel[self._qvel_indices[0]] += magnitude_rad_s

  def step(self, normalized_action: np.ndarray | float) -> np.ndarray:
    action = float(np.asarray(normalized_action).reshape(-1)[0])
    if not math.isfinite(action):
      raise FloatingPointError("policy action is non-finite")
    action = float(np.clip(action, -1.0, 1.0))
    self.data.ctrl[0] = action * self.contract.torque_limit_nm
    for _ in range(self.contract.decimation):
      mujoco.mj_step(self.model, self.data)
    return self.observation()
