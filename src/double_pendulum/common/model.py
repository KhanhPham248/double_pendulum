"""MuJoCo specification for the base-actuated double pendulum."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco


@dataclass(frozen=True)
class DoublePendulumModelCfg:
  """Physical parameters shared by training and deployment models."""

  link_length: float = 0.5
  link_radius: float = 0.025
  link_mass: float = 0.5
  base_damping: float = 0.025
  elbow_damping: float = 0.015

  def __post_init__(self) -> None:
    if min(self.link_length, self.link_radius, self.link_mass) <= 0.0:
      raise ValueError("link dimensions and mass must be positive")
    if min(self.base_damping, self.elbow_damping) < 0.0:
      raise ValueError("joint damping must be non-negative")


def _set_hinge_damping(joint: mujoco.MjsJoint, value: float) -> None:
  """Set hinge damping across scalar and vector-valued MjSpec APIs."""

  damping = joint.damping
  if hasattr(damping, "shape"):
    damping[0] = value
  else:
    joint.damping = value


def build_double_pendulum_spec(
  cfg: DoublePendulumModelCfg = DoublePendulumModelCfg(),
) -> mujoco.MjSpec:
  """Build two hinges without an actuator; mJLab adds the base motor."""

  spec = mujoco.MjSpec()
  link1 = spec.worldbody.add_body(name="link1")
  base_joint = link1.add_joint(
    name="base_joint",
    type=mujoco.mjtJoint.mjJNT_HINGE,
    axis=(0.0, 1.0, 0.0),
  )
  _set_hinge_damping(base_joint, cfg.base_damping)
  link1_geom = link1.add_geom(
    name="link1_geom",
    type=mujoco.mjtGeom.mjGEOM_CAPSULE,
    fromto=(0.0, 0.0, 0.0, 0.0, 0.0, -cfg.link_length),
    size=(cfg.link_radius,),
  )
  link1_geom.mass = cfg.link_mass
  link1_geom.contype = 0
  link1_geom.conaffinity = 0

  link2 = link1.add_body(name="link2", pos=(0.0, 0.0, -cfg.link_length))
  elbow_joint = link2.add_joint(
    name="elbow_joint",
    type=mujoco.mjtJoint.mjJNT_HINGE,
    axis=(0.0, 1.0, 0.0),
  )
  _set_hinge_damping(elbow_joint, cfg.elbow_damping)
  link2_geom = link2.add_geom(
    name="link2_geom",
    type=mujoco.mjtGeom.mjGEOM_CAPSULE,
    fromto=(0.0, 0.0, 0.0, 0.0, 0.0, -cfg.link_length),
    size=(cfg.link_radius,),
  )
  link2_geom.mass = cfg.link_mass
  link2_geom.contype = 0
  link2_geom.conaffinity = 0
  link2.add_site(name="tip", pos=(0.0, 0.0, -cfg.link_length), size=(0.01,))
  return spec
