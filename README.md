# Double Pendulum Reinforcement Learning

An independent reinforcement learning project for the **swing-up and balancing
control of an underactuated double pendulum** in MuJoCo.

The system consists of two rotary joints but only one actuator, located at the
base joint. The elbow joint is passive, so the controller must exploit the
system dynamics to inject energy, swing the pendulum upward, and stabilize both
links around the upright configuration.

## Project scope

The current task uses a single policy to learn the complete behavior:

1. move the pendulum from the hanging state;
2. build sufficient energy for swing-up;
3. capture the upright configuration;
4. maintain stable balance with limited control effort.

The environment is built on **MuJoCo** and **mJLab**. Training pipelines for
**Soft Actor-Critic (SAC)** and **Proximal Policy Optimization (PPO)** are kept
separate because the algorithms use different data-collection and optimization
flows. They share the same physical model, task definition, observation and
action contract, evaluation tools, and ONNX runtime.

## Control interface

The policy observes the joint angles and velocities through a continuous
trigonometric representation:

```text
[sin(q1), cos(q1), sin(q2), cos(q2), qdot1, qdot2]
```

It produces one normalized action that is converted into torque at the base
joint. No torque is applied directly to the elbow joint.

```text
MuJoCo state -> policy observation -> RL policy -> base-joint torque
```

## Design goals

- Keep the task definition independent of the learning algorithm.
- Support SAC and PPO without coupling their training loops.
- Use one consistent policy contract for training, evaluation, and inference.
- Export the latest deterministic policy to ONNX for portable simulation.
- Make new tasks and policies easy to add without changing existing ones.

## Future direction

The single-policy task provides the initial baseline. The architecture is
intended to evolve toward a hierarchical controller containing:

- a dedicated swing-up expert;
- a dedicated balancing expert;
- a meta-policy that selects or transitions between the two experts.

Each expert will remain independently trainable, testable, and deployable. This
separation makes it possible to improve one behavior without retraining or
modifying the others, while still supporting an integrated closed-loop control
system later.

## Repository overview

```text
assets/                  MuJoCo model of the double pendulum
scripts/                 Training, evaluation, and simulation entry points
src/double_pendulum/
  algorithms/            Independent SAC and PPO implementations
  common/                Shared model and policy contracts
  tasks/                 Reinforcement-learning task definitions
  training/              Algorithm-specific training pipelines
  export/                Checkpoint and ONNX export utilities
  runtime/               Standalone MuJoCo and ONNX inference
  evaluation/            Metrics and policy evaluation
src/mjlab/               Vendored mJLab components required by the project
tests/                   Unit and integration contract tests
```

This repository focuses on a clear and extensible research structure for
developing, comparing, and composing reinforcement-learning controllers for an
underactuated double-pendulum system.

Training-machine setup and execution instructions are available in
[TRAINING_GUIDE.md](TRAINING_GUIDE.md).
