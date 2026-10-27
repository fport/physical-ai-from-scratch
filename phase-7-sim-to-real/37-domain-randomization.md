# 37 · Breaking the sim on purpose: domain randomization

> **Status: coming soon.** This page is a skeleton. The experiment has not been run yet; the experiment below is a **plan**, not a result. Numbers are written only after the run.

| | |
|---|---|
| **Phase** | [7 · Sim to real](README.md) |
| **Prerequisites** | [33](../phase-6-learning-by-trial/33-parallel-rl.md), [36](../phase-7-sim-to-real/36-system-identification.md) |
| **Tools** | MuJoCo or Colab GPU |
| **Status** | 🚧 coming soon |

## Learning goal

Train a robust policy by deliberately perturbing the simulation (domain randomization) and see the cost of perturbing too much.

## Lesson question

As the randomization range widens, how does success change in a separate test sim?

## Why it matters

In Unitree's training code each environment's friction is drawn from 0.5–1.25 (`randomize_friction = True`, `friction_range = [0.5, 1.25]`). Source: [legged_robot_config.py](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/base/legged_robot_config.py). See [lesson 04](../phase-0-world-frame-by-frame/04-friction.md).

## Planned experiment

Sweep the width of friction/mass/latency randomization → success in a separate test sim standing in for "real".

As in every lesson, a **sweep**: one parameter changes, one number is measured. The result is written to `results/37/result.json`.

## Sources

- [Tobin et al. 2017, arXiv 1703.06907](https://arxiv.org/abs/1703.06907)
- [Tan et al. 2018, arXiv 1804.10332](https://arxiv.org/abs/1804.10332)
- [NVIDIA · Train an SO-101 Robot From Sim-to-Real](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/index.html)

---

_Want to help write this lesson? See [Contributing](../contributing.md)._
