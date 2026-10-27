# 33 · Thousands of robots falling at once: PPO intuition

> **Status: coming soon.** This page is a skeleton. The experiment has not been run yet; the experiment below is a **plan**, not a result. Numbers are written only after the run.

| | |
|---|---|
| **Phase** | [6 · Learning by trial](README.md) |
| **Prerequisites** | [32](../phase-6-learning-by-trial/32-q-learning.md) |
| **Tools** | MuJoCo Playground (Colab GPU) |
| **Status** | 🚧 coming soon |

## Learning goal

See that a policy improves through trial (policy gradient → PPO intuition) and that thousands of parallel robots on a GPU cut training to minutes.

## Lesson question

As the number of parallel environments grows, how do wall-clock time and reward change?

## Why it matters

Unitree's open training code `unitree_rl_gym` trains with 4,096 parallel environments by default (`num_envs = 4096`). Source: [legged_robot_config.py](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/base/legged_robot_config.py).

> Note: It has not been verified yet that the Playground locomotion notebook runs on a free Colab GPU.

## Planned experiment

Playground locomotion notebook; sweep the number of parallel environments → wall-clock time and reward.

As in every lesson, a **sweep**: one parameter changes, one number is measured. The result is written to `results/33/result.json`.

## Sources

- [MuJoCo Playground](https://playground.mujoco.org/) · [arXiv 2502.08844](https://arxiv.org/abs/2502.08844)
- [PPO, arXiv 1707.06347](https://arxiv.org/abs/1707.06347)

---

_Want to help write this lesson? See [Contributing](../contributing.md)._
