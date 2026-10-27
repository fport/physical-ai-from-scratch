# 34 · When the agent games the reward: reward hacking

> **Status: coming soon.** This page is a skeleton. The experiment has not been run yet; the experiment below is a **plan**, not a result. Numbers are written only after the run.

| | |
|---|---|
| **Phase** | [6 · Learning by trial](README.md) |
| **Prerequisites** | [33](../phase-6-learning-by-trial/33-parallel-rl.md) |
| **Tools** | MuJoCo Playground (Colab GPU) |
| **Status** | 🚧 coming soon |

## Learning goal

See that an agent can game its reward (reward hacking).

## Lesson question

If you reward only forward velocity, how does the robot learn to move forward?

## Why it matters

_Will open with a working real-world example (with sources). To be written._

## Planned experiment

A deliberately broken reward: forward velocity only → hopping/crawling forward.

As in every lesson, a **sweep**: one parameter changes, one number is measured. The result is written to `results/34/result.json`.

## Sources

- [MuJoCo Playground](https://playground.mujoco.org/) · [arXiv 2502.08844](https://arxiv.org/abs/2502.08844)

---

_Want to help write this lesson? See [Contributing](../contributing.md)._
