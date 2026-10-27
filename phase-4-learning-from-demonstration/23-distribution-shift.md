# 23 · Does a perfect imitator sink at the first mistake? Distribution shift

> **Status: coming soon.** This page is a skeleton. The experiment has not been run yet; the experiment below is a **plan**, not a result. Numbers are written only after the run.

| | |
|---|---|
| **Phase** | [4 · Learning from demonstration](README.md) |
| **Prerequisites** | [21](../phase-4-learning-from-demonstration/21-behavior-cloning.md), [22](../phase-4-learning-from-demonstration/22-evaluation.md) |
| **Tools** | MuJoCo + PyTorch |
| **Status** | 🚧 coming soon |

## Learning goal

See that an imitating policy drifts into unfamiliar states after a small error and that errors compound (distribution shift); the DAgger idea.

## Lesson question

How much does a small push during rollout reduce success, and how much do correction demonstrations recover?

## Why it matters

_Will open with a working real-world example (with sources). To be written._

## Planned experiment

Add a small push during rollout → success drop; recovery after adding correction demonstrations.

As in every lesson, a **sweep**: one parameter changes, one number is measured. The result is written to `results/23/result.json`.

## Sources

- [DAgger, arXiv 1011.0686](https://arxiv.org/abs/1011.0686)
- [LeRobot · imitation learning on real robots](https://huggingface.co/docs/lerobot/il_robots) (`dagger` strategy)
- [Berkeley CS285](https://rail.eecs.berkeley.edu/deeprlcourse/)

---

_Want to help write this lesson? See [Contributing](../contributing.md)._
