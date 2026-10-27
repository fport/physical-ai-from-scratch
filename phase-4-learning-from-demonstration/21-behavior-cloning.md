# 21 · Nobody wrote code: behavior cloning

> **Status: coming soon.** This page is a skeleton. The experiment has not been run yet; the experiment below is a **plan**, not a result. Numbers are written only after the run.

| | |
|---|---|
| **Phase** | [4 · Learning from demonstration](README.md) |
| **Prerequisites** | [07](../phase-1-space-and-body/07-forward-kinematics.md), [20](../phase-4-learning-from-demonstration/20-datasets.md) |
| **Tools** | MuJoCo + PyTorch (Mac) |
| **Status** | 🚧 coming soon |

## Learning goal

Write behavior cloning from scratch with a 30-line MLP.

## Lesson question

How does success change as demonstrations go from 5 to 50?

## Why it matters

_Will open with a working real-world example (with sources). To be written._

## Planned experiment

A state-based "reach the cube" task in sim; sweep the number of demonstrations (5/10/20/50) → success.

As in every lesson, a **sweep**: one parameter changes, one number is measured. The result is written to `results/21/result.json`.

## Sources

- [Berkeley CS285](https://rail.eecs.berkeley.edu/deeprlcourse/) (behavior cloning)
- [Robot Learning: A Tutorial, arXiv 2510.12403](https://arxiv.org/abs/2510.12403)

---

_Want to help write this lesson? See [Contributing](../contributing.md)._
