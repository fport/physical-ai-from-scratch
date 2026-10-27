# 39 · Paths from noise: diffusion and flow

> **Status: coming soon.** This page is a skeleton. The experiment has not been run yet; the experiment below is a **plan**, not a result. Numbers are written only after the run.

| | |
|---|---|
| **Phase** | [8 · Big models](README.md) |
| **Prerequisites** | [24](../phase-4-learning-from-demonstration/24-two-right-answers.md), [38](../phase-8-big-models/38-vla.md) |
| **Tools** | NumPy/PyTorch (Colab CPU) |
| **Status** | 🚧 coming soon |

## Learning goal

Get an intuition for how models that generate actions from noise (diffusion / flow matching) handle multimodality.

## Lesson question

On lesson 24's data, does a small flow/diffusion model hit the obstacle?

## Why it matters

_Will open with a working real-world example (with sources). To be written._

## Planned experiment

Lesson 24's 2D data: does a small flow/diffusion model produce both paths separately; collision rate vs the MLP.

As in every lesson, a **sweep**: one parameter changes, one number is measured. The result is written to `results/39/result.json`.

## Sources

- [Diffusion Policy, arXiv 2303.04137](https://arxiv.org/abs/2303.04137)
- [π0, arXiv 2410.24164](https://arxiv.org/abs/2410.24164)

---

_Want to help write this lesson? See [Contributing](../contributing.md)._
