# 38 · Inside a VLA: image + sentence → action

> **Status: coming soon.** This page is a skeleton. The experiment has not been run yet; the experiment below is a **plan**, not a result. Numbers are written only after the run.

| | |
|---|---|
| **Phase** | [8 · Big models](README.md) |
| **Prerequisites** | [25](../phase-4-learning-from-demonstration/25-action-chunking.md), [26](../phase-4-learning-from-demonstration/26-end-to-end-imitation.md) |
| **Tools** | LeRobot (Mac/MPS or Colab GPU) |
| **Status** | 🚧 coming soon |

## Learning goal

Read a VLA's inputs and outputs (image + sentence + joint angles → action chunk) as tensor shapes.

## Lesson question

What does SmolVLA take and return in one step, and how many ms does it take?

## Why it matters

_Will open with a working real-world example (with sources). To be written._

> Note: Whether SmolVLA runs on Mac MPS has not been verified yet.

## Planned experiment

SmolVLA (450M) inference: input/output shapes, chunk length, ms per step.

As in every lesson, a **sweep**: one parameter changes, one number is measured. The result is written to `results/38/result.json`.

## Sources

- [SmolVLA, arXiv 2506.01844](https://arxiv.org/abs/2506.01844)
- [LeRobot · SmolVLA](https://huggingface.co/docs/lerobot/smolvla)

---

_Want to help write this lesson? See [Contributing](../contributing.md)._
