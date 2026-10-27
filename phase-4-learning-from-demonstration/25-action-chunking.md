# 25 · The robot changed its mind every frame: action chunking

> **Status: coming soon.** This page is a skeleton. The experiment has not been run yet; the experiment below is a **plan**, not a result. Numbers are written only after the run.

| | |
|---|---|
| **Phase** | [4 · Learning from demonstration](README.md) |
| **Prerequisites** | [23](../phase-4-learning-from-demonstration/23-distribution-shift.md), [24](../phase-4-learning-from-demonstration/24-two-right-answers.md) |
| **Tools** | LeRobot ACT (sim) |
| **Status** | 🚧 coming soon |

## Learning goal

See that predicting an action chunk instead of single steps reduces jitter (the ACT idea).

## Lesson question

As chunk size goes from 1 to 100, how do jerk and success change?

## Why it matters

The LeRobot docs present ACT as the first recommended policy. Source: [LeRobot · ACT](https://huggingface.co/docs/lerobot/act).

## Planned experiment

Chunk size sweep (1/10/50/100) → jerk and success, LeRobot ACT in sim.

As in every lesson, a **sweep**: one parameter changes, one number is measured. The result is written to `results/25/result.json`.

## Sources

- [ACT, arXiv 2304.13705](https://arxiv.org/abs/2304.13705)
- [LeRobot · ACT](https://huggingface.co/docs/lerobot/act)

---

_Want to help write this lesson? See [Contributing](../contributing.md)._
