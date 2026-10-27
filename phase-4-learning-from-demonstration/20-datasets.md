# 20 · What does a robot dataset look like?

> **Status: coming soon.** This page is a skeleton. The experiment has not been run yet; the experiment below is a **plan**, not a result. Numbers are written only after the run.

| | |
|---|---|
| **Phase** | [4 · Learning from demonstration](README.md) |
| **Prerequisites** | [19](../phase-4-learning-from-demonstration/19-replay.md) |
| **Tools** | LeRobot + real SO-101 |
| **Status** | 🚧 coming soon |

## Learning goal

Read what a robot dataset looks like (frame + state + action + time).

## Lesson question

How many frames is a 50-episode recording, and what is stored in each frame?

## Why it matters

The LeRobot hardware guide gives a worked example: 50 episodes × 30 fps × 30 s ≈ 45,000 frames. Source: [LeRobot · hardware guide](https://huggingface.co/docs/lerobot/hardware_guide).

## Planned experiment

Inspect `lerobot/svla_so100_pickplace` on the HF Hub; open our own 5-episode recording in the LeRobot visualizer.

As in every lesson, a **sweep**: one parameter changes, one number is measured. The result is written to `results/20/result.json`.

## Sources

- [LeRobot · hardware guide](https://huggingface.co/docs/lerobot/hardware_guide)
- [LeRobot · imitation learning on real robots](https://huggingface.co/docs/lerobot/il_robots)

---

_Want to help write this lesson? See [Contributing](../contributing.md)._
