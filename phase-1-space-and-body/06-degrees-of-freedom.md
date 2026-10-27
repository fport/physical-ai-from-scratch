# 06 · 6 motors, but not 6 degrees of freedom

> **Status: coming soon.** This page is a skeleton. The experiment has not been run yet; the experiment below is a **plan**, not a result. Numbers are written only after the run.

| | |
|---|---|
| **Phase** | [1 · Space and body](README.md) |
| **Prerequisites** | [05](../phase-1-space-and-body/05-coordinate-frames.md) |
| **Tools** | MuJoCo + real SO-101 |
| **Status** | 🚧 coming soon |

## Learning goal

Understand the difference between degrees of freedom and motor count, and why a 5-axis arm cannot reach every pose.

## Lesson question

Can the SO-101 circle around a cube while keeping its gripper horizontal?

## Why it matters

The SO-101 has 6 motors: 5 joints (shoulder pan, shoulder lift, elbow flex, wrist flex, wrist roll) and 1 gripper. Source: [LeRobot · SO-101](https://huggingface.co/docs/lerobot/so101).

## Planned experiment

Try to circle a cube with the gripper held horizontal; measure the angle error between the requested and the reached orientation. Real arm as the human layer.

As in every lesson, a **sweep**: one parameter changes, one number is measured. The result is written to `results/06/result.json`.

## Sources

- [LeRobot · SO-101](https://huggingface.co/docs/lerobot/so101)
- [Lynch & Park, *Modern Robotics*](https://hades.mech.northwestern.edu/index.php/Modern_Robotics) (Ch. 2, configuration space)

---

_Want to help write this lesson? See [Contributing](../contributing.md)._
