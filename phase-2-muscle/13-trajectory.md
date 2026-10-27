# 13 · Joints moved in a straight line, the tip drew a curve: trajectories

> **Status: coming soon.** This page is a skeleton. The experiment has not been run yet; the experiment below is a **plan**, not a result. Numbers are written only after the run.

| | |
|---|---|
| **Phase** | [2 · Muscle](README.md) |
| **Prerequisites** | [09](../phase-1-space-and-body/09-inverse-kinematics.md), [11](../phase-2-muscle/11-pd-control.md) |
| **Tools** | MuJoCo + real SO-101 |
| **Status** | 🚧 coming soon |

## Learning goal

Learn that how you get to the target (the trajectory) matters.

## Lesson question

Is a straight line in joint space also a straight line at the tip?

## Why it matters

_Will open with a working real-world example (with sources). To be written._

## Planned experiment

Joint-space vs Cartesian interpolation; peak acceleration of a step target vs a smooth (min-jerk) profile.

As in every lesson, a **sweep**: one parameter changes, one number is measured. The result is written to `results/13/result.json`.

## Sources

- [Lynch & Park, *Modern Robotics*](https://hades.mech.northwestern.edu/index.php/Modern_Robotics) (Ch. 9, trajectory generation)

---

_Want to help write this lesson? See [Contributing](../contributing.md)._
