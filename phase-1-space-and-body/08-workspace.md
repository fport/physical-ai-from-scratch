# 08 · Can this arm touch its own base? Workspace

> **Status: coming soon.** This page is a skeleton. The experiment has not been run yet; the experiment below is a **plan**, not a result. Numbers are written only after the run.

| | |
|---|---|
| **Phase** | [1 · Space and body](README.md) |
| **Prerequisites** | [07](../phase-1-space-and-body/07-forward-kinematics.md) |
| **Tools** | MuJoCo + real SO-101 |
| **Status** | 🚧 coming soon |

## Learning goal

See the shape and limits of the workspace (the set of reachable points).

## Lesson question

What shape does the cloud of tip points from random joint angles draw?

## Why it matters

_Will open with a working real-world example (with sources). To be written._

## Planned experiment

50,000 random joint configurations → tip point cloud and its cross-section at table height; 3 boundary points checked on the real arm.

As in every lesson, a **sweep**: one parameter changes, one number is measured. The result is written to `results/08/result.json`.

## Sources

- [Lynch & Park, *Modern Robotics*](https://hades.mech.northwestern.edu/index.php/Modern_Robotics) (Ch. 4)

---

_Want to help write this lesson? See [Contributing](../contributing.md)._
