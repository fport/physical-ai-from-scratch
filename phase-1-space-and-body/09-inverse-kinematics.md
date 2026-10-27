# 09 · Same point, two different elbows: inverse kinematics

> **Status: coming soon.** This page is a skeleton. The experiment has not been run yet; the experiment below is a **plan**, not a result. Numbers are written only after the run.

| | |
|---|---|
| **Phase** | [1 · Space and body](README.md) |
| **Prerequisites** | [07](../phase-1-space-and-body/07-forward-kinematics.md), [08](../phase-1-space-and-body/08-workspace.md) |
| **Tools** | MuJoCo + NumPy |
| **Status** | 🚧 coming soon |

## Learning goal

Build IK as "crawling towards the target in small steps" (numerical IK with the Jacobian) and see multiple solutions.

## Lesson question

If you change the starting angle, does the same target get a different solution?

## Why it matters

_Will open with a working real-world example (with sources). To be written._

## Planned experiment

Sweep the starting configuration → different solutions for the same target; iterations and remaining error.

As in every lesson, a **sweep**: one parameter changes, one number is measured. The result is written to `results/09/result.json`.

## Sources

- [Lynch & Park, *Modern Robotics*](https://hades.mech.northwestern.edu/index.php/Modern_Robotics) (Ch. 5–6, Jacobian and IK)
- [Hugging Face Robotics Course](https://huggingface.co/learn/robotics-course) (Unit 2)

---

_Want to help write this lesson? See [Contributing](../contributing.md)._
