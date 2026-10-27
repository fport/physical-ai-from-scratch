# 07 · Two angles, one point: forward kinematics

> **Status: coming soon.** This page is a skeleton. The experiment has not been run yet; the experiment below is a **plan**, not a result. Numbers are written only after the run.

| | |
|---|---|
| **Phase** | [1 · Space and body](README.md) |
| **Prerequisites** | [05](../phase-1-space-and-body/05-coordinate-frames.md) |
| **Tools** | MuJoCo + NumPy |
| **Status** | 🚧 coming soon |

## Learning goal

Compute the tip position from joint angles with forward kinematics (FK).

## Lesson question

How many mm does a 10-line FK function deviate from the simulator's own answer?

## Why it matters

_Will open with a working real-world example (with sources). To be written._

## Planned experiment

Compare a 2-joint FK with the simulated tip; then FK from the simulated chain of the 5-joint SO-101. Metric: difference (mm).

As in every lesson, a **sweep**: one parameter changes, one number is measured. The result is written to `results/07/result.json`.

## Sources

- [Lynch & Park, *Modern Robotics*](https://hades.mech.northwestern.edu/index.php/Modern_Robotics) (Ch. 4)
- [Hugging Face Robotics Course](https://huggingface.co/learn/robotics-course) (Unit 2, classical robotics)

---

_Want to help write this lesson? See [Contributing](../contributing.md)._
