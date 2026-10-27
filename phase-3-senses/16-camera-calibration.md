# 16 · Your camera sees straight lines as curves: calibration

> **Status: coming soon.** This page is a skeleton. The experiment has not been run yet; the experiment below is a **plan**, not a result. Numbers are written only after the run.

| | |
|---|---|
| **Phase** | [3 · Senses](README.md) |
| **Prerequisites** | [05](../phase-1-space-and-body/05-coordinate-frames.md) |
| **Tools** | OpenCV (Mac / Colab) |
| **Status** | 🚧 coming soon |

## Learning goal

Learn that a camera is a matrix projecting 3D onto 2D and that calibration finds that matrix.

## Lesson question

How far does the reprojection error drop (in pixels) once distortion is corrected?

## Why it matters

_Will open with a working real-world example (with sources). To be written._

## Planned experiment

OpenCV calibration with a checkerboard; reprojection error (pixels) before/after undistortion. A phone camera is enough.

As in every lesson, a **sweep**: one parameter changes, one number is measured. The result is written to `results/16/result.json`.

## Sources

- [Tedrake, *Robotic Manipulation*](https://manipulation.mit.edu/) (Ch. 4, geometric pose estimation)

---

_Want to help write this lesson? See [Contributing](../contributing.md)._
