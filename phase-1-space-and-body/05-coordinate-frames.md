# 05 · There is no "left" for a robot: coordinate frames

> **Status: coming soon.** This page is a skeleton. The experiment has not been run yet; the experiment below is a **plan**, not a result. Numbers are written only after the run.

| | |
|---|---|
| **Phase** | [1 · Space and body](README.md) |
| **Prerequisites** | [01](../phase-0-world-frame-by-frame/01-world-as-text.md) |
| **Tools** | MuJoCo + Menagerie |
| **Status** | 🚧 coming soon |

## Learning goal

Learn that every point is defined relative to a frame, and that a transform matrix means "changing your point of view".

## Lesson question

When the arm rotates, which of the cube's coordinates change: in the world, base or gripper frame?

## Why it matters

_Will open with a working real-world example (with sources). To be written._

## Planned experiment

The same cube's coordinates in the world, base and gripper frames; when the arm rotates only one of them changes. Menagerie `robotstudio_so101` model.

As in every lesson, a **sweep**: one parameter changes, one number is measured. The result is written to `results/05/result.json`.

## Sources

- [Lynch & Park, *Modern Robotics*](https://hades.mech.northwestern.edu/index.php/Modern_Robotics) (Ch. 3, rigid-body motions)
- [Tedrake, *Robotic Manipulation*](https://manipulation.mit.edu/) (App. A, spatial algebra)
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)

---

_Want to help write this lesson? See [Contributing](../contributing.md)._
