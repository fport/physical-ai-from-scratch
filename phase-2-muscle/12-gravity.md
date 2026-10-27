# 12 · The arm isn't weak, physics makes it sag: gravity

> **Status: coming soon.** This page is a skeleton. The experiment has not been run yet; the experiment below is a **plan**, not a result. Numbers are written only after the run.

| | |
|---|---|
| **Phase** | [2 · Muscle](README.md) |
| **Prerequisites** | [11](../phase-2-muscle/11-pd-control.md) |
| **Tools** | MuJoCo + real SO-101 |
| **Status** | 🚧 coming soon |

## Learning goal

Understand that gravity creates torque on an outstretched arm, which is why pure P control settles below the target.

## Lesson question

How many degrees does the arm sag when stretched horizontally, and how much does gravity compensation remove?

## Why it matters

_Will open with a working real-world example (with sources). To be written._

> Note: In lesson 11's Go2 experiment the body stayed ~7 cm below the target stance: an example of the sag this lesson is about.

## Planned experiment

Sag angle of the horizontally stretched arm: sim (mass sweep) and real arm; gravity compensation on/off.

As in every lesson, a **sweep**: one parameter changes, one number is measured. The result is written to `results/12/result.json`.

## Sources

- [Lynch & Park, *Modern Robotics*](https://hades.mech.northwestern.edu/index.php/Modern_Robotics) (Ch. 8, dynamics; Ch. 11)

---

_Want to help write this lesson? See [Contributing](../contributing.md)._
