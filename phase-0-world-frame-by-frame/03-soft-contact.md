# 03 · In simulation the table is a sponge: soft contact

> **Status: coming soon.** This page is a skeleton. The experiment has not been run yet; the experiment below is a **plan**, not a result. Numbers are written only after the run.

| | |
|---|---|
| **Phase** | [0 · World, frame by frame](README.md) |
| **Prerequisites** | [02](../phase-0-world-frame-by-frame/02-timestep.md) |
| **Tools** | MuJoCo (Mac / Colab CPU) |
| **Status** | 🚧 coming soon |

## Learning goal

See that contact in simulation is a soft spring, not a hard wall, and that objects sink into each other a little.

## Lesson question

How many millimetres does a cube resting on a table sink in, and which number decides it?

## Why it matters

_Will open with a working real-world example (with sources). To be written._

> Note: In lesson 01's run the default soft contact sinks the cube a few mm into the table; this lesson turns that observation into a measurement.

## Planned experiment

Sweep `solref` (contact stiffness/damping): penetration depth of the cube into the table (mm) and bounce as stiffness rises.

As in every lesson, a **sweep**: one parameter changes, one number is measured. The result is written to `results/03/result.json`.

## Sources

- [MuJoCo · Computation](https://mujoco.readthedocs.io/en/stable/computation/index.html)
- [MuJoCo XML reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html) (`solref`, `solimp`)

---

_Want to help write this lesson? See [Contributing](../contributing.md)._
