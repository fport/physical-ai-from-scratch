# 15 · I cleaned the noise, the robot got late: filtering and latency

> **Status: coming soon.** This page is a skeleton. The experiment has not been run yet; the experiment below is a **plan**, not a result. Numbers are written only after the run.

| | |
|---|---|
| **Phase** | [3 · Senses](README.md) |
| **Prerequisites** | [10](../phase-2-muscle/10-position-servo.md) |
| **Tools** | MuJoCo + real SO-101 |
| **Status** | 🚧 coming soon |

## Learning goal

See that every sensor is noisy and that filtering adds latency.

## Lesson question

As the moving-average window grows, how much does noise drop and how much does latency rise?

## Why it matters

_Will open with a working real-world example (with sources). To be written._

> Note: The STS3215 encoder resolution has not been verified yet; it will be checked in the Feetech documentation before the lesson is written.

## Planned experiment

Add noise to joint readings → window sweep: noise ↓, latency (ms) ↑. Compared with real arm readings.

As in every lesson, a **sweep**: one parameter changes, one number is measured. The result is written to `results/15/result.json`.

## Sources

- [Tedrake, *Underactuated Robotics*](https://underactuated.mit.edu/) (state estimation)

---

_Want to help write this lesson? See [Contributing](../contributing.md)._
