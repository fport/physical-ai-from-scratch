# 36 · My simulated motor wasn't like the real one: system identification

> **Status: coming soon.** This page is a skeleton. The experiment has not been run yet; the experiment below is a **plan**, not a result. Numbers are written only after the run.

| | |
|---|---|
| **Phase** | [7 · Sim to real](README.md) |
| **Prerequisites** | [10](../phase-2-muscle/10-position-servo.md), [12](../phase-2-muscle/12-gravity.md), [35](../phase-7-sim-to-real/35-latency-and-control-rate.md) |
| **Tools** | Real SO-101 + MuJoCo |
| **Status** | 🚧 coming soon |

## Learning goal

Measure things to make the simulated motor match the real one (system identification).

## Lesson question

For the same step command, what is the tracking error between sim and real arm before and after tuning?

## Why it matters

NVIDIA's SO-101 sim-to-real course dedicates its fourth strategy to measuring the actuator gap. Source: [NVIDIA · Train an SO-101 Robot From Sim-to-Real](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/index.html).

## Planned experiment

Same step command to real and simulated arm; grid search over kp/damping → tracking error before/after.

As in every lesson, a **sweep**: one parameter changes, one number is measured. The result is written to `results/36/result.json`.

## Sources

- [NVIDIA · Train an SO-101 Robot From Sim-to-Real](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/index.html)
- [Tedrake, *Underactuated Robotics*](https://underactuated.mit.edu/) (system identification)

---

_Want to help write this lesson? See [Contributing](../contributing.md)._
