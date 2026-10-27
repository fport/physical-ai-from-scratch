# 35 · How many milliseconds behind is the robot's brain?

> **Status: coming soon.** This page is a skeleton. The experiment has not been run yet; the experiment below is a **plan**, not a result. Numbers are written only after the run.

| | |
|---|---|
| **Phase** | [7 · Sim to real](README.md) |
| **Prerequisites** | [11](../phase-2-muscle/11-pd-control.md), [15](../phase-3-senses/15-noise-and-latency.md), [26](../phase-4-learning-from-demonstration/26-end-to-end-imitation.md) |
| **Tools** | MuJoCo + real SO-101 |
| **Status** | 🚧 coming soon |

## Learning goal

See how control rate and latency break stability.

## Lesson question

When 0–100 ms of latency is added to the control loop, how do overshoot and success change?

## Why it matters

LeRobot records at ~30 fps and has a separate `rtc` inference mode for slow policies. Source: [LeRobot · imitation learning on real robots](https://huggingface.co/docs/lerobot/il_robots).

## Planned experiment

Add 0–100 ms latency to the PID / policy loop in sim → overshoot and success; measured loop Hz on the real arm.

As in every lesson, a **sweep**: one parameter changes, one number is measured. The result is written to `results/35/result.json`.

## Sources

- [LeRobot · imitation learning on real robots](https://huggingface.co/docs/lerobot/il_robots)

---

_Want to help write this lesson? See [Contributing](../contributing.md)._
