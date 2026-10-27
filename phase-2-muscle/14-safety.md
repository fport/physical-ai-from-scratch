# 14 · A robot's first rule: being able to stop itself

> **Status: coming soon.** This page is a skeleton. The experiment has not been run yet; the experiment below is a **plan**, not a result. Numbers are written only after the run.

| | |
|---|---|
| **Phase** | [2 · Muscle](README.md) |
| **Prerequisites** | [10](../phase-2-muscle/10-position-servo.md), [13](../phase-2-muscle/13-trajectory.md) |
| **Tools** | Real SO-101 |
| **Status** | 🚧 coming soon |

## Learning goal

Learn that joint limits, torque/velocity limits and an emergency stop must be in place before touching a real arm.

## Lesson question

How many ms pass between the emergency-stop command and the arm actually stopping?

## Why it matters

_Will open with a working real-world example (with sources). To be written._

> Note: The curriculum research did not find a separate safety page in the LeRobot docs (unverified); this lesson rests on engineering practice and its sources will be added when written.

## Planned experiment

Software limits + a foam-collision test (torque limit on/off), E-STOP latency (ms).

As in every lesson, a **sweep**: one parameter changes, one number is measured. The result is written to `results/14/result.json`.

## Sources

- [LeRobot · SO-101](https://huggingface.co/docs/lerobot/so101)

---

_Want to help write this lesson? See [Contributing](../contributing.md)._
