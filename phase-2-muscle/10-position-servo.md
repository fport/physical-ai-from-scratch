# 10 · You give the motor an angle, not a force: position servos

> **Status: coming soon.** This page is a skeleton. The experiment has not been run yet; the experiment below is a **plan**, not a result. Numbers are written only after the run.

| | |
|---|---|
| **Phase** | [2 · Muscle](README.md) |
| **Prerequisites** | [06](../phase-1-space-and-body/06-degrees-of-freedom.md) |
| **Tools** | LeRobot + real SO-101, MuJoCo |
| **Status** | 🚧 coming soon |

## Learning goal

Understand that the SO-101's motors are position servos that take an angle, with a control loop inside the motor.

## Lesson question

When you send a servo a target angle, how do the measured position and load change?

## Why it matters

The SO-101's joints use Feetech STS3215 servos; the LeRobot docs list a 1/345 gear ratio. Source: [LeRobot · SO-101](https://huggingface.co/docs/lerobot/so101).

## Planned experiment

Real arm: send a target angle, log the measured position and load. Do the same with the simulated actuator.

As in every lesson, a **sweep**: one parameter changes, one number is measured. The result is written to `results/10/result.json`.

## Sources

- [LeRobot · SO-101](https://huggingface.co/docs/lerobot/so101)
- [Lynch & Park, *Modern Robotics*](https://hades.mech.northwestern.edu/index.php/Modern_Robotics) (Ch. 11, robot control)

---

_Want to help write this lesson? See [Contributing](../contributing.md)._
