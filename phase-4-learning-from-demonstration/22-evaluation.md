# 22 · I said 90% success. How sure am I? Evaluation

> **Status: coming soon.** This page is a skeleton. The experiment has not been run yet; the experiment below is a **plan**, not a result. Numbers are written only after the run.

| | |
|---|---|
| **Phase** | [4 · Learning from demonstration](README.md) |
| **Prerequisites** | [21](../phase-4-learning-from-demonstration/21-behavior-cloning.md) |
| **Tools** | MuJoCo + NumPy |
| **Status** | 🚧 coming soon |

## Learning goal

Learn that a success rate is an estimate and how much confidence a given number of trials buys.

## Lesson question

If you try the same policy 10, 50 and 200 times, how much does the confidence interval narrow?

## Why it matters

_Will open with a working real-world example (with sources). To be written._

> Note: Lesson 19's result file already gives Wilson 95% intervals for 8-trial rates (e.g. 1/8 → 0.02–0.47); this lesson covers the idea in depth.

## Planned experiment

Run the same policy 10, 50 and 200 times; watch the Wilson interval narrow.

As in every lesson, a **sweep**: one parameter changes, one number is measured. The result is written to `results/22/result.json`.

## Sources

- [LeRobot · imitation learning on real robots](https://huggingface.co/docs/lerobot/il_robots) (evaluation with `lerobot-rollout`)

---

_Want to help write this lesson? See [Contributing](../contributing.md)._
