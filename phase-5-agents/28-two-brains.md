# 28 · The big model doesn't move the arm, it gives orders

> **Status: coming soon.** This page is a skeleton. The experiment has not been run yet; the experiment below is a **plan**, not a result. Numbers are written only after the run.

| | |
|---|---|
| **Phase** | [5 · Agents](README.md) |
| **Prerequisites** | [18](../phase-3-senses/18-color-vs-open-vocabulary.md), [26](../phase-4-learning-from-demonstration/26-end-to-end-imitation.md), [27](../phase-5-agents/27-agent-loop.md) |
| **Tools** | Ollama + MuJoCo |
| **Status** | 🚧 coming soon |

## Learning goal

See that an agent's tools are the skills built in earlier phases: the big model knows what to do, the small skill knows how.

## Lesson question

In a 3-step task, which step fails most often?

## Why it matters

_Will open with a working real-world example (with sources). To be written._

## Planned experiment

`find` (lesson 18) + `move_to` + `pick` (lesson 26 policy): per-step error in a 3-step task.

As in every lesson, a **sweep**: one parameter changes, one number is measured. The result is written to `results/28/result.json`.

## Sources

- _To be added when written._

---

_Want to help write this lesson? See [Contributing](../contributing.md)._
