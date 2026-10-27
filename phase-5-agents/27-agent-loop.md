# 27 · The agent loop in 25 lines

> **Status: coming soon.** This page is a skeleton. The experiment has not been run yet; the experiment below is a **plan**, not a result. Numbers are written only after the run.

| | |
|---|---|
| **Phase** | [5 · Agents](README.md) |
| **Prerequisites** | [09](../phase-1-space-and-body/09-inverse-kinematics.md) |
| **Tools** | Ollama + MuJoCo |
| **Status** | 🚧 coming soon |

## Learning goal

Write an LLM agent's "think → call a tool → observe" loop from scratch.

## Lesson question

Over 10 commands, how often does a local model call the right tool?

## Why it matters

_Will open with a working real-world example (with sources). To be written._

## Planned experiment

Ollama tool calling + a `move_to(x, y)` tool (IK, lesson 09); rate of correct tool calls over 10 commands.

As in every lesson, a **sweep**: one parameter changes, one number is measured. The result is written to `results/27/result.json`.

## Sources

- [Ollama · tool calling](https://ollama.com/blog/tool-support)

---

_Want to help write this lesson? See [Contributing](../contributing.md)._
