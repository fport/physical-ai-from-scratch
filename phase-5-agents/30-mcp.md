# 30 · Three agents, one arm: MCP

> **Status: coming soon.** This page is a skeleton. The experiment has not been run yet; the experiment below is a **plan**, not a result. Numbers are written only after the run.

| | |
|---|---|
| **Phase** | [5 · Agents](README.md) |
| **Prerequisites** | [27](../phase-5-agents/27-agent-loop.md), [29](../phase-5-agents/29-agent-safety.md) |
| **Tools** | MCP + MuJoCo |
| **Status** | 🚧 coming soon |

## Learning goal

Expose the arm as an MCP server and hand it to any agent as a tool.

## Lesson question

How many ms does a tool call over MCP add?

## Why it matters

_Will open with a working real-world example (with sources). To be written._

## Planned experiment

MCP server with `get_state`, `move_to`; connecting from two clients, tool call latency (ms).

As in every lesson, a **sweep**: one parameter changes, one number is measured. The result is written to `results/30/result.json`.

## Sources

- [Model Context Protocol](https://modelcontextprotocol.io/)

---

_Want to help write this lesson? See [Contributing](../contributing.md)._
