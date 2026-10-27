# 31 · Getting a robot to work by talking to it: Strands

> Look at the measurement, not the agent's answer.

> **Status: draft.** The experiment has been run and the notebook is ready. In the new order this lesson is 31. (In the Koddan Kasa video series its label was "Levha 32", and the notebook still calls it "Levha 32".) The prerequisite lessons 27–30 have not been written yet.

| | |
|---|---|
| **Phase** | [5 · Agents](README.md) |
| **Type** | Experiment |
| **Prerequisites** | [27–30](27-agent-loop.md) (not required for the experiment) |
| **Time** | ~60 min (including model download) |
| **Tools** | `strands-robots` 0.5.3 · `strands-agents` · MuJoCo **3.16.0** · Ollama `qwen3.5:9b` or Claude API |
| **Code** | [notebook](../notebooks/31-strands-agent.ipynb) |
| **Status** | 📝 draft · notebook ready |

## Why it matters

Language models can now call tools. `strands-robots` (AWS Strands Labs, Apache-2.0) gives a simulated SO-100 arm to an LLM agent as **a single tool**. Inside this tool there are 77 actions ("Actions (77 total)"). With a few lines of code you can tell a robot "pick up the red cube". The question is: when the agent says "done", has the cube really been lifted?

## Concept

The agent loop: **think → call tool → observe → correct**. If the observation step is skipped or ignored, the agent does not notice it was wrong. In this lesson we read the result not from the agent's sentence but from the simulation's measurement (the cube's height).

## Experiment (in the notebook)

- The scene is the same as the README example: `Robot("so100")`, a 5 cm red cube, front camera.
- Agent model: local `qwen3.5:9b` via Ollama (Mac, no API key). Prompt: "pick up the red cube" and its Turkish version.
- Every tool call, the joint angles and the cube position are logged.

## Results in numbers

6 trials with `qwen3.5:9b`:

| Outcome | Trials |
|---|---|
| Cube lifted (52.15 cm and 35.08 cm) | 2 |
| Agent said "I succeeded" while the cube was on the table | 1 |
| Crashed (tool call error) | 1 |
| Gave up / kept calling the fake policy | 2 |

With `qwen2.5:7b`, no tool was called in 1 trial.

Anatomy of the successful run (`en-qwen35-2`): **16 tool calls** (3 of them errors). On closing, the tool said "Closed on nothing"; at second 171 the agent read the cube's height (2.49 cm), saw the error, went back down, attached the cube to the gripper with `attach_bodies` and lifted it to 50 cm. Simulation time advanced **5.91 s**, wall-clock time was **351 s**: the robot moved for ~6 seconds, the model thought for ~6 minutes.

## Limits

- **The grasp in the sim is not real.** The gripper presses on the cube with 29.80 N, but friction carries at most 0.00 N of the cube's 0.98 N weight. The agent "glues" the cube to the gripper with `attach_bodies(mode="weld")`. This is a **grasp assist**, not a physical grasp; it has no counterpart on a real arm. The weld was called by the agent, not the library.
- `strands-robots` 0.5.3 raises a `TypeError` on the `attach_bodies(weld)` call with MuJoCo 3.17 (`mjNEQDATA` 11 → 12). The notebook is pinned to MuJoCo 3.16.0.
- The runs used a local 9B model, not a strong cloud model. A 6-trial rate is an estimate; the confidence interval is wide.

## Exercises

1. **Easy.** Run the same prompt 5 times. In how many is the cube lifted, and in how many does the agent wrongly say "I succeeded"?
2. **Medium.** Add "after lifting, check the cube's height with `get_body_state`" to the system prompt. Do the success rate and the false-claim rate change?
3. **Hard.** Hide `attach_bodies` from the agent (remove it from the tool list). What does the agent do? What does this mean for a real arm?

## Sources

- [`strands-robots` (PyPI)](https://pypi.org/project/strands-robots/) (Apache-2.0) and the README example.
- [Strands Agents](https://strandsagents.com/).
- [Hugging Face Robotics Course](https://huggingface.co/learn/robotics-course).
- Numbers from the Koddan Kasa 32 run logs and the internal verification note (25.10.2026).
