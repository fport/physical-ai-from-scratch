# How to read a lesson

Every lesson follows the same skeleton. Wherever you enter, you know where you are.

## Metadata

The table at the top of the page: phase, type, prerequisites, time, tools, code and **status** (✅ done · 📝 draft · 🚧 coming soon).

## Sections

| Section | What it does |
|---|---|
| **Why it matters** | Opens with something that works in the world: one number from a robot dog's open training code, a spec of a factory arm, a paper's starting point. Every claim has a source. |
| **Concept** | Intuition without code: an analogy, a table, a two-line formula. |
| **Build it** | The experiment step by step. Each code block runs on its own. The full script lives in `experiments/`, the notebook in `notebooks/`. |
| **Results in numbers** | The actual output of the experiment: version, hardware, table. The answer to the lesson question is here. |
| **Limits** | What this result does not show. Where simulation really departs from reality, what we didn't measure. |
| **Exercises** | Easy → medium → hard. Most ask you to change a single number in the experiment. Predict first, then run. |
| **Key terms** | Term · what people say · what it actually means. |
| **Sources** | Primary sources and where the numbers on this page come from. |

## Every experiment is a sweep

One parameter changes, one number is measured. In lesson 02 the timestep is swept from 2 to 50 ms, in lesson 04 the ramp angle in 0.05° steps, in lesson 24 the number of demonstrations from 10 to 500. The lesson's twist comes out of that sweep's data: the cube stays at 12 ms but goes through at 10; a cube 100× heavier slips at the same angle; more demonstrations increase collisions.

## Result file

Notebooks write their results to `results/NN/result.json`. Numbers in the book and in the videos are read from this file.
