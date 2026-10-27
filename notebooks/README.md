# Notebooks

Each experiment's notebook runs on Colab or local Jupyter. The first cell installs dependencies; the last cell writes the result to `results/NN/result.json`.

| Notebook | Lesson | Needs | Note |
|---|---|---|---|
| [01-world-as-text.ipynb](01-world-as-text.ipynb) | [01 · The world is a text file](../phase-0-world-frame-by-frame/01-world-as-text.md) | MuJoCo 3.17, CPU | |
| [02-timestep.ipynb](02-timestep.ipynb) | [02 · Timestep](../phase-0-world-frame-by-frame/02-timestep.md) | MuJoCo 3.17, CPU | |
| [04-friction.ipynb](04-friction.ipynb) | [04 · Friction](../phase-0-world-frame-by-frame/04-friction.md) | MuJoCo 3.17, CPU | |
| [11-go2-two-numbers.ipynb](11-go2-two-numbers.ipynb) | [11 · 12 joints, two numbers](../phase-2-muscle/11-pd-control.md) | MuJoCo, Menagerie `unitree_go2` (cloned by the notebook) | |
| [19-replay.ipynb](19-replay.ipynb) | [19 · Replay is not learning](../phase-4-learning-from-demonstration/19-replay.md) | MuJoCo 3.17, Menagerie `robotstudio_so101` (cloned by the notebook) | |
| [24-two-right-answers.ipynb](24-two-right-answers.ipynb) | [24 · Two right answers](../phase-4-learning-from-demonstration/24-two-right-answers.md) | NumPy, CPU (~1 min) | |
| [31-strands-agent.ipynb](31-strands-agent.ipynb) | [31 · Strands](../phase-5-agents/31-strands-agent.md) | `strands-robots` 0.5.3, MuJoCo **3.16.0**, Ollama or Claude API | Labelled "Levha 32" inside |

> The notebooks were written for the Turkish Koddan Kasa video series and copied here as they are, so their text is Turkish. They contain phrases like "videodaki deney" (the experiment in the video), "Levha NN" (the video's label) and references to scripts in the video repo (`src/v6/...`); the experiment itself is complete inside each notebook.
