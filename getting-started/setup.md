# Setup: one Mac, zero cost

Every experiment in the first four phases runs on a laptop or on free Colab. No GPU, robot or paid account needed.

## Option A · Your own Mac (recommended)

The numbers in the book were produced on an Apple Silicon Mac with Python 3.12, MuJoCo 3.17.0 and NumPy 2.5.

```bash
git clone https://github.com/fport/physical-ai-from-scratch.git
cd physical-ai-from-scratch
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python experiments/02-timestep/experiment.py
```

The first line of the output should be `mujoco 3.17.0`. With another version the last digits may shift slightly; every experiment notes this.

## Option B · Colab

Each lesson's notebook lives in `notebooks/`. Its first cell installs its own dependencies (`pip install mujoco==3.17.0`). Notebooks that need a Menagerie model (Go2, SO-101) clone that folder themselves.

> The notebooks were written for the Turkish video series, so their text is Turkish; the code runs the same.

> "Open in Colab" badges will be added soon.

## What each lesson needs

| Tag | Meaning | Where |
|---|---|---|
| **MJ** | MuJoCo, locally on a Mac or Colab CPU | most of Phases 0–3 |
| **LR** | LeRobot | from Phase 4 |
| **ARM** | Real SO-101 | human layer from 06; main experiment in 10, 14, 19, 26, 36, 40 |
| **CO-GPU** | Colab GPU or rented GPU | 26 (training), 33–34, Appendix E |
| **OL** | Local language model via Ollama | Phase 5 |

What can't be done on a Mac is marked honestly: Isaac Sim/Lab is not supported on macOS ([requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)); π0 inference needs a > 8 GB NVIDIA GPU ([openpi](https://github.com/Physical-Intelligence/openpi)). The book covers these only through their sources ([Appendix G](../appendix/g-big-robot-models.md)).

## If something goes wrong

- `ModuleNotFoundError: mujoco` → the virtual environment isn't active: `source .venv/bin/activate`.
- Lesson 31 (Strands) needs MuJoCo **3.16.0**; install it in a separate virtual environment (the lesson explains why).
