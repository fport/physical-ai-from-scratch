# Setup

Parts 1–7 and most of 8–10 run on a laptop. The numbers in the book were produced on an Apple Silicon Mac (Apple M5) with Python 3.14 and the exact versions locked in `uv.lock` (MuJoCo 3.17.0, NumPy 2.5.4, SciPy, PyTorch 2.14, Gymnasium, OpenCV, Matplotlib).

## Install

The repository is a [uv](https://docs.astral.sh/uv/) project: `pyproject.toml` lists the dependencies, `uv.lock` pins every version, and `.python-version` pins Python 3.14 (uv downloads it if you don't have it).

```bash
git clone https://github.com/fport/physical-ai-from-scratch.git
cd physical-ai-from-scratch
uv sync
uv run python experiments/02-timestep/experiment.py
```

Without uv: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt` (the file is exported from `uv.lock`).

The first line of every experiment prints the library versions. With other versions the last digits of some results may differ.

## Robot models

Experiments that use the SO-101, the Franka Panda or the Unitree Go2 load them from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) through `experiments/common/menagerie.py`, which clones the three model folders once into `.cache/mujoco_menagerie` at a pinned commit.

## Colab

Notebooks in `notebooks/` install their own dependencies in the first cell. Lessons with run type **GPU** say which Colab runtime they need.

## What a Mac can't do

Some tools are not available on macOS and are covered from their sources only: Isaac Sim/Lab requires Linux or Windows ([requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)); π0 inference needs an NVIDIA GPU with more than 8 GB ([openpi](https://github.com/Physical-Intelligence/openpi)).

## Read the book as a website

```bash
cd site
npm install
npm run dev
```
