# Experiments

Scripts that run directly on a Mac (or any computer). Each one runs a lesson's sweep and prints the result as a table.

```bash
pip install -r ../requirements.txt
python 02-timestep/experiment.py
python 01-world-as-text/experiment.py
python 04-friction/experiment.py       # ~10 s
```

| Script | Lesson | Time (Apple M5) |
|---|---|---|
| [01-world-as-text/experiment.py](01-world-as-text/experiment.py) | [01](../phase-0-world-frame-by-frame/01-world-as-text.md) | < 1 s |
| [02-timestep/experiment.py](02-timestep/experiment.py) | [02](../phase-0-world-frame-by-frame/02-timestep.md) | < 1 s |
| [04-friction/experiment.py](04-friction/experiment.py) | [04](../phase-0-world-frame-by-frame/04-friction.md) | ~10 s |

Numbers were produced with MuJoCo 3.17.0, NumPy 2.5.4 and Python 3.12, and match the lesson tables exactly.
