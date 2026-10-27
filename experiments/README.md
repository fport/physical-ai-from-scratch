# Experiments

One folder per lesson: `NN-slug/experiment.py` runs the lesson's sweep and prints a table; `output.txt` is the exact output of the run the lesson's numbers come from.

```bash
uv sync                                  # once, at the repo root (see start-here/setup.md)
uv run python experiments/02-timestep/experiment.py
diff <(uv run python experiments/02-timestep/experiment.py) experiments/02-timestep/output.txt
```

- Robot models (SO-101, Franka Panda, Unitree Go2) are loaded through `common/menagerie.py` from a pinned MuJoCo Menagerie commit.
- Deterministic experiments reproduce `output.txt` byte for byte; lines that report wall-clock time vary with the machine and are labelled as such.
- Some experiments import helpers from earlier lessons (for example `ik_solve` from `10-inverse-kinematics`, `scripted_pick` from `16-grasping-and-pick-place`, `ppo.py` from `33-ppo`).
- Most scripts finish in seconds to a few minutes on a laptop CPU; the reinforcement-learning sweeps in Part 8 take longer and say so in their lessons.
