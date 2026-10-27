---
name: start-learning
version: 1.0.0
description: >
  Onboarding for the Physical AI Engineering from Scratch course (44 lessons on
  simulation, kinematics, control, perception, planning, imitation learning, RL,
  sim-to-real and robot foundation models). Asks about goals and background, runs a
  short placement quiz, checks the local setup, and writes PHYSICAL-AI-LEARNING.md,
  the study plan the `learn` skill follows. Trigger phrases: "start learning",
  "start the physical AI course", "make me a study plan", "fiziksel AI öğrenmek
  istiyorum", "çalışma planı çıkar".
tags: [onboarding, robotics, physical-ai, learning-plan, mujoco]
---

# Start learning: Physical AI Engineering from Scratch

You are setting a learner up for the course **Physical AI Engineering from Scratch**
(https://fport.github.io/physical-ai-from-scratch/, source: https://github.com/fport/physical-ai-from-scratch).
The course is 44 lessons in 10 parts plus a capstone. Every lesson has a runnable
experiment (`experiments/<slug>/experiment.py`) whose exact output is saved as
`output.txt`. General math, ML and RL background is not taught in the course; each
lesson links matching lessons of AI Engineering from Scratch under "Learn first".

Your output is one file in the current directory: `PHYSICAL-AI-LEARNING.md`. Every
`learn` session reads and updates it.

If your host has a tool for multiple-choice questions, use it; otherwise ask with
lettered options and wait for the answer. Ask one question at a time.

## 0. Existing plan?

If `PHYSICAL-AI-LEARNING.md` already exists, show its current lesson and progress,
and ask whether to continue (hand over to `learn`) or start over. Don't overwrite it
without a yes.

## 1. Find the course files

Look, in order, for `curriculum.yaml` in: the current directory, `./physical-ai-from-scratch/`,
`~/physical-ai-from-scratch/`. If found, that is the course root.

If not found, offer three options:
- **A. Clone it** (recommended, lets you run the experiments):
  `git clone https://github.com/fport/physical-ai-from-scratch.git` then `cd physical-ai-from-scratch && uv sync`
  (needs [uv](https://docs.astral.sh/uv/); uv installs Python 3.14 itself).
- **B. Read only:** use the raw files at
  `https://raw.githubusercontent.com/fport/physical-ai-from-scratch/main/<path>`.
- **C. Website only:** https://fport.github.io/physical-ai-from-scratch/ (Turkish under `/tr/`).

Record the choice in the plan.

## 2. Interview (short)

1. **Language:** English or Türkçe? Turkish pages live under `tr/` with the same file names.
   In Turkish, technical terms stay in English (timestep, policy, Jacobian, …); talk to the
   learner the same way.
2. **Goal** — pick one:
   - A. The full course, in order
   - B. Manipulation: make an arm pick things up
   - C. Get to a first learned policy fast
   - D. Legged locomotion
   - E. Classical control and planning
   - F. I know ML; I need the robotics part
   - G. Language models and robots
3. **Background** (multi-select): Python · NumPy · linear algebra · calculus · probability ·
   PyTorch · RL basics · control theory · ROS · real robot hardware.
4. **Hardware:** Do they have an SO-101 arm (follower, and a leader for teleoperation)?
   Most lessons run in simulation; lesson 40 and the "On the real robot" sections need hardware.
5. **Time:** hours per week.

## 3. Placement quiz (10 questions, ~5 minutes)

Ask them one by one, multiple choice, no hints. Keep a score per part. Don't reveal the
correct answer until the end, then explain each miss in one sentence with the lesson link.

1. *(Part 1)* In a physics simulation, why can a fast cube pass straight through a thin table?
   A) floating-point error B) collisions are only checked at the discrete time steps ✔ C) the table has no mass D) gravity is too strong
2. *(Part 2)* For a rotation matrix $R$, what is $R^{-1}$?
   A) $-R$ B) $R^\top$ ✔ C) $\det(R)\,R$ D) it has no inverse
3. *(Part 3)* What does a manipulator Jacobian map?
   A) joint angles → tip position B) joint velocities → tip velocity ✔ C) torques → angles D) pixels → angles
4. *(Part 4)* A proportional (P-only) controller holds an arm out against gravity. What do you see?
   A) it reaches the target exactly B) it sags below the target ✔ C) it oscillates forever D) it moves the wrong way
5. *(Part 5)* A moving average over $n$ samples taken every $\Delta t$ delays the signal by about…
   A) $n\,\Delta t$ B) $\frac{n-1}{2}\Delta t$ ✔ C) $\Delta t/n$ D) nothing
6. *(Part 6)* What does model predictive control do at every control step?
   A) replays a stored trajectory B) re-solves a short-horizon optimization from the current state ✔ C) retrains a network D) samples a random action
7. *(Part 7)* A policy succeeds 9 times out of 10. Which statement is honest?
   A) "it is 90% reliable" B) "the success rate is likely between roughly 60% and 98%" ✔ C) "it never fails" D) "it fails 10% of the time"
8. *(Part 7)* A behavior-cloning policy makes a small mistake and ends up in states it never saw in the demos. The name and a classic fix?
   A) overfitting / dropout B) distribution shift / DAgger ✔ C) vanishing gradient / ReLU D) aliasing / filtering
9. *(Part 8)* An RL episode ends because of a time limit, not because the task failed. What should the value target do?
   A) treat it as a failure (value 0) B) bootstrap from the next state's value ✔ C) ignore the whole episode D) double the reward
10. *(Part 9–10)* Why train a policy across randomized friction, mass and delay?
   A) to speed up the simulator B) so the real robot looks like just another variant ✔ C) to reduce the dataset size D) to make training deterministic

## 4. Choose the start and the path

Paths (lesson numbers):

| Goal | Path |
|---|---|
| A Full course | 01 → 44 |
| B Manipulation | 01–05 · 06–07 · 08–10 · 11–16 · 19–21 · 25–31 · 40 |
| C First learned policy | 01 · 02 · 05 · 07 · 08 · 10 · 12 · 13 · 16 · 25 · 26 · 27 · 28 |
| D Locomotion | 01–05 · 11 · 12 · 25 · 32–36 · 37–39 |
| E Control and planning | 01–16 · 22–24 |
| F From ML | 01 · 02 · 05 · 06 · 07 · 08 · 09 · 10 · 11 · 12, then any part |
| G Language models and robots | 08 · 10 · 15 · 16 · 21 · 41 · 42 |

Skip a part's lessons only if the learner answered that part's quiz questions correctly
**and** said they know the topic; mark skipped lessons "skipped (placement)" so they can
return. When a lesson's "Learn first" topics are missing from the learner's background,
add the AI Engineering from Scratch links next to that lesson in the plan.

Lesson titles, slugs and parts come from `curriculum.yaml` (or `SUMMARY.md` / `tr/SUMMARY.md`).
Page path: `<part dir>/<slug>.md` (English) or `tr/<part dir>/<slug>.md` (Turkish); website:
`https://fport.github.io/physical-ai-from-scratch/<part dir>/<slug>/` (Turkish: `/tr/` prefix).

## 5. Check the setup (only with a local clone)

Ask before running anything. Then:

```bash
uv sync
uv run python experiments/02-timestep/experiment.py | head -3
```

The first line should start with `mujoco 3.17.0`. If uv is missing, point to
https://docs.astral.sh/uv/ ; without uv, `pip install -r requirements.txt` in a venv works too.

## 6. Write PHYSICAL-AI-LEARNING.md

```markdown
# Physical AI learning plan

- Learner goal: <goal>
- Language: <en|tr>
- Course source: <local path | raw GitHub | website>
- Hardware: <none | SO-101 follower | follower + leader>
- Time per week: <hours>
- Started: <YYYY-MM-DD>

## Placement
- Score: <k>/10 — strong: <parts>, review: <parts>

## Path
- [ ] 01 · The world as text: MJCF and URDF — part-1-simulation/01-world-as-text.md
- [ ] 02 · Time integration and the timestep — part-1-simulation/02-timestep.md
- … (every lesson on the chosen path, in order; skipped ones marked)

## Current
- Next lesson: <NN · title>

## Log
- <YYYY-MM-DD>: plan created
```

Then tell the learner how to start the first lesson with the `learn` skill, using their
host's syntax: Claude Code `/learn`, Codex `learn` (or pick it from `/skills`), other hosts
"Use the learn skill to start my next lesson." If you don't know the host, give the
natural-language form.
