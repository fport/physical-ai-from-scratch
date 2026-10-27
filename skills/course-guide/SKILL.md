---
name: course-guide
version: 1.0.0
description: >
  Answers questions about the structure of Physical AI Engineering from Scratch: where a
  topic is taught, what to read before a lesson, which lessons need a GPU or the real
  SO-101 arm, how to run an experiment, and how lessons map to the Koddan Kasa videos.
  Trigger phrases: "where is inverse kinematics taught", "what do I need before lesson
  33", "which lessons need hardware", "IK hangi derste", "bu derse nereden başlamalıyım".
tags: [navigation, curriculum, robotics, physical-ai]
---

# Course guide

Help the learner find their way in **Physical AI Engineering from Scratch**
(https://fport.github.io/physical-ai-from-scratch/, https://github.com/fport/physical-ai-from-scratch).

## Sources of truth (read before answering)

- `curriculum.yaml` — parts, lessons (number, slug, English and Turkish titles), run type
  (CPU / GPU / ART / HW), time, prerequisite lessons and AI Engineering from Scratch links.
- `SUMMARY.md` / `tr/SUMMARY.md` — the table of contents.
- `glossary.md` / `tr/glossary.md` — term → lesson.
- `start-here/reading-paths.md`, `start-here/prerequisites.md`, `start-here/video-map.md`.
- The lesson pages themselves for anything about content.

Use a local clone when present (look for `curriculum.yaml` in the current directory,
`./physical-ai-from-scratch/` or `~/physical-ai-from-scratch/`); otherwise read the raw
files from `https://raw.githubusercontent.com/fport/physical-ai-from-scratch/main/<path>`.

## How to answer

- **"Where is X taught?"** — search the glossary, then lesson titles, then lesson text. Give
  the lesson number, title and link (website: `https://fport.github.io/physical-ai-from-scratch/<part dir>/<slug>/`,
  Turkish with `/tr/`).
- **"What do I need before lesson N?"** — follow `prereq` in `curriculum.yaml` recursively and
  list the chain in order, plus the lesson's "Learn first" links for background.
- **"Which lessons need a GPU / the real arm?"** — filter by `run` (GPU, ART, HW).
- **"How do I run lesson N?"** — `uv sync` once, then
  `uv run python experiments/<slug>/experiment.py`; mention runtime notes from the lesson.
- **Videos** — use `start-here/video-map.md`.

Answer briefly, with links. If the course doesn't cover something, say so and, if relevant,
point to the closest lesson or to AI Engineering from Scratch for general ML/math topics.
Hand over to `learn` when the learner wants to study a lesson, and to `start-learning` when
they want a plan.
