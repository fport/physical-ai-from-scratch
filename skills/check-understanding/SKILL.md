---
name: check-understanding
version: 1.0.0
description: >
  Quizzes the learner on one part or lesson of Physical AI Engineering from Scratch,
  using questions built from the lesson pages and their experiment results, grades the
  answers and records the result in PHYSICAL-AI-LEARNING.md. Trigger phrases: "quiz me
  on part 4", "check my understanding of lesson 18", "test me", "beni sına",
  "bölüm 7 quiz".
tags: [quiz, assessment, robotics, physical-ai]
---

# Check understanding

Quiz the learner on a part (e.g. "Part 4") or a single lesson of **Physical AI
Engineering from Scratch**. Find pages as described in `learn` (local clone, raw GitHub
or website); use the plan's language if `PHYSICAL-AI-LEARNING.md` exists.

## Build the quiz

Read every lesson page in scope and its `experiments/<slug>/output.txt`. Write 5 questions
for a single lesson, 8 for a part, mixing:

- **Concept** — "Why does X happen?" (from "Core idea").
- **Read the result** — show a row or two of a results table and ask what it means
  (from "Experiment").
- **Predict the sweep** — "If we doubled Y, what would happen to Z?", answerable from the
  lesson's tables or reasoning.
- **Failure mode** — from "Break it".
- **Real robot** — what changes on hardware, from "On the real robot".

Every correct answer must be supported by the page or `output.txt`. Don't use numbers that
aren't there. Prefer short free-text answers; use multiple choice for numeric reading.

## Run it

Ask one question at a time, wait, then give one or two sentences of feedback with the lesson
section to reread when the answer is off. At the end: score, the two weakest topics, and the
exact lessons/sections to review.

## Record it

If `PHYSICAL-AI-LEARNING.md` exists, add to its "Log":
`- YYYY-MM-DD: quiz <scope> — k/n; review: <lessons>`.
