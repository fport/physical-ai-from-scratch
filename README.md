# Physical AI Engineering from Scratch

An open-source course book that starts with a laptop and ends with a robot arm doing a task on its own. Every lesson rests on an experiment that was actually run. Every number comes from that experiment or from a linked source.

> 40 lessons · 10 phases · 8 appendix lessons. Python, MuJoCo, LeRobot. First on your own Mac, then on a real arm.

**[Setup](getting-started/setup.md)** · **[How to read a lesson](getting-started/how-to-read-a-lesson.md)** · **[Reading order](getting-started/reading-order.md)** · **[First lesson: why did the cube go through the table?](phase-0-world-frame-by-frame/02-timestep.md)** · **[Türkçe](tr/README.md)**

## What is this book?

Physical AI is AI where the code you write moves a motor, a joint, a "muscle". This book builds it from the bottom up: first how a physics engine computes the world, then where an arm is in space and how its motors turn, then how a robot sees, how it learns from demonstrations and from trial, and finally how you get work done by talking to it through a language model.

Every lesson asks the same question: **which muscle does this line of code move?**

## Who is it for?

- Anyone who can read Python and wants to start robotics from scratch.
- High-school physics (F = m·a, torque) and a little linear algebra help, but lessons re-explain them when needed.
- The first four phases need **only a computer**. The real arm (SO-101) joins from lesson 6 as a "human layer"; readers without an arm can run the simulated counterpart of every lesson.

## Map

| Phase | Name | Question | Lessons |
|---|---|---|---|
| 0 | [World, frame by frame](phase-0-world-frame-by-frame/README.md) | How does a physics engine compute the world? | 01–04 |
| 1 | [Space and body](phase-1-space-and-body/README.md) | Where is the arm, and where does it see things? | 05–09 |
| 2 | [Muscle](phase-2-muscle/README.md) | How does code move a motor? | 10–14 |
| 3 | [Senses](phase-3-senses/README.md) | What does the robot see and feel? | 15–18 |
| 4 | [Learning from demonstration](phase-4-learning-from-demonstration/README.md) | Show it, and it learns. | 19–26 |
| 5 | [Agents](phase-5-agents/README.md) | Getting work done by talking | 27–31 |
| 6 | [Learning by trial](phase-6-learning-by-trial/README.md) | You write the reward. | 32–34 |
| 7 | [Sim to real](phase-7-sim-to-real/README.md) | It worked in sim. Why not on the table? | 35–37 |
| 8 | [Big models](phase-8-big-models/README.md) | Robot foundation models | 38–39 |
| 9 | [Capstone](phase-9-capstone/README.md) | Say → see → plan → grasp | 40 |
| — | [Appendix](appendix/README.md) | Deep dives and topics that don't run on a Mac | A–H |

## Status

The book is being written. Every lesson states its status at the top:

| Mark | Meaning | Lessons |
|---|---|---|
| ✅ done | Text, experiment, numbers and exercises are ready | 01, 02, 04 |
| 📝 draft · notebook ready | The experiment was run; notebook and verified numbers exist; the text is not finished | 11, 19, 24, 31 |
| 🚧 coming soon | Skeleton: learning goal, lesson question, planned experiment, sources. No results | the rest |

## How to study

1. **Read.** Each lesson opens with something that works in the real world (a robot dog's training code, a factory arm, a paper), then builds the concept.
2. **Run.** Each lesson has an experiment: `python experiments/.../experiment.py` on a Mac, or the notebook on Colab. Experiments are **sweeps**: one number changes, one result is measured.
3. **Change.** The exercises ask you to change that number. Predict first, then run.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python experiments/02-timestep/experiment.py
```

To open the book as a website locally: `cd site && npm install && npm run dev` ([site/README.md](site/README.md)).

## Videos

This book is the long-form companion of the **Koddan Kasa · Physical AI from Scratch** short-video series (Turkish) on Instagram, TikTok and YouTube Shorts. Each video summarizes one lesson and links to it.

## Honesty

- Every number in the videos and in the book comes either from an experiment we ran (with version, seed and result file) or from a linked source. Rules: [Numbers and sources](getting-started/numbers-and-sources.md).
- We never present something we didn't run as if we ran it. Lessons that need a GPU or hardware say so up front.
- Teleoperation and replay are not learning; the book dedicates a lesson to that difference ([lesson 19](phase-4-learning-from-demonstration/19-replay.md)).

## Acknowledgements

The structure was inspired by Rohit Ghumare and contributors' [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) curriculum (MIT): numbered lessons grouped into phases, a "problem → concept → build it → use it → exercises → key terms → sources" flow in every lesson, and the "raw version before the framework" principle. No text or code was copied; this is a new curriculum written specifically for physical AI. The sources behind the curriculum are in [References](references.md).

## License

Not chosen yet.
