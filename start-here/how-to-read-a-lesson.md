# How to read a lesson

Every lesson has the same shape, so you always know where the theory, the code and the evidence are.

## Metadata

| Row | Meaning |
|---|---|
| **Part** | The part the lesson belongs to |
| **Run type** | **CPU** runs on a Mac or Colab CPU · **GPU** needs a Colab GPU · **ART** uses results we ship (logs, checkpoints) and runs on CPU · **HW** needs the real SO-101 |
| **Prerequisites** | Earlier lessons of this book you need |
| **Learn first** | Lessons of [AI Engineering from Scratch](https://aiengineeringfromscratch.com/index.html) that cover the general background (math, ML, deep learning, RL) |
| **Time** | Reading and running, roughly |
| **Code** | The experiment script (and notebook, if any) |
| **Status** | ✅ done · 📝 draft · 🚧 coming soon |

## Sections

| Section | What it gives you |
|---|---|
| **The problem** | One concrete failure or question the lesson resolves |
| **Core idea** | The minimum theory: definitions and a few equations |
| **Experiment** | Predict, run, observe. Short code, then a table of measured numbers |
| **Break it** | One parameter pushed until the method fails, and why |
| **On the real robot** | What changes on hardware, with sourced facts only |
| **Check yourself** | Three exercises, easy to hard, with answers |
| **Takeaways and sources** | Three bullets, the next lesson, primary sources |

## Every experiment is a sweep

One parameter changes and one quantity is measured, so the result is a table, not a single anecdote. The exact output of each run is saved next to its script as `output.txt`; the numbers in the lesson come from there.
