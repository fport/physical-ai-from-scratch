# Contributing

This book is being written and will be open to contributions. This page will be updated once the repo is public.

## Finishing a lesson

To turn a skeleton lesson (🚧) into a finished one:

1. **Run the experiment.** Write the sweep from the lesson's "Planned experiment" section; try at least 5 values. Print versions, fix the seed if there is one, and write the result to `results/NN/result.json`.
2. **Write the page following the template:** [How to read a lesson](getting-started/how-to-read-a-lesson.md). Finished examples: [01](phase-0-world-frame-by-frame/01-world-as-text.md), [02](phase-0-world-frame-by-frame/02-timestep.md), [04](phase-0-world-frame-by-frame/04-friction.md).
3. **Give every claim a source.** [Numbers and sources](getting-started/numbers-and-sources.md).
4. **Add the code:** a script that runs on a Mac at `experiments/NN-slug/experiment.py`, a Colab notebook at `notebooks/NN-slug.ipynb`.
5. Update the status in the metadata table, and keep [SUMMARY.md](SUMMARY.md) and the phase page in sync.
6. **Both languages.** English is the default (repo root); the Turkish version lives under [`tr/`](tr/README.md) with the same file names. In Turkish, technical terms stay in English.

## Code conventions

- Code, identifiers and comments in English.
- Pinned dependency versions (`mujoco==3.17.0`).
- One script runs one sweep and prints the result as a table.

## Reporting a bug

If a number doesn't match: tell us the lesson, the versions you used and your output. Every number in the book must be reproducible; a number that doesn't match is a bug.
