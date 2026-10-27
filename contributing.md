# Contributing

This book is open to contributions: open an issue or a pull request on GitHub.

## Fixing a number

Every number in a lesson must be reproducible. If one doesn't match: name the lesson, your library versions and your output. A number that doesn't reproduce is a bug.

## Writing or improving a lesson

1. Read the authoring guide: [`tools/AUTHORING.md`](tools/AUTHORING.md) (template, honesty rules, Turkish rules).
2. Lesson structure comes from [`curriculum.yaml`](curriculum.yaml); after changing it run `uv run python tools/build_book.py`, which regenerates `SUMMARY.md`, part pages and each lesson's metadata block. Don't edit those generated parts by hand.
3. Write the experiment in `experiments/NN-slug/experiment.py` (English code, fixed seeds, prints a table) and save its exact stdout as `output.txt`.
4. Write the lesson in English (`part-…/NN-slug.md`) and Turkish (`tr/part-…/NN-slug.md`). In Turkish, technical terms stay in English.
5. Check that every relative link resolves and the site builds: `cd site && npm run build`.

## Code conventions

- English identifiers, comments and output.
- Dependencies in `pyproject.toml`, pinned in `uv.lock` (`uv add <pkg>`, then `uv export --no-hashes --no-dev --format requirements-txt > requirements.txt`).
- Before a pull request: `uv run ruff check .` and `uv run python tools/check_book.py`.
- Robot models through `experiments/common/menagerie.py`.
