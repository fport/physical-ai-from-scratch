# Authoring guide (for lesson writers)

The book teaches the **robotics-specific** part of physical AI, in strict engineering order, rigorously and without padding. General math / ML / DL / RL basics / transformers / diffusion / MCP are **not re-taught**: the lesson's metadata already links the right [AI Engineering from Scratch](https://aiengineeringfromscratch.com/index.html) (AIEFS) lessons under "Learn first". In the text, say "(see AIEFS: <title>)" with the link when you lean on such a concept, then move on.

## Files you own for lesson NN

| File | What |
|---|---|
| `part-X-…/NN-slug.md` | English lesson |
| `tr/part-X-…/NN-slug.md` | Turkish lesson (same structure, same numbers) |
| `experiments/NN-slug/experiment.py` | Runnable experiment (English code and comments) |
| `experiments/NN-slug/output.txt` | The exact stdout of your final run (`python experiment.py > output.txt`) |
| `experiments/NN-slug/*.png` | Optional: at most one plot, if it truly helps (matplotlib, Agg backend) |

**Do not edit** the first `# NN · Title` line or anything between `<!-- meta -->` and `<!-- /meta -->`: `tools/build_book.py` regenerates them from `curriculum.yaml`. Do not edit `curriculum.yaml`, `SUMMARY.md`, part `README.md` files or any file outside your lessons. Do not run git commands that commit, push, reset, stash or switch branches.

## Lesson body (after the meta block)

English headings / Turkish headings, in this order:

1. A one-line motto as a blockquote (`> …`) right after the meta block: the single idea of the lesson.
2. `## The problem` / `## Problem` — one concrete failure or question this lesson resolves (a few sentences, a real system if possible, sourced).
3. `## Core idea` / `## Ana fikir` — written for a software engineer from another field, in layers (see "Explaining for engineers" below). Math in `$…$` (inline) and `$$…$$` (display). No derivations that AIEFS already covers.
4. `## Experiment` / `## Deney` — predict → run → observe. Show the essential code (short blocks, the full script lives in `experiments/`), then **a results table with your measured numbers**. State MuJoCo/NumPy/PyTorch version and that it ran on an Apple Silicon Mac (Apple M5).
5. `## Break it` / `## Boz` — push one parameter until the method fails; show the measured failure and explain the mechanism.
6. `## On the real robot` / `## Gerçek robotta` — what changes on the SO-101 / Go2 / a real arm. Only sourced facts (link the primary source). Anything not measured on hardware is said to be unmeasured.
7. `## Check yourself` / `## Kendini dene` — 3 exercises (easy → hard). Answers at the end under `### Answers` / `### Cevaplar` (numbers only if you actually computed them).
8. `## Takeaways and sources` / `## Akılda kalanlar ve kaynaklar` — 3 bullets, "Next: lesson NN+1", then a source list (primary sources: papers, official docs, code).

Target: **1,200–1,800 words** per language, tight technical prose. No hype, no video-style hooks, no rhetorical filler, no emoji. Prefer tables and equations over paragraphs. Every lesson must teach something a robotics engineer actually uses.

## Explaining for engineers

The reader is a good programmer who has not studied control, robotics or signal processing. Every Core idea follows the same layers; the reader can stop after any layer and still have understood something.

1. `### In plain words` / `### Sade anlatımla` — an everyday analogy and the idea in 1–3 short paragraphs, no symbols.
2. A diagram when the idea has structure (a signal chain, a control loop, a frame tree, a pipeline, a state machine): a Mermaid code block (```` ```mermaid ````), `flowchart TD` (top-down; wide left-to-right charts get too small), at most ~8 nodes, labels in the page's language.
3. `### … as code` / `### … kod olarak` — the formula written as short Python a programmer can read, with comments for every variable. The code may be simplified pseudocode; the full script stays in `experiments/`.
4. A small table of the key formulas, and one sentence that defines every symbol right after it.
5. `<details><summary>Math details: …</summary> … </details>` (TR: `Matematik ayrıntısı: …`) — derivations, assumptions, edge cases and a "Go deeper" line with AIEFS / primary-source links. Leave a blank line after `<summary>…</summary>` and before `</details>` so Markdown and math render inside.

Figures, preferred way: a small `experiments/NN-slug/plot.py` that parses that lesson's `output.txt` and saves the PNG next to it (no re-run of the experiment needed, so long experiments stay untouched). Alternative: when a curve tells the story better than a table (an error vs. parameter trade-off, a trajectory, a learning curve), the experiment script also saves one PNG next to itself (matplotlib Agg, ~9×3.4 in, 150 dpi, log axes when values span decades) **without changing its printed output**, and the lesson embeds it with descriptive alt text: `![…](../experiments/NN-slug/figure.png)`. Keep the table too: the numbers in the text must still come from `output.txt`.

## Interactive figures and quizzes (website only)

Both are invisible on GitHub (an HTML comment / a separate data file) and come alive on the website. Reference implementation: lesson 12 (`site/src/figures/12-pd-step.ts`, `quizzes/12-joint-control.yaml`).

**Interactive figure.** Only where touching a parameter teaches something the static table cannot (a trade-off, a threshold, a geometry). One per lesson at most.
- Module `site/src/figures/NN-name.ts` (NN = lesson number), `export default function (root, lang)`. Build the box with `frame()` from `./_kit` (title, caption, canvas, sliders/checkboxes/selects/buttons, readouts); animate with `animate()` (runs only while on screen); random numbers with `rng(seed)`; charts with `axes()`. Colors only from `colors()` (works in light and dark). No external libraries. Every visible string has `en` and `tr`.
- Marker on its own line in both language pages, same place: `<!-- figure: NN-name -->`, usually right after the Core idea's `</details>` or inside Break it. `site/scripts/sync.mjs` turns it into the slot.
- Honesty: it is a toy model in the browser and `frame()` says so. Use the lesson's parameters where you can (masses, gains, ranges), but never print a readout as if it were a measured result, and don't quote figure readouts in the lesson text. Cheap enough to run at 60 fps on a laptop.
- Canvas text: `mono` font, ≥ 12 px at the 640-px drawing width; leave room for labels.

**Quiz.** `quizzes/<slug>.yaml`, 3 multiple-choice questions (3–4 options, one correct, `answer` = 0-based index; vary its position). Inserted automatically at the top of "Check yourself" / "Kendini dene". Each `q`, option and `why` has `en` and `tr` (markdown: `$math$`, `code`, **bold**; one line each). Ask about understanding (why, what happens if, which reading of the table), not trivia. Every correct answer and every number in `why` must be on the lesson page or in its `output.txt`; `why` says where (part/table) when it uses a number. Wrong options are plausible misconceptions, not jokes.

## Experiments

- Run with `uv run python …` from the repo root (Python 3.14 via `.python-version`; MuJoCo 3.17.0, NumPy 2.5.4, SciPy, PyTorch 2.14 CPU/MPS, Gymnasium, OpenCV, Matplotlib, all locked in `uv.lock`). Before finishing: `uv run ruff check .` and `uv run python tools/check_book.py`.
- Menagerie models: `sys.path.insert(0, <repo>/experiments)` then `from common.menagerie import path; path("robotstudio_so101")` (also `unitree_go2`, `franka_emika_panda`). Already cached; do not clone elsewhere.
- Deterministic (fixed seeds), self-contained, prints its results as a table, finishes in **under ~3 minutes** on CPU. Print versions on the first line.
- Every experiment is a **sweep**: one parameter changes, one quantity is measured; the twist of the lesson comes from that data.
- English identifiers, comments and output.

## Honesty rules (non-negotiable)

- Every number from our runs must appear in `output.txt`. Never write a number you did not measure or did not read in a linked source.
- External claims link a primary source. If you cannot verify something, leave it out.
- Simulation results are called simulation results. Model outputs are not measurements.
- Real-hardware steps (SO-101) are written as procedures; results are explicitly "not yet measured on the real arm".
- Do not mention any other company's docs/design as inspiration.

## Turkish version

- Same structure, same numbers, same code blocks (code stays English).
- **Technical terms stay in English** with Turkish suffixes: timestep, contact, joint, actuator, frame, rotation matrix, quaternion, Jacobian, inverse kinematics, singularity, controller, gain, overshoot, damping, stiffness, trajectory, policy, reward, rollout, dataset, sweep, notebook, seed… Everyday words stay Turkish (kütle, hız, açı, eklem, masa, küp, kol, hata).
- Decimal comma in Turkish (4,37), decimal point in English (4.37).
- Correct Turkish grammar and characters (ç, ğ, ı, İ, ö, ş, ü).

## Links

- Lessons link each other relatively (`../part-3-kinematics/10-inverse-kinematics.md`, `09-jacobians.md`); Turkish lessons link Turkish lessons.
- Scripts: from an English lesson `../experiments/NN-slug/experiment.py`; from a Turkish lesson `../../experiments/NN-slug/experiment.py`.
- No in-page anchors with non-ASCII characters.
