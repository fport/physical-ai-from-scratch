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
3. `## Core idea` / `## Ana fikir` — the minimum theory: definitions, 1–4 equations, a table if useful. Math in `$…$` (inline) and `$$…$$` (display). No derivations that AIEFS already covers.
4. `## Experiment` / `## Deney` — predict → run → observe. Show the essential code (short blocks, the full script lives in `experiments/`), then **a results table with your measured numbers**. State MuJoCo/NumPy/PyTorch version and that it ran on an Apple Silicon Mac (Apple M5).
5. `## Break it` / `## Boz` — push one parameter until the method fails; show the measured failure and explain the mechanism.
6. `## On the real robot` / `## Gerçek robotta` — what changes on the SO-101 / Go2 / a real arm. Only sourced facts (link the primary source). Anything not measured on hardware is said to be unmeasured.
7. `## Check yourself` / `## Kendini dene` — 3 exercises (easy → hard). Answers at the end under `### Answers` / `### Cevaplar` (numbers only if you actually computed them).
8. `## Takeaways and sources` / `## Akılda kalanlar ve kaynaklar` — 3 bullets, "Next: lesson NN+1", then a source list (primary sources: papers, official docs, code).

Target: **1,200–1,800 words** per language, tight technical prose. No hype, no video-style hooks, no rhetorical filler, no emoji. Prefer tables and equations over paragraphs. Every lesson must teach something a robotics engineer actually uses.

## Experiments

- Run with `/Users/furkanportakal/www/fiziksel-ai-sifirdan/.venv/bin/python` (MuJoCo 3.17.0, NumPy 2.5.4, SciPy, PyTorch 2.14 CPU/MPS, Gymnasium, OpenCV, Matplotlib).
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
