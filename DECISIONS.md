# Open decisions

Decisions still to be made for the book. This file is not part of the table of contents (SUMMARY.md).

## 1 · Repo name

Decided (2026-10-27): **`fport/physical-ai-from-scratch`**, **private**. The local folder is still named `fiziksel-ai-sifirdan`.

## 2 · License

There is no LICENSE file yet. A common choice is separate licenses for text and code:

- **Code** (`experiments/`, `notebooks/`): MIT or Apache-2.0.
- **Text** (lesson pages): CC BY 4.0 (free use with attribution) or CC BY-SA 4.0 (derivatives under the same license).

Menagerie models keep their own licenses (`robotstudio_so101` Apache-2.0, `unitree_go2` BSD-3-Clause); the notebooks don't vendor the models, they clone them.

## 3 · Making it public

The repo is private: https://github.com/fport/physical-ai-from-scratch. When it goes public:

- Add "Open in Colab" badges to the notebooks (the badge URL needs the repo path).
- Point the video descriptions at the lesson pages.

## 4 · Website: Starlight or GitBook?

`site/` holds an Astro Starlight site (dark theme; English at `/`, Turkish at `/tr/`; the markdown sources stay at the repo root and under `tr/`). Locally: `cd site && npm install && npm run dev`. The static output (`npm run build` → `site/dist/`) can be hosted anywhere, e.g. a Cloudflare subdomain under `getporti.com`. Not deployed yet.

`.gitbook.yaml` is still there: the English root can be connected to gitbook.com with GitHub Sync (Turkish would be a second space with root `tr/`). Once one option is chosen the other can be removed.

## 5 · Content

- **The notebooks were written for the Turkish video series.** Their text is Turkish and they mention "Levha NN" and scripts in the video repo (`src/v6/...`). Clean them up and translate to English, or keep them as they are?
- **Lesson 31's label.** Strands is lesson 31 in the new order; the video and notebook say "Levha 32".
- **The earlier lesson 11 notebook** (`kk-11-pid`, single-hinge PID including the I term) is not in the book; lesson 11 uses the Go2 experiment. It could be added as an extra for the I term.
- **Real-arm lessons** (06, 10, 14, the real part of 19, 26, 36, 40): is there a leader arm? If not, LeRobot keyboard/phone teleop, or sim only.
