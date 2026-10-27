# Open decisions

Decisions still to be made for the book. This file is not part of the table of contents (SUMMARY.md).

## 1 · Repo name

Decided (2026-10-27): **`fport/physical-ai-from-scratch`**, **public**. The local folder is still named `fiziksel-ai-sifirdan`.

## 2 · License

Decided (2026-10-27): code is MIT (`LICENSE`), written content is CC BY 4.0 (`LICENSE-CONTENT`), copyright Furkan Portakal 2026. Menagerie models are not vendored and keep their own licenses.

## 3 · GitHub Pages

The repo is public and GitHub Pages is enabled (source: GitHub Actions). `.github/workflows/pages.yml` builds `site/` and deploys it to https://fport.github.io/physical-ai-from-scratch/ on every push to `main`.

Still to do: add "Open in Colab" badges to the notebooks and point the video descriptions at the lesson pages.

## 4 · Website

`site/` holds an Astro Starlight site (dark theme; English at `/`, Turkish at `/tr/`; the markdown sources stay at the repo root and under `tr/`). Locally: `cd site && npm install && npm run dev`. The static output (`npm run build` → `site/dist/`) can be hosted anywhere, e.g. a Cloudflare subdomain under `getporti.com`. Deployed with GitHub Pages (see section 3).

`.gitbook.yaml` is still there: the English root can be connected to gitbook.com with GitHub Sync (Turkish would be a second space with root `tr/`). Once one option is chosen the other can be removed.

## 5 · Content

- **Notebooks** were written for the Turkish video series (Turkish text, "Levha NN" labels, old lesson numbers). Every lesson now has a maintained English script in `experiments/`; the notebooks could be retired or regenerated from those scripts.
- **Real-arm runs:** lesson 40 and the "On the real robot" sections are procedures; nothing has been measured on the real SO-101 yet. Is there a leader arm for teleoperation?
- **Lesson 39** results move a few plants per row with the random seed; averaging 3–5 seeds per row would make the table sturdier (several times the runtime).
- **Part 8 runtimes:** lessons 34–35 take 20–30 minutes on a loaded laptop; a GPU or fewer seeds would shorten them.
