# Site

The book's website: [Astro](https://astro.build/) + [Starlight](https://starlight.astro.build/), dark theme, English by default with a Turkish version under `/tr/`.

The source is the markdown at the repo root (English) and under `tr/` (Turkish), in GitBook layout with a `SUMMARY.md` each. Nothing in this folder is written by hand: before every `dev` and `build`, `scripts/sync.mjs` converts that markdown into `src/content/docs/` and builds the sidebar from `SUMMARY.md`. To edit a lesson, edit the `.md` file at the root or under `tr/`.

```bash
cd site
npm install
npm run dev        # http://localhost:4321
npm run build      # dist/ (with search index)
```

When `SUMMARY.md` or a lesson's status (the **Status** row of its metadata table) changes, restart the dev server so the sidebar updates (the Astro 7 dev server runs in the background: `npx astro dev stop`).

## Conversion rules (`scripts/sync.mjs`)

| In the source | On the site |
|---|---|
| First `# Title` | Page title (frontmatter `title`) |
| `> motto` right below the title | Page description |
| `README.md` | The folder's index page; the root `README.md` becomes the landing page (with phase cards) |
| Relative `.md` link | Site route (`tr/…` pages go under `/tr/`) |
| Link to another file (`experiment.py`, `.ipynb`) | The file on GitHub |
| `> **Status: coming soon.** …` / `> **Durum: yakında.** …` | "Coming soon" / "Yakında" aside |
| **Status** row in the metadata table | `DONE` / `DRAFT` badge in the sidebar; "coming soon" lessons are dimmed |
| Labels in `tr/SUMMARY.md` | Turkish sidebar labels (matched by position to `SUMMARY.md`) |

Styling lives in `src/styles/custom.css`. With no stored preference the theme opens dark (`src/components/ThemeProvider.astro`).
