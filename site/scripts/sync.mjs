// Turns the book's markdown sources (repo root = English, tr/ = Turkish) into Starlight content.
// Sources are never modified: src/content/docs/ and src/sidebar.generated.json are rewritten on every run.
//
// - The first "# Title" line becomes the frontmatter title; the blockquote right below it the description.
// - README.md → index.md; the root README.md → index.mdx (landing page with phase cards).
// - Relative .md links become site routes; links to other files (scripts, notebooks) go to GitHub.
// - "> **Status: …**" / "> **Durum: …**" blocks become Starlight asides.
// - The sidebar comes from SUMMARY.md (labels translated from tr/SUMMARY.md); lesson badges from
//   the Status row of each page's metadata table.
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const SITE = path.dirname(path.dirname(fileURLToPath(import.meta.url)))
const ROOT = path.dirname(SITE)
const OUT = path.join(SITE, 'src/content/docs')
const FIG = path.join(SITE, 'public/figures')
const IMAGE = /\.(png|jpe?g|svg|gif|webp)$/i
const REPO = 'https://github.com/fport/physical-ai-from-scratch'
const BASE = (process.env.SITE_BASE || '/').replace(/\/?$/, '/')

const LANGS = {
  en: {
    dir: '', statusRow: 'Status',
    aside: { 'coming soon': ['caution', 'Coming soon'], draft: ['note', 'Draft'], done: ['tip', 'Done'] },
    status: { done: 'tam', draft: 'taslak', 'coming soon': 'yakinda' },
    tagline: 'Start with a laptop, end with a robot arm doing a task on its own. 44 lessons, every number from a real run.',
    start: 'Start the first lesson', setup: 'Setup', lessons: 'Lessons', map: 'Map', links: '**[Setup]',
  },
  tr: {
    dir: 'tr/', statusRow: 'Durum',
    aside: { yakında: ['caution', 'Yakında'], taslak: ['note', 'Taslak'], tam: ['tip', 'Tam'] },
    status: { tam: 'tam', taslak: 'taslak', yakında: 'yakinda' },
    tagline: 'Bir dizüstüyle başla, bir robot kola kendi başına iş yaptır. 44 ders, her sayı gerçek bir koşudan.',
    start: 'İlk derse başla', setup: 'Kurulum', lessons: 'Ders', map: 'Harita', links: '**[Kurulum]',
  },
}

const PAGES = new Set()
// READMEs open with a GitHub-only hero + badges block; the site has its own splash hero
const read = (p) => fs.readFileSync(path.join(ROOT, p), 'utf8').replace(/^<!-- github-only[\s\S]*?<!-- \/github-only -->\n+/, '')
const yaml = (s) => JSON.stringify(s)
const escapeRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

// repo path → site slug (no leading/trailing slash) and route
const slug = (repoPath) => repoPath.replace(/\.md$/, '').replace(/(^|\/)README$/, '').replace(/\/$/, '').toLowerCase()
const route = (repoPath) => BASE + (slug(repoPath) ? slug(repoPath) + '/' : '')

function rewriteLinks(body, srcPath) {
  return body.replace(/\]\(([^)\s]+)\)/g, (m, target) => {
    if (/^(https?:|mailto:|#)/.test(target)) return m
    const [file, anchor] = target.split('#')
    const resolved = path.posix.normalize(path.posix.join(path.posix.dirname(srcPath), file))
    if (resolved.endsWith('.md') && PAGES.has(resolved)) return `](${route(resolved)}${anchor ? '#' + anchor : ''})`
    const abs = path.join(ROOT, resolved)
    if (IMAGE.test(resolved) && fs.existsSync(abs)) {
      fs.mkdirSync(path.dirname(path.join(FIG, resolved)), { recursive: true })
      fs.copyFileSync(abs, path.join(FIG, resolved))
      return `](${BASE}figures/${resolved})`
    }
    const kind = fs.existsSync(abs) && fs.statSync(abs).isDirectory() ? 'tree' : 'blob'
    return `](${REPO}/${kind}/main/${resolved})`
  })
}

function statusAsides(body, L) {
  const lines = body.split('\n')
  const out = []
  for (let i = 0; i < lines.length; i++) {
    const m = lines[i].match(/^> \*\*(?:Status|Durum): ([^.*]+)\.\*\*\s*(.*)$/)
    const kind = m && L.aside[m[1].trim().toLocaleLowerCase('tr')]
    if (!kind) { out.push(lines[i]); continue }
    const text = [m[2]]
    while (i + 1 < lines.length && lines[i + 1].startsWith('> ')) text.push(lines[++i].slice(2))
    out.push(`:::${kind[0]}[${kind[1]}]`, text.join('\n').trim(), ':::')
  }
  return out.join('\n')
}

function splitTitle(src) {
  const m = src.match(/^# (.+)\n+/)
  if (!m) throw new Error('missing title')
  return { title: m[1].trim(), body: src.slice(m[0].length) }
}

function motto(body) {
  const m = body.match(/^> (?!\*\*(?:Status|Durum))(.+)/)
  return m ? m[1].replace(/[*_`]/g, '').trim() : undefined
}

function status(src, L) {
  const row = src.match(new RegExp(`^\\| \\*\\*${L.statusRow}\\*\\* \\| (.+) \\|$`, 'm'))
  const text = row ? row[1] : (src.match(/^> \*\*(?:Status|Durum): ([^.*]+)\./m) || [])[1]
  if (!text) return null
  const t = text.toLocaleLowerCase('tr')
  for (const [k, v] of Object.entries(L.status)) if (t.includes(k)) return v
  return null
}

function convert(srcPath, L, extra = {}) {
  const { title, body } = splitTitle(read(srcPath))
  const fm = { title, ...extra }
  const d = motto(body)
  if (d && !fm.description) fm.description = d
  fm.editUrl = `${REPO}/edit/main/${srcPath}`
  const front = ['---', ...Object.entries(fm).map(([k, v]) => `${k}: ${typeof v === 'string' ? yaml(v) : JSON.stringify(v)}`), '---', '']
  return { front: front.join('\n'), body: statusAsides(rewriteLinks(body, srcPath), L) }
}

function parseSummary(file) {
  const groups = []
  const top = []
  let group = null
  for (const line of read(file).split('\n')) {
    const g = line.match(/^## (.+)$/)
    if (g) { group = { label: g[1].trim(), items: [] }; groups.push(group); continue }
    const it = line.match(/^\s*\* \[(.+)\]\((.+)\)$/)
    if (!it) continue
    const repoPath = path.posix.join(path.posix.dirname(file), it[2])
    PAGES.add(repoPath)
    ;(group ? group.items : top).push({ label: it[1], file: repoPath })
  }
  return { groups, top }
}

const summaries = { en: parseSummary('SUMMARY.md'), tr: parseSummary('tr/SUMMARY.md') }

// --- sidebar: English structure, Turkish labels by position ---
const sidebar = summaries.en.groups.map((g, gi) => {
  const tg = summaries.tr.groups[gi]
  return {
    label: g.label,
    translations: tg ? { tr: tg.label } : {},
    collapsed: gi > 1,
    items: g.items.map((item, ii) => {
      const entry = { slug: slug(item.file), label: item.label, translations: tg?.items[ii] ? { tr: tg.items[ii].label } : {} }
      const st = /\/\d\d-[^/]+\.md$/.test(item.file) ? status(read(item.file), LANGS.en) : null
      if (st === 'tam') entry.badge = { text: { en: 'done', tr: 'tam' }, variant: 'success' }
      if (st === 'taslak') entry.badge = { text: { en: 'draft', tr: 'taslak' }, variant: 'caution' }
      if (st === 'yakinda') entry.attrs = { class: 'coming-soon' }
      return entry
    }),
  }
})

// --- content ---
fs.rmSync(OUT, { recursive: true, force: true })
fs.rmSync(FIG, { recursive: true, force: true })
let count = 0
for (const [lang, L] of Object.entries(LANGS)) {
  const { groups, top } = summaries[lang]
  for (const { file } of [...top, ...groups.flatMap((g) => g.items)]) {
    if (file === `${L.dir}README.md`) continue
    const rel = file.replace(/(^|\/)README\.md$/, '$1index.md')
    const { front, body } = convert(file, L)
    fs.mkdirSync(path.dirname(path.join(OUT, rel)), { recursive: true })
    fs.writeFileSync(path.join(OUT, rel), front + body)
    count++
  }

  // landing page: README + phase cards
  const cards = groups.filter((g) => /(^|\/)(part-\d+|capstone)[^/]*\/README\.md$/.test(g.items[0]?.file || '')).map((g) => {
    const m = motto(splitTitle(read(g.items[0].file)).body) || ''
    const nums = g.items.slice(1).map((x) => x.label.split(' · ')[0])
    const range = nums.length > 1 ? `${nums[0]}–${nums[nums.length - 1]}` : nums[0]
    return `  <LinkCard title=${yaml(g.label)} description=${yaml(`${m} · ${L.lessons} ${range}`)} href=${yaml(route(g.items[0].file))} />`
  })
  const home = convert(`${L.dir}README.md`, L, {
    template: 'splash',
    hero: {
      tagline: L.tagline,
      actions: [
        { text: L.start, link: route(`${L.dir}part-1-simulation/01-world-as-text.md`), icon: 'right-arrow' },
        { text: L.setup, link: route(`${L.dir}start-here/setup.md`), variant: 'minimal' },
      ],
    },
  })
  const body = home.body
    .replace(new RegExp(`^${escapeRe(L.links)}.*\\n+`, 'm'), '')
    .replace(new RegExp(`^## ${L.map}\\n[\\s\\S]*?(?=\\n## )`, 'm'), [`## ${L.map}`, '', '<CardGrid>', ...cards, '</CardGrid>', ''].join('\n'))
  fs.mkdirSync(path.join(OUT, L.dir), { recursive: true })
  fs.writeFileSync(path.join(OUT, L.dir, 'index.mdx'), home.front + "import { CardGrid, LinkCard } from '@astrojs/starlight/components'\n\n" + body)
  count++
}
fs.writeFileSync(path.join(SITE, 'src/sidebar.generated.json'), JSON.stringify(sidebar, null, 2))
console.log(`sync: ${count} pages, ${sidebar.length} sidebar groups`)
