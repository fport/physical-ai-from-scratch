// Turns the book's markdown sources (repo root = English, tr/ = Turkish) into Starlight content.
// Sources are never modified: src/content/docs/ and src/sidebar.generated.json are rewritten on every run.
//
// - The first "# Title" line becomes the frontmatter title; the blockquote right below it the description.
// - README.md → index.md; the root README.md → index.mdx (landing page with phase cards).
// - Relative .md links become site routes; links to other files (scripts, notebooks) go to GitHub.
// - "> **Status: …**" / "> **Durum: …**" blocks become Starlight asides.
// - The sidebar comes from SUMMARY.md (labels translated from tr/SUMMARY.md); lesson badges from
//   the Status row of each page's metadata table.
// - "<!-- figure: NN-name -->" lines become interactive figure slots (module src/figures/NN-name.ts).
// - quizzes/NN-slug.yaml becomes clickable quizzes by stage: "pre" before Core idea, "check" before
//   Experiment, "post" (default) at the top of "Check yourself" / "Kendini dene".
// - Lessons with a "Build it" section get an "Act on this lesson" checklist under the metadata table.
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import YAML from 'yaml'

const SITE = path.dirname(path.dirname(fileURLToPath(import.meta.url)))
const ROOT = path.dirname(SITE)
const OUT = path.join(SITE, 'src/content/docs')
const FIG = path.join(SITE, 'public/figures')
const IMAGE = /\.(png|jpe?g|svg|gif|webp)$/i
const REPO = 'https://github.com/fport/physical-ai-from-scratch'
const BASE = (process.env.SITE_BASE || '/').replace(/\/?$/, '/')

const LANGS = {
  en: {
    dir: '', statusRow: 'Status', check: 'Check yourself', core: 'Core idea', experiment: 'Experiment', build: 'Build it', why: 'Why:',
    quiz: { pre: (n) => `Before you read · ${n} questions. Guess first; the lesson tells you why.`, check: (n) => `Check after building · ${n} questions.`, post: (n) => `Quick check · ${n} questions. Click an answer.` },
    act: { title: 'Act on this lesson', note: 'Tick each step when you have done it. Saved in this browser only.', read: ['Read', 'The problem and the core idea; answer the questions before you read.'], build: ['Build it', 'Type the steps yourself, then run'], run: ['Run the experiment', 'Predict first, then run'], break: ['Break it', 'Change the one parameter and watch it fail.'], quiz: ['Check yourself', 'Quizzes and exercises at the end.'], done: 'Mark lesson complete', undo: 'Completed · undo' },
    aside: { 'coming soon': ['caution', 'Coming soon'], draft: ['note', 'Draft'], done: ['tip', 'Done'] },
    status: { done: 'tam', draft: 'taslak', 'coming soon': 'yakinda' },
    tagline: 'Start with a laptop, end with a robot arm doing a task on its own. 44 lessons, every number from a real run.',
    start: 'Start the first lesson', setup: 'Setup', lessons: 'Lessons', map: 'Map', links: '**[Setup]',
  },
  tr: {
    dir: 'tr/', statusRow: 'Durum', check: 'Kendini dene', core: 'Ana fikir', experiment: 'Deney', build: 'Sıfırdan yaz', why: 'Neden:',
    quiz: { pre: (n) => `Okumadan önce · ${n} soru. Önce tahmin et; nedenini ders anlatıyor.`, check: (n) => `Yazdıktan sonra · ${n} soru.`, post: (n) => `Hızlı kontrol · ${n} soru. Bir cevaba tıkla.` },
    act: { title: 'Bu derste yap', note: 'Her adımı yapınca işaretle. Yalnız bu tarayıcıda saklanır.', read: ['Oku', 'Problem ve ana fikir; okumadan önceki soruları cevapla.'], build: ['Sıfırdan yaz', 'Adımları kendin yaz, sonra çalıştır'], run: ['Deneyi çalıştır', 'Önce tahmin et, sonra çalıştır'], break: ['Boz', 'Tek parametreyi değiştir ve bozulmasını izle.'], quiz: ['Kendini dene', 'Sondaki quiz ve alıştırmalar.'], done: 'Dersi tamamlandı say', undo: 'Tamamlandı · geri al' },
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

// interactive figure slots: a marker line in the markdown → an empty div the page script fills
function figures(body, lang) {
  return body.replace(/^<!-- figure: ([0-9][\w-]*) -->$/gm, (m, name) => {
    if (!fs.existsSync(path.join(SITE, 'src/figures', `${name}.ts`))) throw new Error(`figure module missing: ${name}`)
    return `<div class="fa-fig not-content" data-fig="${name}" data-lang="${lang}"></div>`
  })
}

// quizzes/NN-slug.yaml → markdown inside HTML wrappers, so math and inline code still render
const STAGES = ['pre', 'check', 'post']
function quizBlock(qs, stage, name, L, lang) {
  return [
    `<div class="fa-quiz" data-stage="${stage}" data-lesson="${name}-${lang}-${stage}">`,
    `<p class="fa-quiz-tag">${L.quiz[stage](qs.length)}</p>`,
    '',
    ...qs.flatMap((q, i) => [
      `<div class="fa-q" data-answer="${q.answer}">`,
      '',
      `**${i + 1}.** ${q.q[lang].trim()}`,
      '',
      ...q.options.map((o) => `- ${o[lang].trim()}`),
      '',
      '<div class="fa-q-why">',
      '',
      `**${L.why}** ${q.why[lang].trim()}`,
      '',
      '</div>',
      '</div>',
      '',
    ]),
    '<p class="fa-quiz-score"></p>',
    '</div>',
    '',
  ].join('\n')
}

function quiz(body, srcPath, L, lang) {
  const name = path.posix.basename(srcPath, '.md')
  const file = path.join(ROOT, 'quizzes', `${name}.yaml`)
  if (!fs.existsSync(file)) return body
  const all = YAML.parse(fs.readFileSync(file, 'utf8'))
  const anchor = { pre: [L.core, 'before'], check: [L.experiment, 'before'], post: [L.check, 'after'] }
  for (const stage of STAGES) {
    const qs = all.filter((q) => (q.stage || 'post') === stage)
    if (!qs.length) continue
    const [title, where] = anchor[stage]
    const heading = new RegExp(`^## ${escapeRe(title)}\\n+`, 'm')
    if (!heading.test(body)) throw new Error(`${srcPath}: no "## ${title}" heading for the ${stage} quiz`)
    const block = quizBlock(qs, stage, name, L, lang)
    body = body.replace(heading, (h) => (where === 'after' ? h + block + '\n' : block + '\n' + h))
  }
  return body
}

// "Act on this lesson": a checklist under the metadata table, for lessons with a Build it section
function actPanel(body, srcPath, L, lang) {
  if (!new RegExp(`^## ${escapeRe(L.build)}$`, 'm').test(body)) return body
  const name = path.posix.basename(srcPath, '.md')
  const run = (f) => `<code>uv run python experiments/${name}/${f}</code>`
  const steps = [
    ['read', L.act.read[1]],
    ['build', `${L.act.build[1]} ${run('build.py')}`],
    ['run', `${L.act.run[1]} ${run('experiment.py')}`],
    ['break', L.act.break[1]],
    ['quiz', L.act.quiz[1]],
  ]
  const panel = [
    `<div class="fa-act not-content" data-lesson="${name}" data-done="${L.act.done}" data-undo="${L.act.undo}">`,
    `<p class="fa-act-title">${L.act.title}</p>`,
    `<p class="fa-act-note">${L.act.note}</p>`,
    '<ol>',
    ...steps.map(([k, text], i) => `<li><label><input type="checkbox" data-step="${k}" /> <b>${String(i + 1).padStart(2, '0')} · ${L.act[k][0]}</b> <span>${text}</span></label></li>`),
    '</ol>',
    `<button type="button" class="fa-act-done">${L.act.done}</button>`,
    '</div>',
    '',
  ].join('\n')
  return body.replace(/<!-- \/meta -->\n/, (m) => m + '\n' + panel + '\n')
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
  const lang = L.dir ? 'tr' : 'en'
  return { front: front.join('\n'), body: actPanel(quiz(figures(statusAsides(rewriteLinks(body, srcPath), L), lang), srcPath, L, lang), srcPath, L, lang) }
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
  const home = convert(`${L.dir}README.md`, L, { template: 'splash', tableOfContents: false })
  const body = home.body
    .replace(new RegExp(`^${escapeRe(L.links)}.*\\n+`, 'm'), '')
    .replace(new RegExp(`^## ${L.map}\\n[\\s\\S]*?(?=\\n## )`, 'm'), [`## ${L.map}`, '', '<CardGrid>', ...cards, '</CardGrid>', ''].join('\n'))
  const imports = [
    "import { CardGrid, LinkCard } from '@astrojs/starlight/components'",
    "import Landing from '../../components/Landing.astro'".replace('../../', lang === 'tr' ? '../../../' : '../../'),
    "import Sims from '../../components/Sims.astro'".replace('../../', lang === 'tr' ? '../../../' : '../../'),
  ].join('\n')
  fs.mkdirSync(path.join(OUT, L.dir), { recursive: true })
  fs.writeFileSync(
    path.join(OUT, L.dir, 'index.mdx'),
    home.front + imports + `\n\n<Landing lang="${lang}" />\n\n<Sims lang="${lang}" />\n\n` + body,
  )
  count++
}
fs.writeFileSync(path.join(SITE, 'src/sidebar.generated.json'), JSON.stringify(sidebar, null, 2))
console.log(`sync: ${count} pages, ${sidebar.length} sidebar groups`)
