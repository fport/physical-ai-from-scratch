"""Build the book's structure from curriculum.yaml (English at the root, Turkish under tr/).

Writes SUMMARY.md, tr/SUMMARY.md, part index pages, start-here/prerequisites.md, the appendix
index, and the metadata block of every lesson (between <!-- meta --> and <!-- /meta -->).
Lesson bodies are never touched; missing lessons and pages are created as skeletons.

Run:  .venv/bin/python tools/build_book.py
"""
import os
import re

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = yaml.safe_load(open(os.path.join(ROOT, "curriculum.yaml")))
AIEFS = {k: (C["aiefs_base"] + v[0], v[1]) for k, v in C["aiefs"].items()}

T = {
    "en": dict(
        part="Part", overview="Overview", lesson="Lesson", run="Run type", time="Time", status="Status",
        prereq="Prerequisites", learn="Learn first", code="Code", none="none", minutes="min",
        status_txt={"done": "✅ done", "draft": "📝 draft", "skeleton": "🚧 coming soon"},
        run_txt={"CPU": "CPU · Mac or Colab CPU", "GPU": "GPU · Colab GPU", "ART": "ART · shipped results, runs on CPU", "HW": "HW · real SO-101"},
        skeleton_aside="> **Status: coming soon.** This lesson has not been written yet. The structure below is the plan; no results are claimed.",
        learn_first_intro="This part assumes the following. If any of it is new, learn it on AI Engineering from Scratch first:",
        lessons_h="Lessons", start="Start here", appendix="Appendix", reference="Reference", intro="Introduction",
        ref_pages=[("Glossary", "glossary.md"), ("References", "references.md"), ("Notebooks", "notebooks/README.md"), ("Contributing", "contributing.md")],
        app_title="Appendix", app_item="Appendix", app_intro="Topics that are optional, need a GPU, or can only be covered from sources.",
        prereq_title="Prerequisites",
        prereq_intro=("This book teaches the robotics-specific parts of physical AI. General math, machine learning, deep learning, "
                      "reinforcement-learning basics, transformers and diffusion are not re-taught here: each part links to the matching "
                      "lessons of the free [AI Engineering from Scratch](https://aiengineeringfromscratch.com/index.html) course. "
                      "You need Python and the items listed for the part you are starting."),
        exp="Experiment", notebook="notebook", sections=["The problem", "Core idea", "Experiment", "Break it", "On the real robot", "Check yourself", "Takeaways and sources"],
    ),
    "tr": dict(
        part="Bölüm", overview="Genel bakış", lesson="Ders", run="Çalışma türü", time="Süre", status="Durum",
        prereq="Önkoşul", learn="Önce öğren", code="Kod", none="yok", minutes="dk",
        status_txt={"done": "✅ tam", "draft": "📝 taslak", "skeleton": "🚧 yakında"},
        run_txt={"CPU": "CPU · Mac ya da Colab CPU", "GPU": "GPU · Colab GPU", "ART": "ART · hazır sonuçlar, CPU'da çalışır", "HW": "HW · gerçek SO-101"},
        skeleton_aside="> **Durum: yakında.** Bu ders henüz yazılmadı. Aşağıdaki yapı plandır; hiçbir sonuç iddia edilmiyor.",
        learn_first_intro="Bu bölüm aşağıdakileri bildiğini varsayar. Yeni olan varsa önce AI Engineering from Scratch'te öğren:",
        lessons_h="Dersler", start="Başlarken", appendix="Ekler", reference="Başvuru", intro="Giriş",
        ref_pages=[("Sözlük", "glossary.md"), ("Kaynakça", "references.md"), ("Notebook'lar", "notebooks/README.md"), ("Katkı", "contributing.md")],
        app_title="Ekler", app_item="Ek", app_intro="İsteğe bağlı, GPU isteyen ya da yalnız kaynaklardan anlatılabilen konular.",
        prereq_title="Önkoşullar",
        prereq_intro=("Bu kitap physical AI'nin robotiğe özgü kısmını öğretir. Genel matematik, machine learning, deep learning, "
                      "reinforcement learning temelleri, transformer ve diffusion burada yeniden anlatılmaz: her bölüm ücretsiz "
                      "[AI Engineering from Scratch](https://aiengineeringfromscratch.com/index.html) kursunun ilgili derslerine bağlanır. "
                      "Python ve başladığın bölüm için listelenenler yeterli."),
        exp="Deney", notebook="notebook", sections=["Problem", "Ana fikir", "Deney", "Boz", "Gerçek robotta", "Kendini dene", "Akılda kalanlar ve kaynaklar"],
    ),
}

LESSONS = {}
for p in C["parts"]:
    for l in p["lessons"]:
        LESSONS[l["n"]] = dict(l, part=p)


def lesson_path(n):
    l = LESSONS[n]
    return f"{l['part']['dir']}/{l['slug']}.md"


def write(path, text):
    full = os.path.join(ROOT, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    if not os.path.exists(full) or open(full).read() != text:
        open(full, "w").write(text)


def rel(from_path, to_path):
    return os.path.relpath(to_path, os.path.dirname(from_path))


def aiefs_link(key):
    url, title = AIEFS[key]
    return f"[{title}]({url})"


def meta_block(n, lang, path):
    t, l = T[lang], LESSONS[n]
    p = l["part"]
    base = "" if lang == "en" else "tr/"
    pre = ", ".join(f"[{m:02d}]({rel(path, base + lesson_path(m))})" for m in l["prereq"]) or t["none"]
    learn = " · ".join(aiefs_link(k) for k in l.get("aiefs", p["aiefs"])) or t["none"]
    code = []
    exp = f"experiments/{l['slug']}/experiment.py"
    if os.path.exists(os.path.join(ROOT, exp)):
        code.append(f"[`{exp}`]({rel(path, exp)})")
    nb = f"notebooks/{l['slug']}.ipynb"
    if os.path.exists(os.path.join(ROOT, nb)):
        code.append(f"[{t['notebook']}]({rel(path, nb)})")
    rows = [
        "<!-- meta -->", "| | |", "|---|---|",
        f"| **{t['part']}** | [{p['id'] if p['id'] <= 10 else ''}{' · ' if p['id'] <= 10 else ''}{p[lang]}]({rel(path, base + p['dir'] + '/README.md')}) |",
        f"| **{t['run']}** | {t['run_txt'][l['run']]} |",
        f"| **{t['prereq']}** | {pre} |",
        f"| **{t['learn']}** | {learn if p['id'] <= 10 else t['none']} |",
        f"| **{t['time']}** | ~{l['time']} {t['minutes']} |",
    ]
    if code:
        rows.append(f"| **{t['code']}** | {' · '.join(code)} |")
    rows += [f"| **{t['status']}** | {t['status_txt'][l['status']]} |", "<!-- /meta -->"]
    return "\n".join(rows)


def update_lesson(n, lang):
    t, l = T[lang], LESSONS[n]
    path = ("" if lang == "en" else "tr/") + lesson_path(n)
    full = os.path.join(ROOT, path)
    title = f"# {n:02d} · {l[lang]}"
    meta = meta_block(n, lang, path)
    if not os.path.exists(full):
        body = [t["skeleton_aside"], ""] + sum(([f"## {s}", "", "_—_", ""] for s in t["sections"]), [])
        write(path, f"{title}\n\n{meta}\n\n" + "\n".join(body))
        return
    s = open(full).read()
    s = re.sub(r"^# .*", title, s, count=1)
    if "<!-- meta -->" in s:
        s = re.sub(r"<!-- meta -->.*?<!-- /meta -->", lambda _: meta, s, count=1, flags=re.S)
    else:
        s = re.sub(r"^(# .*\n)", lambda m: m.group(1) + "\n" + meta + "\n", s, count=1)
    write(path, s)


def part_readme(p, lang):
    t = T[lang]
    base = "" if lang == "en" else "tr/"
    path = f"{base}{p['dir']}/README.md"
    head = f"# {t['part']} {p['id']} · {p[lang]}" if p["id"] <= 10 else f"# {p[lang]}"
    out = [head, "", f"> {p['goal_' + lang]}", ""]
    if p["aiefs"]:
        out += [f"## {t['learn']}", "", t["learn_first_intro"], ""] + [f"- {aiefs_link(k)}" for k in p["aiefs"]] + [""]
    out += [f"## {t['lessons_h']}", "", f"| # | {t['lesson']} | {t['run']} | {t['time']} | {t['status']} |", "|---|---|---|---|---|"]
    for l in p["lessons"]:
        out.append(f"| {l['n']:02d} | [{l[lang]}]({l['slug']}.md) | {l['run']} | ~{l['time']} {t['minutes']} | {t['status_txt'][l['status']]} |")
    write(path, "\n".join(out) + "\n")


def prerequisites_page(lang):
    t = T[lang]
    base = "" if lang == "en" else "tr/"
    path = f"{base}start-here/prerequisites.md"
    out = [f"# {t['prereq_title']}", "", t["prereq_intro"], ""]
    for p in C["parts"]:
        if not p["aiefs"]:
            continue
        out += [f"## [{t['part']} {p['id']} · {p[lang]}]({rel(path, base + p['dir'] + '/README.md')})", ""]
        out += [f"- {aiefs_link(k)}" for k in p["aiefs"]] + [""]
    write(path, "\n".join(out))


def simple_page(path, title, lang):
    full = os.path.join(ROOT, path)
    if os.path.exists(full):
        s = open(full).read()
        write(path, re.sub(r"^# .*", f"# {title}", s, count=1))
    else:
        write(path, f"# {title}\n\n{T[lang]['skeleton_aside']}\n")


def appendix(lang):
    t = T[lang]
    base = "" if lang == "en" else "tr/"
    out = [f"# {t['app_title']}", "", t["app_intro"], "", "| | |", "|---|---|"]
    for a in C["appendix"]:
        out.append(f"| {a['key']} | [{a[lang]}]({a['slug']}.md) |")
        simple_page(f"{base}appendix/{a['slug']}.md", f"{t['app_item']} {a['key']} · {a[lang]}", lang)
    write(f"{base}appendix/README.md", "\n".join(out) + "\n")


def summary(lang):
    t = T[lang]
    base = "" if lang == "en" else "tr/"
    out = ["# Table of contents", "", f"* [{t['intro']}](README.md)", "", f"## {t['start']}", ""]
    for s in C["start_here"]:
        out.append(f"* [{s[lang]}](start-here/{s['slug']}.md)")
        if s["slug"] != "prerequisites":
            simple_page(f"{base}start-here/{s['slug']}.md", s[lang], lang)
    out.append("")
    for p in C["parts"]:
        out += [f"## {t['part']} {p['id']} · {p[lang]}" if p["id"] <= 10 else f"## {p[lang]}", "", f"* [{t['overview']}]({p['dir']}/README.md)"]
        out += [f"  * [{l['n']:02d} · {l[lang]}]({p['dir']}/{l['slug']}.md)" for l in p["lessons"]] + [""]
    out += [f"## {t['appendix']}", "", f"* [{t['app_title']}](appendix/README.md)"]
    out += [f"  * [{a['key']} · {a[lang]}](appendix/{a['slug']}.md)" for a in C["appendix"]] + [""]
    out += [f"## {t['reference']}", ""] + [f"* [{a}]({b})" for a, b in t["ref_pages"]] + [""]
    write(f"{base}SUMMARY.md", "\n".join(out))


for lang in ("en", "tr"):
    for p in C["parts"]:
        part_readme(p, lang)
    for n in LESSONS:
        update_lesson(n, lang)
    prerequisites_page(lang)
    appendix(lang)
    summary(lang)
print(f"built: {len(LESSONS)} lessons × 2 languages, {len(C['parts'])} parts")
