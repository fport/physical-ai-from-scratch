"""Repository checks that run without the heavy experiment dependencies.

1. Every relative link and image in the book's Markdown resolves to a file.
2. The files generated from curriculum.yaml are up to date (re-running build_book changes nothing).
3. Every lesson has an English and a Turkish page and an experiments/<slug>/output.txt.
4. Interactive figure markers (<!-- figure: NN-name -->) match between languages and have a module in
   site/src/figures/; every quizzes/<slug>.yaml is well formed and belongs to a lesson.

Run:  uv run python tools/check_book.py
"""
import glob
import os
import re
import subprocess
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP = ("site/", ".venv/", ".cache/", "node_modules/")


def markdown_files():
    return [f for f in glob.glob("**/*.md", root_dir=ROOT, recursive=True) if not f.startswith(SKIP)]


def broken_links():
    bad = []
    for f in markdown_files():
        text = open(os.path.join(ROOT, f), encoding="utf-8").read()
        text = re.sub(r"```.*?```", "", text, flags=re.S)
        text = re.sub(r"`[^`\n]*`", "", text)
        for target in re.findall(r"\]\(([^)\s]+)\)", text):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            path = os.path.normpath(os.path.join(ROOT, os.path.dirname(f), target.split("#")[0]))
            if not os.path.exists(path):
                bad.append(f"{f}: {target}")
    return bad


def generated_drift():
    before = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout
    subprocess.run([sys.executable, os.path.join(ROOT, "tools", "build_book.py")], cwd=ROOT, check=True,
                   capture_output=True)
    after = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout
    return sorted(set(after.splitlines()) - set(before.splitlines()))


def missing_pieces():
    c = yaml.safe_load(open(os.path.join(ROOT, "curriculum.yaml"), encoding="utf-8"))
    missing = []
    for part in c["parts"]:
        for lesson in part["lessons"]:
            for page in (f"{part['dir']}/{lesson['slug']}.md", f"tr/{part['dir']}/{lesson['slug']}.md",
                         f"experiments/{lesson['slug']}/output.txt"):
                if not os.path.exists(os.path.join(ROOT, page)):
                    missing.append(page)
    return missing


def lessons():
    c = yaml.safe_load(open(os.path.join(ROOT, "curriculum.yaml"), encoding="utf-8"))
    return [(p["dir"], le["slug"]) for p in c["parts"] for le in p["lessons"]]


def figures_and_quizzes():
    bad = []
    slugs = set()
    for d, slug in lessons():
        slugs.add(slug)
        marks = {}
        for page in (f"{d}/{slug}.md", f"tr/{d}/{slug}.md"):
            text = open(os.path.join(ROOT, page), encoding="utf-8").read()
            marks[page] = re.findall(r"^<!-- figure: ([0-9][\w-]*) -->$", text, re.M)
            for name in marks[page]:
                if not os.path.exists(os.path.join(ROOT, "site/src/figures", f"{name}.ts")):
                    bad.append(f"{page}: figure module site/src/figures/{name}.ts missing")
        a, b = marks.values()
        if a != b:
            bad.append(f"{slug}: figure markers differ between languages: {a} vs {b}")
    for f in sorted(glob.glob("quizzes/*.yaml", root_dir=ROOT)):
        slug = os.path.basename(f)[:-5]
        if slug not in slugs:
            bad.append(f"{f}: no lesson with this slug")
            continue
        try:
            qs = yaml.safe_load(open(os.path.join(ROOT, f), encoding="utf-8"))
        except yaml.YAMLError as e:
            bad.append(f"{f}: {e}")
            continue
        for i, q in enumerate(qs if isinstance(qs, list) else [None]):
            ok = (isinstance(q, dict) and all(isinstance(q.get(k), dict) and {"en", "tr"} <= set(q[k]) for k in ("q", "why"))
                  and isinstance(q.get("options"), list) and len(q["options"]) >= 2
                  and all(isinstance(o, dict) and {"en", "tr"} <= set(o) for o in q["options"])
                  and isinstance(q.get("answer"), int) and 0 <= q["answer"] < len(q["options"]))
            if not ok:
                bad.append(f"{f}: question {i + 1} needs q/options/why with en and tr, and a valid answer index")
    return bad


def main():
    problems = 0
    for title, items in [("broken links", broken_links()), ("missing lesson files", missing_pieces()),
                         ("figure and quiz problems", figures_and_quizzes()),
                         ("generated files out of date (run tools/build_book.py)", generated_drift())]:
        print(f"{title}: {len(items)}")
        for item in items[:50]:
            print(f"  {item}")
        problems += len(items)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
