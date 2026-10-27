"""Repository checks that run without the heavy experiment dependencies.

1. Every relative link and image in the book's Markdown resolves to a file.
2. The files generated from curriculum.yaml are up to date (re-running build_book changes nothing).
3. Every lesson has an English and a Turkish page and an experiments/<slug>/output.txt.

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


def main():
    problems = 0
    for title, items in [("broken links", broken_links()), ("missing lesson files", missing_pieces()),
                         ("generated files out of date (run tools/build_book.py)", generated_drift())]:
        print(f"{title}: {len(items)}")
        for item in items[:50]:
            print(f"  {item}")
        problems += len(items)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
