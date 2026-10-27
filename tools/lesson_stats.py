"""Words of prose (outside code fences) and code lines per lesson page, against the length target.

Run:  uv run python tools/lesson_stats.py [NN ...]      (default: all lessons, English pages)
      uv run python tools/lesson_stats.py --tr 12 18     (Turkish pages)
"""
import os
import re
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = 3000   # prose words per page, Build it and math details included


def stats(text):
    code = re.findall(r"^```.*?^```", text, re.M | re.S)
    prose = re.sub(r"^```.*?^```", "", text, flags=re.M | re.S)
    prose = re.sub(r"<!-- meta -->.*?<!-- /meta -->", "", prose, flags=re.S)
    return len(prose.split()), sum(c.count("\n") - 1 for c in code)


def main():
    args = sys.argv[1:]
    tr = "--tr" in args
    want = {a for a in args if a != "--tr"}
    c = yaml.safe_load(open(os.path.join(ROOT, "curriculum.yaml"), encoding="utf-8"))
    print(f"{'lesson':<44s} {'words':>6s} {'code':>5s}  build-it")
    for part in c["parts"]:
        for le in part["lessons"]:
            if want and le["slug"][:2] not in want:
                continue
            page = os.path.join(ROOT, "tr" if tr else "", part["dir"], le["slug"] + ".md")
            text = open(page, encoding="utf-8").read()
            words, code = stats(text)
            has = "yes" if re.search(r"^## (Build it|Sıfırdan yaz)$", text, re.M) else "-"
            flag = "  over" if words > TARGET else ""
            print(f"{le['slug']:<44s} {words:6d} {code:5d}  {has}{flag}")


if __name__ == "__main__":
    main()
