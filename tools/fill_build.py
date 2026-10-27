"""Fill a Build it draft: {{src:path:name}} -> the exact source of that function/class from physai,
{{out:Step N}} -> that step's block from the lesson's build_output.txt.

Run:  uv run python tools/fill_build.py <draft.md> <experiments/NN-slug/build_output.txt> > section.md
Keeps the lesson's code identical to the package (tools/check_book.py verifies it).
"""
import ast
import re
import sys

ROOT = __file__.rsplit("/tools/", 1)[0]


def source(path, name):
    text = open(f"{ROOT}/{path}", encoding="utf-8").read()
    for node in ast.parse(text).body:
        if getattr(node, "name", None) == name:
            start = (node.decorator_list[0].lineno if getattr(node, "decorator_list", None) else node.lineno) - 1
            return "```python\n" + "\n".join(text.splitlines()[start:node.end_lineno]) + "\n```"
    raise SystemExit(f"{name} not found in {path}")


def output(text, step):
    m = re.search(rf"^== {re.escape(step)}\..*?(?=^== |\Z)", text, re.M | re.S)
    if not m:
        raise SystemExit(f"{step} not found in output")
    return m.group(0).rstrip()


draft, out = open(sys.argv[1], encoding="utf-8").read(), open(sys.argv[2], encoding="utf-8").read()
draft = re.sub(r"\{\{src:([^:}]+):(\w+)\}\}", lambda m: source(m.group(1), m.group(2)), draft)
draft = re.sub(r"\{\{out:([^}]+)\}\}", lambda m: output(out, m.group(1)), draft)
sys.stdout.write(draft)
