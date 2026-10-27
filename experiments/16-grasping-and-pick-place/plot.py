"""Lesson 16 figure: grasp outcomes vs friction (part A) and success vs friction-cone ratio (part C).

Parsed from output.txt; writes grasp_outcomes.png next to this script. Run:  python plot.py
"""
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROW = re.compile(r"^\s*([\d.]+)\s+(\d+)/(\d+)\s+\[.*?\]\s+(\S+)\s+(\S+)\s+(\S+)\s+(\d+) / (\d+) / (\d+); (\d+)")


def parse(text, tag, next_tag):
    """Rows: value, successes, trials, ratio (or None), held, not closed, dropped."""
    chunk = text.split(tag)[1]
    if next_tag:
        chunk = chunk.split(next_tag)[0]
    rows = []
    for line in chunk.splitlines():
        m = ROW.match(line)
        if m:
            g = m.groups()
            ratio = None if g[5] == "-" else float(g[5])
            rows.append((float(g[0]), int(g[1]), int(g[2]), ratio, int(g[6]), int(g[7]), int(g[8])))
    return rows


def main():
    text = open(os.path.join(HERE, "output.txt"), encoding="utf-8").read()
    a = parse(text, "[A]", "[B]")
    c = parse(text, "[C]", None)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), dpi=150)

    ax = axes[0]
    x = range(len(a))
    held = [r[4] for r in a]
    not_closed = [r[5] for r in a]
    dropped = [r[6] for r in a]
    ax.bar(x, held, color="tab:green")
    ax.bar(x, not_closed, bottom=held, color="tab:gray")
    ax.bar(x, dropped, bottom=[h + n for h, n in zip(held, not_closed)], color="tab:red")
    ax.set_xticks(list(x), [f"{r[0]:g}" for r in a], fontsize=7)
    ax.set_xlabel("cube friction μ (30 g, jaw cap 0.5 N·m)")
    ax.set_ylabel("trials (of 10)")
    ax.set_title("A. Low friction fails while closing", fontsize=9)
    ax.legend(handles=[Patch(color=c, label=lab) for c, lab in [("tab:green", "held"), ("tab:gray", "jaw never settled"),
                                                                ("tab:red", "dropped in lift")]],
              frameon=True, fontsize=7, loc="center right")

    ax = axes[1]
    ratios = [r[3] for r in c]
    rate = [r[1] / r[2] for r in c]
    ax.plot(ratios, rate, "o", ms=5)
    failed = 0
    for r in c:
        if r[1]:
            offset = (0, 6)
        else:
            offset = (0, -12 - 9 * (failed % 2))
            failed += 1
        ax.annotate(f"{r[0]:g} kg", (r[3], r[1] / r[2]), textcoords="offset points", xytext=offset,
                    fontsize=6, ha="center")
    ax.axvline(1.0, color="k", lw=0.8, ls="--")
    ax.text(1.03, 0.5, "cone limit = 1", fontsize=8)
    ax.set_xscale("log")
    ax.set_ylim(-0.25, 1.15)
    ax.set_xlabel("friction-cone ratio μ(N1+N2)/(m g)  (μ 0.3, cap 0.5 N·m)")
    ax.set_ylabel("success rate")
    ax.set_title("C. Heavier cubes: success flips just above ratio 1", fontsize=9)
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "grasp_outcomes.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
