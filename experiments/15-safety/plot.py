"""Lesson 15 figure: impact peak and steady push vs torque cap and vs relative-target limit, from output.txt.

Writes safeguard_sweeps.png next to this script. Run:  python plot.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def block(text, tag, next_tag):
    return text.split(tag)[1].split(next_tag)[0].splitlines()


def numeric_rows(lines, n):
    rows = []
    for line in lines:
        parts = line.split()
        if len(parts) == n:
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                pass
    return rows


def main():
    text = open(os.path.join(HERE, "output.txt"), encoding="utf-8").read()
    cap = numeric_rows(block(text, "[B]", "[C]"), 4)          # cap, peak, hold, speed
    rel = numeric_rows(block(text, "[C]", "[D]"), 4)          # delta, speed at impact, peak, hold
    no_limit = next(line.split() for line in block(text, "[C]", "[D]") if line.split()[:1] == ["none"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), dpi=150)

    ax = axes[0]
    ax.plot([r[0] for r in cap], [r[1] for r in cap], "o-", ms=4, label="impact peak")
    ax.plot([r[0] for r in cap], [r[2] for r in cap], "s-", ms=4, label="steady push")
    ax.set_xlabel("torque cap on shoulder_pan (N·m)")
    ax.set_ylabel("force on the post (N)")
    ax.set_title("B. Torque cap: lowers both", fontsize=9)

    ax = axes[1]
    ax.plot([r[0] for r in rel], [r[2] for r in rel], "o-", ms=4, label="impact peak")
    ax.plot([r[0] for r in rel], [r[3] for r in rel], "s-", ms=4, label="steady push")
    ax.axhline(float(no_limit[2]), color="gray", lw=0.8, ls="--")
    ax.text(0.03, float(no_limit[2]) * 0.97, "no limit", fontsize=8, color="gray", va="top")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("relative-target limit δ per 20 ms tick (rad), tighter →")
    ax.set_title("C. Rate limit: lowers the impact, not the push", fontsize=9)
    for ax in axes:
        ax.set_ylim(bottom=0)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "safeguard_sweeps.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
