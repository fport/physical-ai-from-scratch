"""Plot section C of output.txt (success and torque rate vs action-rate penalty weight) without re-running.

Run:  uv run python experiments/35-reward-design/plot.py   -> writes success_vs_penalty.png
"""
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROW = re.compile(r"^\s+([\d.]+)\s+(\d+/\d+/\d+)\s+(\d+)/300 \[\s*([\d.]+),\s*([\d.]+)\]\s+([\d.]+)\s+(\d+)\s+([\d.]+)\s*$")


def parse_section_c(text):
    """Rows of (w, [per-seed successes], pooled %, wilson lo, wilson hi, torque rate) from section [C]."""
    rows = []
    for line in text.split("[C]")[1].splitlines():
        m = ROW.match(line)
        if m:
            rows.append((float(m[1]), [int(s) for s in m[2].split("/")], int(m[3]) / 3, float(m[4]),
                         float(m[5]), int(m[7])))
    return rows


def main():
    rows = parse_section_c(open(os.path.join(HERE, "output.txt"), encoding="utf-8").read())
    x = list(range(len(rows)))
    labels = [f"{r[0]:g}" for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), dpi=150)

    ax = axes[0]
    y = [r[2] for r in rows]
    err = [[r[2] - r[3] for r in rows], [r[4] - r[2] for r in rows]]
    ax.errorbar(x, y, yerr=err, fmt="o-", ms=5, capsize=4, label="pooled (Wilson 95%)")
    for i, r in enumerate(rows):
        ax.plot([i + d for d in (-0.1, 0, 0.1)], r[1], "x", color="C1", ms=6,
                label="one seed" if i == 0 else None)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 105)
    ax.set_xlabel("penalty weight w on ||a_t − a_(t−1)||²")
    ax.set_ylabel("success (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8, loc="lower left")

    ax = axes[1]
    ax.plot(x, [r[5] for r in rows], "s-", ms=5, color="C2")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1100)
    ax.set_xlabel("penalty weight w")
    ax.set_ylabel("torque rate (N·m/s)")
    ax.grid(True, alpha=0.3)

    fig.suptitle("A smoothness penalty costs success long before it buys smooth torque", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "success_vs_penalty.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
