"""Plot end-to-end success and perception error vs unmodelled camera tilt (part B)
from output.txt (writes success_vs_tilt.png next to this script). Does not re-run the experiment.

Run:  uv run python experiments/44-capstone/plot.py
"""
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CONTRACT_MM = 8.0    # perception contract of the lesson
ROW = re.compile(r"^(\d+\.\d)\s+([\d.]+)\s+(\d+)/(\d+)\s+([\d.]+)\s+\[([\d.]+), ([\d.]+)\]")


def parse():
    """Rows of the tilt table: (tilt deg, median err mm, k, n, rate, wilson lo, wilson hi)."""
    rows, in_table = [], False
    for line in open(os.path.join(HERE, "output.txt"), encoding="utf-8"):
        if line.startswith("tilt deg"):
            in_table = True
            continue
        m = ROW.match(line) if in_table else None
        if m:
            t, err, k, n, rate, lo, hi = m.groups()
            rows.append((float(t), float(err), int(k), int(n), float(rate), float(lo), float(hi)))
    return rows


def main():
    rows = parse()
    tilt = [r[0] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), dpi=150)

    ax = axes[0]
    rate = [r[4] for r in rows]
    ax.errorbar(tilt, rate, yerr=[[r[4] - r[5] for r in rows], [r[6] - r[4] for r in rows]],
                fmt="o-", ms=4, capsize=3)
    for r in rows:
        ax.annotate(f"{r[2]}/{r[3]}", (r[0], r[6]), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=7)
    ax.set_ylim(0, 1.1)
    ax.set_xlim(-0.2, 2.2)
    ax.set_xticks(tilt)
    ax.set_xlabel("unmodelled camera tilt (deg)")
    ax.set_ylabel("end-to-end success (Wilson 95%)")
    ax.set_title("qwen2.5:7b, 15 episodes per tilt", fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(tilt, [r[1] for r in rows], "s-", ms=4)
    ax.axhline(CONTRACT_MM, color="gray", lw=0.8, ls=":")
    ax.text(tilt[0], CONTRACT_MM + 0.6, "perception contract 8 mm", fontsize=7, color="gray")
    ax.set_xticks(tilt)
    ax.set_xlabel("unmodelled camera tilt (deg)")
    ax.set_ylabel("median estimate error (mm)")
    ax.set_title("the error grows linearly; success falls off a cliff", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Capstone: a small calibration error is invisible until the grasp misses", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "success_vs_tilt.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
