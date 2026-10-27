"""Plot closed-loop success (pooled over 3 seeds, Wilson 95%) and open-loop RMSE versus the number of
demonstrations from output.txt part A (writes success_vs_demos.png next to this script).

Run:  uv run python experiments/28-behavior-cloning/plot.py
"""
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def parse():
    """Part A rows (demos, pooled k, pooled n, Wilson lo, hi, open-loop RMSE deg)."""
    rows, section = [], None
    pat = re.compile(r"\s*(\d+)\s+\d+\s+\d+/20\s+\d+/20\s+\d+/20\s+(\d+)/(\d+)\s+\[([\d.]+), ([\d.]+)\]\s+([\d.]+)")
    for line in open(os.path.join(HERE, "output.txt"), encoding="utf-8"):
        if line.startswith("["):
            section = line[1]
            continue
        m = pat.match(line)
        if section == "A" and m:
            rows.append((int(m[1]), int(m[2]), int(m[3]), float(m[4]), float(m[5]), float(m[6])))
    return rows


def main():
    rows = parse()
    d = [r[0] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), dpi=150)

    ax = axes[0]
    rate = [r[1] / r[2] for r in rows]
    err = [[p - r[3] for p, r in zip(rate, rows)], [r[4] - p for p, r in zip(rate, rows)]]
    ax.errorbar(d, rate, yerr=err, fmt="o-", ms=4, capsize=3)
    for x, r, p in zip(d, rows, rate):
        ax.annotate(f"{r[1]}/{r[2]}", (x, p), textcoords="offset points", xytext=(5, -10), fontsize=7)
    ax.set_xscale("log")
    ax.set_xticks(d, labels=[str(k) for k in d])
    ax.set_ylim(0.4, 1.02)
    ax.set_xlim(4, 150)
    ax.set_xlabel("demonstrations")
    ax.set_ylabel("closed-loop success (Wilson 95%)")
    ax.set_title("success saturates at 20 demos", fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(d, [r[5] for r in rows], "s-", ms=4, color="#c8553d")
    for x, r in zip(d, rows):
        ax.annotate(f"{r[5]:.3f}", (x, r[5]), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(d, labels=[str(k) for k in d])
    ax.set_yticks([0.3, 0.5, 1, 2, 4], labels=["0.3", "0.5", "1", "2", "4"])
    ax.set_ylim(0.3, 6)
    ax.minorticks_off()
    ax.set_xlim(4, 150)
    ax.set_xlabel("demonstrations")
    ax.set_ylabel("open-loop action RMSE (deg)")
    ax.set_title("the loss keeps falling", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Behavior cloning on the SO-101 pick: 3 seeds x 20 test positions per point", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "success_vs_demos.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
