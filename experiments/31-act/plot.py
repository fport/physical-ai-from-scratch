"""Plot success and stuck episodes vs chunk size from output.txt (writes chunk_size.png next to this script).

Run:  uv run python experiments/31-act/plot.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main_rows():
    """Rows of the main table as {scheme: [(k, successes, n, lo %, hi %, stuck)]}; k=1 counts for both schemes."""
    rows = {"open-loop": [], "ensembled": []}
    for line in open(os.path.join(HERE, "output.txt"), encoding="utf-8"):
        if line.startswith("Chunk predicted"):
            break
        if not line.startswith("k="):
            continue
        cells = line.replace("[", " ").replace("]", " ").replace(",", " ").replace("k=", "").split()
        k, n_ok, n = int(cells[0]), *(int(v) for v in cells[2].split("/"))
        lo, hi, stuck = float(cells[4]), float(cells[5]), int(cells[9])
        for scheme in ([cells[1]] if k > 1 else rows):
            rows[scheme].append((k, n_ok, n, lo, hi, stuck))
    return rows


def main():
    rows = main_rows()
    style = {"open-loop": ("s--", "tab:orange"), "ensembled": ("o-", "tab:blue")}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), dpi=150)
    for scheme, data in rows.items():
        k = [r[0] for r in data]
        pct = [100 * r[1] / r[2] for r in data]
        err = [[p - r[3] for p, r in zip(pct, data)], [r[4] - p for p, r in zip(pct, data)]]
        fmt, color = style[scheme]
        axes[0].errorbar(k, pct, yerr=err, fmt=fmt, ms=4, capsize=3, color=color, label=scheme)
        axes[1].plot(k, [r[5] for r in data], fmt, ms=4, color=color, label=scheme)
    axes[0].set_ylabel("success (% of 600 episodes)")
    axes[0].set_title("success (Wilson 95%)", fontsize=9)
    axes[1].set_ylabel("stuck at the target (of 600)")
    axes[1].set_title("the pause trap: k <= 8 stays at the target", fontsize=9)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 4, 8, 16, 32])
        ax.set_xticklabels(["1", "4", "8", "16", "32"])
        ax.set_xlabel("chunk size k (steps at 50 Hz)")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Planar reach-pause-drop in simulation: action chunk size vs outcome", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "chunk_size.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
