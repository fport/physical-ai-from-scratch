"""Plot interval width and coverage versus the number of trials from output.txt, part B
(writes interval_width_vs_trials.png next to this script).

Run:  uv run python experiments/25-evaluating-policies/plot.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def parse():
    """Part B rows: (n, mean Wilson width, Wilson coverage, Wald coverage)."""
    rows, section = [], None
    for line in open(os.path.join(HERE, "output.txt"), encoding="utf-8"):
        if line.startswith("["):
            section = line[1]
            continue
        cells = line.split()
        if section == "B" and cells and cells[0].isdigit():
            rows.append((int(cells[0]), float(cells[1]), float(cells[2]), float(cells[3])))
    return rows


def main():
    rows = parse()
    n = [r[0] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), dpi=150)

    ax = axes[0]
    ax.plot(n, [r[1] for r in rows], "o-", ms=4, label="mean Wilson 95% width (measured)")
    ref = [rows[-1][1] * (rows[-1][0] / k) ** 0.5 for k in n]
    ax.plot(n, ref, ":", color="gray", label="shape 1/sqrt(n)")
    for k, w in zip(n, [r[1] for r in rows]):
        ax.annotate(f"{w:.2f}", (k, w), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xscale("log")
    ax.set_xticks(n, labels=[str(k) for k in n])
    ax.set_xlim(4, 140)
    ax.set_xlabel("trials per evaluation n")
    ax.set_ylabel("interval width (success rate)")
    ax.set_title("the interval narrows slowly: 4x trials, half the width", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    ax.plot(n, [r[2] for r in rows], "o-", ms=4, label="Wilson")
    ax.plot(n, [r[3] for r in rows], "s--", ms=4, label="Wald (textbook)")
    ax.axhline(0.95, color="gray", lw=0.8, ls=":", label="target 0.95")
    ax.set_xscale("log")
    ax.set_xticks(n, labels=[str(k) for k in n])
    ax.set_ylim(0.7, 1.0)
    ax.set_xlabel("trials per evaluation n")
    ax.set_ylabel("how often the interval contains p")
    ax.set_title("coverage: Wald fails at small n", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=7, loc="lower right")

    fig.suptitle("Policy A (p = 0.737), 2000 resampled evaluations per n", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "interval_width_vs_trials.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
