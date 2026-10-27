"""Plot cost and replay error versus knot count from output.txt (writes cost_vs_knots.png next to this script).

Run:  uv run python experiments/23-trajectory-optimization/plot.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def sweep_rows():
    """Converged rows of the knot sweep as {method: [(N, cost, replay err theta deg)]}."""
    out = {"shooting": [], "collocation": []}
    for line in open(os.path.join(HERE, "output.txt"), encoding="utf-8"):
        if line.startswith("Break it"):
            break
        cells = [c.strip() for c in line.split("|")]
        if len(cells) == 9 and cells[0] in out and cells[2] == "True":
            out[cells[0]].append((int(cells[1]), float(cells[4]), float(cells[6])))
    return out


def main():
    rows = sweep_rows()
    style = {"shooting": ("o-", "tab:blue"), "collocation": ("s--", "tab:orange")}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), dpi=150)
    for name, pts in rows.items():
        mark, color = style[name]
        n = [p[0] for p in pts]
        axes[0].plot(n, [p[1] for p in pts], mark, ms=4, color=color, label=name)
        axes[1].plot(n, [p[2] for p in pts], mark, ms=4, color=color, label=name)
    axes[0].set_ylabel("cost J (solver's answer)")
    axes[0].set_title("both report the same optimum", fontsize=9)
    axes[1].set_ylabel("goal error when replayed (deg)")
    axes[1].set_title("only shooting's plan actually arrives", fontsize=9)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks([10, 20, 40, 80], labels=["10", "20", "40", "80"])
        ax.set_xlabel("knots N (converged runs)")
        ax.grid(True, which="both", alpha=0.3)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Pendulum swing-up: same cost, different truth", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "cost_vs_knots.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
