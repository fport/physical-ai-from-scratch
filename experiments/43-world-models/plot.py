"""Plot open-loop prediction error vs horizon (part A) and the replanning sweep (part E)
from output.txt (writes error_vs_horizon.png next to this script).

Run:  uv run python experiments/43-world-models/plot.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def parse():
    """Part A rows (H steps, mean, median angle err) and part E rows (k, true tail err, learned tail err)."""
    a, e, section = [], [], None
    for line in open(os.path.join(HERE, "output.txt"), encoding="utf-8"):
        if line.startswith("["):
            section = line[1]
            continue
        cells = line.split()
        if not cells or not cells[0].isdigit():
            continue
        if section == "A" and len(cells) == 5:
            a.append((int(cells[0]), float(cells[2]), float(cells[3])))
        elif section == "E" and len(cells) == 6:
            e.append((int(cells[0]), float(cells[3]), float(cells[5]), cells[2], cells[4]))
    return a, e


def main():
    a, e = parse()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), dpi=150)

    ax = axes[0]
    ax.plot([r[0] for r in a], [r[1] for r in a], "o-", ms=4, label="mean")
    ax.plot([r[0] for r in a], [r[2] for r in a], "s--", ms=4, label="median")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("open-loop rollout horizon H (steps of 0.05 s)")
    ax.set_ylabel("|angle error| (rad)")
    ax.set_title("learned model run on its own outputs", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    ks = [r[0] for r in e]
    ax.plot(ks, [r[1] for r in e], "o-", ms=4, label="MPC in true simulator")
    ax.plot(ks, [r[2] for r in e], "s--", ms=4, label="MPC in learned model")
    for k, t, lrn, st, sl in e:
        ax.annotate(f"{st}\n{sl}", (k, 0.012), ha="center", fontsize=6)
    ax.axhline(0.2, color="gray", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(ks, labels=[str(k) for k in ks])
    ax.set_xlabel("actions executed open loop before replanning (k)")
    ax.set_ylabel("mean tail error (rad)")
    ax.set_ylim(0.008, 2.5)
    ax.text(1.25, 0.0125, "success (true / learned):", fontsize=6, ha="left")
    ax.set_title("swing-up with H = 30", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False, fontsize=7, loc="upper left")

    fig.suptitle("Error compounds with horizon; frequent replanning hides it", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "error_vs_horizon.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
