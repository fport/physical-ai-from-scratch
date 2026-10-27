"""Plot the action-tokenization sweep (part B) from output.txt
(writes tokenization_vs_bins.png next to this script).

Run:  uv run python experiments/41-vla/plot.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CUBE_MM = 30.0       # cube edge (lesson text and experiment.py)
LIFT_CM = 4.0        # success test: cube bottom >= 4 cm


def parse():
    """Rows of part B: (bins, mean tip err mm, max tip err mm, success, cube bottom cm)."""
    rows, in_table = [], False
    for line in open(os.path.join(HERE, "output.txt"), encoding="utf-8"):
        cells = line.split()
        if cells[:2] == ["bins", "bits/dim"]:
            in_table = True
            continue
        if in_table and len(cells) == 8 and cells[0].isdigit():
            rows.append((int(cells[0]), float(cells[4]), float(cells[5]), cells[6] == "True",
                         max(0.0, float(cells[7]))))
    return sorted(rows)


def main():
    rows = parse()
    bins = [r[0] for r in rows]
    ok = [r[3] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), dpi=150)
    ax = axes[0]
    ax.plot(bins, [r[2] for r in rows], "o-", ms=4, label="max tip error")
    ax.plot(bins, [r[1] for r in rows], "s--", ms=4, label="mean tip error")
    ax.axhline(CUBE_MM, color="gray", lw=0.8, ls=":")
    ax.text(bins[-1], CUBE_MM * 1.12, "cube edge 30 mm", ha="right", fontsize=7, color="gray")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(bins, labels=[str(b) for b in bins])
    ax.set_xlabel("bins per action dimension")
    ax.set_ylabel("gripper tip error (mm)")
    ax.set_title("each bit removed doubles the error", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    colors = ["#4c9a5a" if s else "#c8553d" for s in ok]
    ax.bar(range(len(bins)), [r[4] for r in rows], color=colors)
    for i, s in enumerate(ok):
        if not s:
            ax.text(i, 0.3, "not\nlifted", ha="center", fontsize=7, color="#c8553d")
    ax.axhline(LIFT_CM, color="gray", lw=0.8, ls=":")
    ax.text(-0.4, 8.0, "dotted line: success needs bottom >= 4 cm", ha="left", fontsize=7, color="gray")
    ax.set_xticks(range(len(bins)), labels=[str(b) for b in bins])
    ax.set_xlabel("bins per action dimension")
    ax.set_ylabel("cube bottom after replay (cm)")
    ax.set_ylim(0, 8.5)
    ax.set_title("open-loop replay: lifted (green) or not (red)", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Tokenizing lesson 16's SO-101 pick: resolution is cheap until the error exceeds the cube",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "tokenization_vs_bins.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
