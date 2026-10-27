"""Plot the edge-check resolution sweep from output.txt (writes resolution_vs_collisions.png next to this script).

Run:  uv run python experiments/22-motion-planning/plot.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def part_a_rows():
    """Rows of 'Part A' as (res, med checks, paths in collision, worst penetration mm)."""
    rows, inside = [], False
    for line in open(os.path.join(HERE, "output.txt"), encoding="utf-8"):
        if line.startswith("Part A"):
            inside = True
            continue
        if inside and line.startswith("Part B"):
            break
        cells = [c.strip() for c in line.split("|")]
        if inside and len(cells) == 8 and cells[0].replace(".", "").isdigit():
            rows.append((float(cells[0]), int(cells[3]), int(cells[6].split("/")[0]), float(cells[7])))
    return rows


def main():
    rows = part_a_rows()
    res = [r[0] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), dpi=150)

    axes[0].plot(res, [r[1] for r in rows], "o-", ms=4, color="tab:blue")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("median collision checks")
    axes[0].set_title("cost: falls with coarser checks", fontsize=9)

    axes[1].plot(res, [r[2] for r in rows], "o-", ms=4, color="tab:red", label="paths in collision (of 100)")
    axes[1].plot(res, [r[3] for r in rows], "s--", ms=4, color="tab:gray", label="worst penetration (mm)")
    axes[1].set_title("truth (dense re-check): every planner said 100/100", fontsize=9)
    axes[1].legend(frameon=False, fontsize=8)

    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("edge-check resolution (rad)")
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle("RRT-Connect around the box: fewer checks, more paths through the obstacle", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "resolution_vs_collisions.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
