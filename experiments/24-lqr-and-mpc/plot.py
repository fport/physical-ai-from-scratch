"""Plot the LQR region of attraction and the MPPI horizon sweep from output.txt
(writes lqr_region_mpc_horizon.png next to this script).

Run:  uv run python experiments/24-lqr-and-mpc/plot.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUTCOMES = ["ok", "track end", "fell"]
COLORS = ["#4c9a5a", "#e0a43a", "#c8553d"]


def parse():
    """Part B grid {theta0: [outcome per force]}, forces, and C1/C2 rows (horizon, worst |theta| deg)."""
    grid, forces, horizon, section = {}, [], {"C1": [], "C2": []}, None
    for line in open(os.path.join(HERE, "output.txt"), encoding="utf-8"):
        if line.startswith(("Part B", "C1", "C2")):
            section = line.split()[0] + (line.split()[1] if line.startswith("Part") else "")
            continue
        cells = [c.strip() for c in line.split("|")]
        if section == "PartB" and cells[0].startswith("theta0"):
            forces = [int(c.split("=")[1].split()[0]) for c in cells[1:]]
        elif section == "PartB" and cells[0].isdigit():
            grid[int(cells[0])] = cells[1:]
        elif section in horizon and len(cells) == 7 and cells[0].replace(".", "").isdigit():
            horizon[section].append((float(cells[0]), float(cells[5])))
    return grid, forces, horizon


def main():
    grid, forces, horizon = parse()
    thetas = sorted(grid)
    codes = [[OUTCOMES.index(o) for o in grid[t]] for t in thetas]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), dpi=150)
    ax = axes[0]
    ax.imshow(codes, cmap=ListedColormap(COLORS), vmin=0, vmax=2, aspect="auto", origin="lower")
    ax.set_xticks(range(len(forces)), labels=[f"{f} N" for f in forces])
    ax.set_yticks(range(len(thetas)), labels=[str(t) for t in thetas], fontsize=7)
    ax.set_xlabel("force limit F")
    ax.set_ylabel("release angle (deg)")
    ax.set_title("LQR region of attraction (pole released at rest)", fontsize=9)
    ax.legend(handles=[Patch(color=c, label=o) for c, o in zip(COLORS, OUTCOMES)], frameon=False, fontsize=7,
              loc="upper left", bbox_to_anchor=(1.0, 1.0))

    ax = axes[1]
    for key, label, mark in [("C1", "MPPI only", "o-"), ("C2", "MPPI, then LQR below 20 deg", "s--")]:
        pts = horizon[key]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], mark, ms=4, label=label)
    ax.axhline(10, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("MPC horizon (s)")
    ax.set_ylabel("median worst |angle|, last 1 s (deg)")
    ax.set_title("swing-up: too short never catches it", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=7)

    fig.suptitle("Cart-pole: where linear feedback works, and what the MPC horizon buys", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "lqr_region_mpc_horizon.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
