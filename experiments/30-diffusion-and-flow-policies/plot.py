"""Plot the Euler-step sweep of the flow-matching policy from output.txt (writes flow_steps.png next to this script).

Run:  uv run python experiments/30-diffusion-and-flow-policies/plot.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def part_c_rows():
    """Rows of 'Part C' as (steps, collided, lo %, hi %, mean |x| at y=0)."""
    rows, inside = [], False
    for line in open(os.path.join(HERE, "output.txt"), encoding="utf-8"):
        if line.startswith("Part C"):
            inside = True
            continue
        cells = line.replace("[", " ").replace("]", " ").replace(",", " ").split()
        if inside and len(cells) == 10 and cells[0].isdigit():
            rows.append((int(cells[0]), int(cells[1].split("/")[0]), float(cells[3]), float(cells[4]),
                         float(cells[9])))
    return rows


def main():
    rows = part_c_rows()
    steps = [r[0] for r in rows]
    pct = [r[1] / 2 for r in rows]  # 200 trials per row -> percent
    err = [[p - r[2] for p, r in zip(pct, rows)], [r[3] - p for p, r in zip(pct, rows)]]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), dpi=150)
    axes[0].errorbar(steps, pct, yerr=err, fmt="o-", ms=4, capsize=3, color="tab:blue",
                     label="flow policy (Wilson 95%)")
    axes[0].axhline(100, color="tab:red", ls="--", lw=1, label="MSE regression")
    axes[0].set_ylabel("collisions (% of 200 trials)")
    axes[0].set_title("1 step = the mean; more steps = samples from both modes", fontsize=9)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(steps, [r[4] for r in rows], "o-", ms=4, color="tab:blue", label="flow policy")
    axes[1].axhline(0.55, color="tab:green", ls="--", lw=1, label="demonstrations (0.55)")
    axes[1].axhline(0.30, color="tab:gray", ls=":", lw=1, label="obstacle radius (0.30)")
    axes[1].set_ylabel("mean |x| at the obstacle (y = 0)")
    axes[1].set_ylim(0, 0.65)
    axes[1].set_title("lateral clearance grows toward the demos", fontsize=9)
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")

    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(steps)
        ax.set_xticklabels([str(s) for s in steps])
        ax.set_xlabel("Euler integration steps (= network calls)")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Two-mode obstacle task: flow matching vs number of sampling steps", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "flow_steps.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
