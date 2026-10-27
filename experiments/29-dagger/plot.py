"""Plot DAgger iterations vs plain BC from output.txt (writes dagger_iterations.png next to this script).

Run:  uv run python experiments/29-dagger/plot.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def part_b_rows():
    """Rows of '[B]' as dicts: episodes, DAgger (k, n, lo, hi, dist) and, where present, plain BC (k, n, dist)."""
    rows, inside = [], False
    for line in open(os.path.join(HERE, "output.txt"), encoding="utf-8"):
        if line.startswith("[B]"):
            inside = True
            continue
        if inside and line.startswith("[A]"):
            break
        cells = line.split()
        if not inside or len(cells) < 7 or not cells[0].isdigit():
            continue
        k, n = (int(v) for v in cells[3].split("/"))
        lo, hi = float(cells[4].strip("[,")), float(cells[5].strip("]"))
        row = {"episodes": int(cells[1]), "dagger": (k, n, lo, hi), "dagger_dist": float(cells[6])}
        if len(cells) >= 12:
            bk, bn = (int(v) for v in cells[8].split("/"))
            row["bc"] = (bk, bn)
            row["bc_dist"] = float(cells[11])
        rows.append(row)
    return rows


def main():
    rows = part_b_rows()
    ep = [r["episodes"] for r in rows]
    rate = [r["dagger"][0] / r["dagger"][1] for r in rows]
    err = [[p - r["dagger"][2] for p, r in zip(rate, rows)], [r["dagger"][3] - p for p, r in zip(rate, rows)]]
    bc = [r for r in rows if "bc" in r]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), dpi=150)
    axes[0].errorbar(ep, rate, yerr=err, fmt="o-", ms=4, capsize=3, color="tab:blue", label="DAgger (Wilson 95%)")
    axes[0].plot([r["episodes"] for r in bc], [r["bc"][0] / r["bc"][1] for r in bc], "s--", ms=4,
                 color="tab:red", label="plain BC, more demos")
    axes[0].set_ylabel("success rate (3 seeds x 20 tests)")
    axes[0].set_ylim(-0.03, 1.0)
    axes[0].set_title("same expert budget, spent on the learner's states", fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")

    axes[1].plot(ep, [r["dagger_dist"] for r in rows], "o-", ms=4, color="tab:blue", label="DAgger")
    axes[1].plot([r["episodes"] for r in bc], [r["bc_dist"] for r in bc], "s--", ms=4, color="tab:red",
                 label="plain BC")
    axes[1].set_ylabel("distance to nearest demo frame (deg)")
    axes[1].set_ylim(0, None)
    axes[1].set_title("how far the visited arm states are from the data", fontsize=9)
    axes[1].legend(frameon=False, fontsize=8)

    for ax in axes:
        ax.set_xlabel("expert-labelled episodes (DAgger: +10 per iteration)")
        ax.set_xticks(ep)
        ax.grid(True, alpha=0.3)
    fig.suptitle("SO-101 pick in simulation: DAgger iterations vs plain behavior cloning", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "dagger_iterations.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
