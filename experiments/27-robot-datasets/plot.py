"""Plot validation and test error for a frame-level vs an episode-level split from output.txt part C
(writes split_val_vs_test.png next to this script).

Run:  uv run python experiments/27-robot-datasets/plot.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def parse():
    """Part C rows {split: [(episodes, val RMSE, test RMSE, test/val)]}."""
    rows, section = {"frame": [], "episode": []}, None
    for line in open(os.path.join(HERE, "output.txt"), encoding="utf-8"):
        if line.startswith("["):
            section = line[1]
            continue
        cells = line.split()
        if section == "C" and len(cells) == 5 and cells[0].isdigit():
            rows[cells[1]].append((int(cells[0]), float(cells[2]), float(cells[3]), float(cells[4])))
    return rows


def main():
    rows = parse()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), dpi=150)
    colors = {"frame": "#c8553d", "episode": "#3d6fc8"}

    ax = axes[0]
    for split, pts in rows.items():
        e = [p[0] for p in pts]
        ax.plot(e, [p[1] for p in pts], "o-", ms=4, color=colors[split], label=f"{split} split: validation")
        ax.plot(e, [p[2] for p in pts], "s--", ms=4, color=colors[split], alpha=0.7,
                label=f"{split} split: test (unseen episodes)")
    ax.set_xscale("log")
    e = [p[0] for p in rows["frame"]]
    ax.set_xticks(e, labels=[str(k) for k in e])
    ax.set_xlabel("episodes used for training + validation")
    ax.set_ylabel("action RMSE (deg)")
    ax.set_title("frame-split validation stays flat", fontsize=9)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=6.5)

    ax = axes[1]
    w = 0.35
    for i, (split, pts) in enumerate(rows.items()):
        x = [j + (i - 0.5) * w for j in range(len(pts))]
        ax.bar(x, [p[3] for p in pts], width=w, color=colors[split], label=f"{split} split")
        for xi, p in zip(x, pts):
            ax.annotate(f"{p[3]:.2f}", (xi, p[3]), textcoords="offset points", xytext=(0, 2), fontsize=7,
                        ha="center")
    ax.axhline(1.0, color="gray", lw=0.8, ls=":")
    ax.set_xticks(range(len(e)), labels=[f"{k} episodes" for k in e])
    ax.set_ylabel("test RMSE / validation RMSE")
    ax.set_title("how much validation understates the real error", fontsize=9)
    ax.legend(frameon=False, fontsize=7)

    fig.suptitle("State -> action MLP on the lesson-26 demos: what the validation number promises", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "split_val_vs_test.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
