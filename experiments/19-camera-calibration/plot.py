"""Plot the views and tilt sweeps from output.txt (writes fit_vs_truth.png next to this script).

Run:  uv run python experiments/19-camera-calibration/plot.py
"""
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullFormatter, ScalarFormatter  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def block_rows(text, start, end, ncols):
    """Numeric rows with ncols columns between the markers start and end."""
    rows = []
    for line in text.split(start)[1].split(end)[0].splitlines():
        cells = line.split()
        if len(cells) == ncols and all(re.fullmatch(r"[\d.]+", c) for c in cells):
            rows.append([float(c) for c in cells])
    return rows


def panel(ax, x, f_err, train, xlabel, title):
    ax.plot(x, f_err, "o-", ms=4, color="tab:red", label="focal length error (%)")
    ax.set_yscale("log")
    ax.set_ylabel("f_x error vs ground truth (%)", color="tab:red")
    ax.set_xlabel(xlabel)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title(title, fontsize=9)
    tw = ax.twinx()
    tw.plot(x, train, "s--", ms=4, color="tab:blue")
    tw.set_ylim(0, 0.15)
    tw.set_ylabel("training reprojection RMS (px)", color="tab:blue")


def main():
    text = open(os.path.join(HERE, "output.txt"), encoding="utf-8").read()
    views = block_rows(text, "[A2]", "all 60 views", 7)
    tilt = block_rows(text, "[A3]", "[A4]", 4)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), dpi=150)
    panel(axes[0], [r[0] for r in views], [r[3] for r in views], [r[1] for r in views],
          "number of board views", "more views: f error falls, fit error flat")
    axes[0].set_xscale("log")
    axes[0].set_xticks([r[0] for r in views])
    axes[0].xaxis.set_major_formatter(ScalarFormatter())
    axes[0].xaxis.set_minor_formatter(NullFormatter())
    axes[0].set_yticks([0.02, 0.05, 0.1, 0.2, 0.3])
    axes[0].yaxis.set_major_formatter(ScalarFormatter())
    axes[0].yaxis.set_minor_formatter(NullFormatter())
    panel(axes[1], [r[0] for r in tilt], [r[3] for r in tilt], [r[1] for r in tilt],
          "max board tilt (deg), 10 views", "less tilt: f error explodes, fit error flat")
    fig.suptitle("A small reprojection error does not mean a correct camera matrix", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fit_vs_truth.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
