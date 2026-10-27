"""Plot the pixel-noise, camera-height and box-height results from output.txt (writes table_errors.png).

Run:  uv run python experiments/20-pixels-to-table/plot.py
"""
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def block_rows(text, start, end, ncols):
    """Numeric rows with ncols columns between the markers start and end (end=None: to the end)."""
    part = text.split(start)[1]
    if end:
        part = part.split(end)[0]
    rows = []
    for line in part.splitlines():
        cells = line.split()
        if len(cells) == ncols and all(re.fullmatch(r"[\d.]+", c) for c in cells):
            rows.append([float(c) for c in cells])
    return rows


def main():
    text = open(os.path.join(HERE, "output.txt"), encoding="utf-8").read()
    noise = block_rows(text, "[2]", "[3]", 4)
    height = block_rows(text, "[3]", "[4]", 4)
    box = block_rows(text, "[4]", None, 3)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.4), dpi=150)

    x = [r[0] for r in noise]
    axes[0].plot(x, [r[1] for r in noise], "o-", ms=4, color="tab:red", label="homography, 4 markers")
    axes[0].plot(x, [r[2] for r in noise], "s-", ms=4, color="tab:orange", label="homography, 12 markers")
    axes[0].plot(x, [r[3] for r in noise], "^-", ms=4, color="tab:blue", label="depth back-projection")
    axes[0].set_xlabel("pixel noise (px), camera at 0.6 m")
    axes[0].set_ylabel("median table-point error (mm)")
    axes[0].legend(frameon=False, fontsize=7)

    x = [r[0] for r in height]
    axes[1].plot(x, [r[2] for r in height], "s-", ms=4, color="tab:orange", label="homography, 12 markers")
    axes[1].plot(x, [r[3] for r in height], "^-", ms=4, color="tab:blue", label="depth back-projection")
    axes[1].plot(x, [r[1] for r in height], "k:", label="size of one pixel on the table")
    axes[1].set_xlabel("camera height (m), 1 px noise")
    axes[1].set_ylabel("median error (mm)")
    axes[1].legend(frameon=False, fontsize=7)

    x = [r[0] for r in box]
    axes[2].plot(x, [r[1] for r in box], "o-", ms=4, color="tab:red", label="homography (assumes z = 0)")
    axes[2].plot(x, [r[2] for r in box], "^-", ms=4, color="tab:blue", label="depth back-projection")
    axes[2].set_xlabel("object height (mm), no pixel noise")
    axes[2].set_ylabel("error at the top face (mm)")
    axes[2].legend(frameon=False, fontsize=7)

    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle("Pixel to table: errors grow with pixel noise, camera height and object height", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "table_errors.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
