"""Plot the process-noise sweep from output.txt (writes q_sweep.png next to this script).

Run:  uv run python experiments/18-kalman-filter/plot.py
"""
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullFormatter, ScalarFormatter  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def parse():
    """Return the q sweep rows (q, pos mrad, delay ms, vel rad/s, K_pos, K_vel, cover %) and the two baselines."""
    text = open(os.path.join(HERE, "output.txt"), encoding="utf-8").read()
    ma_pos = float(re.search(r"MA N=5\s*: pos RMS ([\d.]+) mrad", text).group(1))
    ma_vel = float(re.search(r"MA N=65 : vel RMS ([\d.]+) rad/s", text).group(1))
    block = text.split("[2]")[1].split("[3]")[0]
    rows = []
    for line in block.splitlines():
        cells = line.split()
        if len(cells) == 7 and re.fullmatch(r"[\d.]+", cells[0]):
            rows.append([float(c) for c in cells])
    return rows, ma_pos, ma_vel


def main():
    rows, ma_pos, ma_vel = parse()
    q = [r[0] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.4), dpi=150)

    axes[0].plot(q, [r[1] for r in rows], "o-", ms=4, color="tab:blue", label="Kalman filter")
    axes[0].axhline(ma_pos, ls="--", color="tab:gray", label="best moving average (lesson 17)")
    axes[0].set_ylabel("position RMS error (mrad)")
    axes[0].set_yscale("log")
    axes[0].set_ylim(2.5, 45)
    axes[0].set_yticks([3, 5, 10, 20])
    axes[0].legend(frameon=False, fontsize=7, loc="upper right")

    axes[1].plot(q, [r[3] for r in rows], "o-", ms=4, color="tab:orange", label="Kalman filter")
    axes[1].axhline(ma_vel, ls="--", color="tab:gray", label="best moving average (lesson 17)")
    axes[1].set_ylabel("velocity RMS error (rad/s)")
    axes[1].set_yscale("log")
    axes[1].set_ylim(0.07, 2.5)
    axes[1].set_yticks([0.1, 0.2, 0.5, 1])
    axes[1].legend(frameon=False, fontsize=7, loc="upper left")

    axes[2].plot(q, [r[6] for r in rows], "o-", ms=4, color="tab:green")
    axes[2].axhline(95, ls="--", color="tab:gray")
    axes[2].text(q[0], 100, "honest filter: ~95 %", fontsize=7, color="tab:gray")
    axes[2].set_ylabel("truth inside own 2-sigma band (%)")
    axes[2].set_ylim(0, 110)

    for ax in axes[:2]:
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("process noise q_c (rad²/s³)")
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle("Tuning q_c: too small lags and is overconfident, too large follows the noise", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "q_sweep.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
