"""Plot pick success vs aim error, light intensity and light direction from output.txt (writes picks_vs_light.png).

Run:  uv run python experiments/21-object-perception/plot.py
"""
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullFormatter, ScalarFormatter  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def section(text, start, end):
    part = text.split(start)[1]
    return part.split(end)[0] if end else part


def picks(line):
    """Successful picks (of 30) from the last 'n/30' cell of a line."""
    return int(re.findall(r"(\d+)/30", line)[-1])


def detected(line):
    return int(re.findall(r"(\d+)/30", line)[0])


def first_number_rows(block):
    """(first number, line) for every table row of a block."""
    rows = []
    for line in block.splitlines():
        m = re.match(r"\s*([\d.]+)\s", line)
        if m and "/30" in line:
            rows.append((float(m.group(1)), line))
    return rows


def main():
    text = open(os.path.join(HERE, "output.txt"), encoding="utf-8").read()
    aim = first_number_rows(section(text, "[1]", "[2]"))
    light = first_number_rows(section(text, "[2]", "[3]"))
    sun = first_number_rows(section(text, "[3]", "[4]"))
    sun_low = first_number_rows(section(text, "[4]", None))

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.4), dpi=150)

    axes[0].plot([a for a, _ in aim], [picks(s) for _, s in aim], "o-", ms=4, color="tab:purple")
    axes[0].set_xlabel("aim error (mm), ground-truth position")
    axes[0].set_title("how precise must perception be?", fontsize=9)

    x = [a for a, _ in light]
    axes[1].plot(x, [detected(s) for _, s in light], "s--", ms=4, color="tab:gray", label="cube detected")
    axes[1].plot(x, [picks(s) for _, s in light], "o-", ms=4, color="tab:red", label="pick succeeded")
    axes[1].set_xscale("log")
    axes[1].set_xticks([0.1, 0.2, 0.5, 1, 2, 3])
    axes[1].xaxis.set_major_formatter(ScalarFormatter())
    axes[1].xaxis.set_minor_formatter(NullFormatter())
    axes[1].set_xlabel("light intensity scale (1 = nominal)")
    axes[1].set_title("HSV threshold S>=120, V>=70", fontsize=9)
    axes[1].legend(frameon=False, fontsize=7)

    axes[2].plot([a for a, _ in sun], [picks(s) for _, s in sun], "o-", ms=4, color="tab:red", label="V_min = 70")
    axes[2].plot([a for a, _ in sun_low], [picks(s) for _, s in sun_low], "s--", ms=4, color="tab:green",
                 label="V_min = 25")
    axes[2].set_xlabel("sun elevation (deg), headlight off")
    axes[2].set_title("grazing light", fontsize=9)
    axes[2].legend(frameon=False, fontsize=7)

    for ax in axes:
        ax.set_ylabel("successful picks (of 30)")
        ax.set_ylim(-1, 31)
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle("Perceive, then pick: a threshold detector fails as a cliff, not a slope", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "picks_vs_light.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
