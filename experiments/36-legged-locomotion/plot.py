"""Plot section C of output.txt (Go2 steps/s vs number of parallel robots) without re-running.

Run:  uv run python experiments/36-legged-locomotion/plot.py   -> writes throughput_vs_batch.png
"""
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
THREADS = [1, 2, 4, 8]


def parse_table(block):
    """{nbatch: [value per thread count]} from one table of section [C]."""
    out = {}
    for line in block.splitlines():
        m = re.match(r"^\s+(\d+)\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)\s*$", line)
        if m:
            out[int(m[1])] = [int(v.replace(",", "")) for v in m.groups()[1:]]
    return out


def main():
    text = open(os.path.join(HERE, "output.txt"), encoding="utf-8").read()
    c = text.split("[C]")[1].split("[D]")[0]
    wall_block, cpu_block = c.split("steps per CPU-second")
    wall, cpu = parse_table(wall_block), parse_table(cpu_block)
    python_loop = int(re.search(r"Python loop, 1 MjData, 1 thread: ([\d,]+)", c)[1].replace(",", ""))
    batches = sorted(wall)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), dpi=150)
    for ax, table, ylabel in [(axes[0], wall, "physics steps per wall-clock second"),
                              (axes[1], cpu, "physics steps per CPU second")]:
        for j, nt in enumerate(THREADS):
            ax.plot(batches, [table[b][j] for b in batches], "o-", ms=4, label=f"{nt} thread{'s' * (nt > 1)}")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(batches, [str(b) for b in batches])
        ax.set_xlabel("robots per rollout call (nbatch)")
        ax.set_ylabel(ylabel)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}k"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_yticks([3e4, 5e4, 1e5, 2e5, 5e5] if table is wall else [3e4, 5e4, 7e4, 1e5, 1.5e5])
        ax.grid(True, which="both", alpha=0.3)
    axes[0].axhline(python_loop, color="gray", ls=":", lw=1)
    axes[0].text(1400, python_loop * 0.9, "Python mj_step loop", fontsize=7, color="gray", ha="center", va="top")
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    axes[1].set_ylim(3e4, 1.5e5)

    fig.suptitle("Threads help only with a batch to split; the cost of one Go2 step stays roughly fixed", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "throughput_vs_batch.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
