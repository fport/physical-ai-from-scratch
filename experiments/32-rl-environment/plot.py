"""Plot section C of output.txt (throughput vs frame_skip) without re-running the experiment.

Run:  uv run python experiments/32-rl-environment/plot.py   -> writes throughput_vs_frame_skip.png
"""
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_section_c(text):
    """Rows of (frame_skip, env steps/CPU s, physics steps/CPU s, sim s/CPU s) from section [C]."""
    block = text.split("[C]")[1].split("[D]")[0]
    rows = []
    for line in block.splitlines():
        m = re.match(r"\s+(\d+)\s+([\d.]+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s*$", line)
        if m:
            rows.append((int(m[1]), int(m[3]), int(m[4]), float(m[5])))
    return rows


def main():
    rows = parse_section_c(open(os.path.join(HERE, "output.txt"), encoding="utf-8").read())
    fs = [r[0] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), dpi=150)

    ax = axes[0]
    ax.plot(fs, [r[1] for r in rows], "o-", ms=4, label="env steps (policy decisions)")
    ax.plot(fs, [r[2] for r in rows], "s--", ms=4, label="physics steps")
    ax.set_xscale("log", base=2)
    ax.set_xticks(fs, [str(f) for f in fs])
    ax.set_xlabel("frame_skip (physics steps per policy step)")
    ax.set_ylabel("steps per CPU second")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.plot(fs, [r[3] for r in rows], "o-", ms=4, color="C2")
    for f, r in zip(fs, rows):
        if f == 4:
            ax.annotate(f"{r[3]:.1f} at frame_skip 4 (50 Hz)", (f, r[3]), textcoords="offset points",
                        xytext=(8, -14), fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(fs, [str(f) for f in fs])
    ax.set_xlabel("frame_skip")
    ax.set_ylabel("simulated seconds per CPU second")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Fewer policy decisions per second: more physics per CPU second, fewer env steps", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "throughput_vs_frame_skip.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
