"""Plot section B of output.txt (success vs gamma) without re-running the 15 training runs.

Run:  uv run python experiments/33-ppo/plot.py   -> writes success_vs_gamma.png
"""
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_section_b(text):
    """Rows of (gamma, horizon s, [per-seed successes], pooled %, wilson lo, wilson hi) from section [B]."""
    block = text.split("[B]")[1].split("[C]")[0]
    rows = []
    pat = r"\s+([\d.]+)\s+([\d.]+) s\s+([\d/]+)\s+\d+/\d+\s+([\d.]+)%\s+\[\s*([\d.]+),\s*([\d.]+)\]"
    for line in block.splitlines():
        m = re.match(pat, line)
        if m:
            seeds = [int(s) for s in m[3].split("/")]
            rows.append((float(m[1]), float(m[2]), seeds, float(m[4]), float(m[5]), float(m[6])))
    return rows


def main():
    rows = parse_section_b(open(os.path.join(HERE, "output.txt"), encoding="utf-8").read())
    x = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(9, 3.4), dpi=150)
    pooled = [r[3] for r in rows]
    err = [[r[3] - r[4] for r in rows], [r[5] - r[3] for r in rows]]
    ax.errorbar(x, pooled, yerr=err, fmt="o-", ms=5, capsize=4, label="pooled, 300 episodes (Wilson 95%)")
    for i, r in enumerate(rows):
        ax.plot([i + d for d in (-0.08, 0, 0.08)], r[2], "x", color="C1", ms=6,
                label="one seed, 100 episodes" if i == 0 else None)
    ax.set_xticks(x, [f"γ = {r[0]:g}\nhorizon {r[1]:g} s" for r in rows])
    ax.set_xlim(-0.4, len(rows) - 0.6)
    ax.set_ylim(0, 105)
    ax.set_ylabel("success (%)")
    ax.set_xlabel("discount γ at 50 Hz (horizon ≈ 20 ms / (1 − γ))")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title("A 1.6 s reach learns best with a horizon of a few hundred ms (300k steps per run)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "success_vs_gamma.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
