"""Lesson 14 figure: designed vs measured contact force over the stiffness sweep, parsed from output.txt.

Writes force_vs_stiffness.png next to this script. Run:  python plot.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_part_b(text):
    """Rows of block [B]: K, K x 5cm, hold force, peak force, penetration, saturated %, lost, tilt."""
    rows = []
    for line in text.split("[B]")[1].splitlines():
        parts = line.split()
        if len(parts) == 8 and parts[0].isdigit():
            rows.append([float(x) for x in parts])
    return rows


def main():
    rows = parse_part_b(open(os.path.join(HERE, "output.txt"), encoding="utf-8").read())
    k = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 3.4), dpi=150)
    ax.plot(k, [r[1] for r in rows], "k--", lw=1, label="designed: K x 5 cm")
    ax.plot(k, [r[2] for r in rows], "o-", ms=4, label="measured holding force")
    ax.plot(k, [r[3] for r in rows], "s-", ms=4, label="measured peak force (impact)")
    first_sat = min(r[0] for r in rows if r[5] > 0)
    ax.axvspan(first_sat / 1.7, max(k) * 1.3, color="tab:red", alpha=0.08)
    ax.text(first_sat / 1.5, 12, "joint torques\nsaturated", fontsize=8, color="tab:red")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(min(k) / 1.3, max(k) * 1.3)
    ax.set_xlabel("task-space stiffness K (N/m)")
    ax.set_ylabel("force on the surface (N)")
    ax.set_title("Pressing 5 cm into a rigid surface: the force follows K until the torque limits", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "force_vs_stiffness.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
