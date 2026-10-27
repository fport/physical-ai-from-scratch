"""Plot open-loop replay versus re-planning success over cube shift, and the farthest successful replay
shift per direction, from output.txt part A (writes replay_vs_shift.png next to this script).

Run:  uv run python experiments/26-teleop-replay-data/plot.py
"""
import math
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def parse():
    """Part A rows (shift cm, replay k, n, Wilson lo, hi, feedback k) and {direction deg: farthest cm}."""
    rows, dirs, section = [], {}, None
    for line in open(os.path.join(HERE, "output.txt"), encoding="utf-8"):
        if line.startswith("["):
            section = line[1]
            continue
        if section != "A":
            continue
        m = re.match(r"\s*([\d.]+)\s+(\d+)/(\d+)\s+\[([\d.]+), ([\d.]+)\]\s+(\d+)/\d+", line)
        if m:
            rows.append((float(m[1]), int(m[2]), int(m[3]), float(m[4]), float(m[5]), int(m[6])))
        m = re.match(r"\s*(\d+) deg:\s+([\d.]+) cm", line)
        if m:
            dirs[int(m[1])] = float(m[2])
    return rows, dirs


def main():
    rows, dirs = parse()
    fig = plt.figure(figsize=(9, 3.4), dpi=150)

    ax = fig.add_subplot(1, 2, 1)
    s = [r[0] for r in rows]
    rate = [r[1] / r[2] for r in rows]
    err = [[p - r[3] for p, r in zip(rate, rows)], [r[4] - p for p, r in zip(rate, rows)]]
    ax.errorbar(s, rate, yerr=err, fmt="o-", ms=4, capsize=2, label="replay (Wilson 95%)")
    ax.plot(s, [r[5] / r[2] for r in rows], "s--", ms=4, label="re-plan from cube pose")
    ax.set_xlabel("cube shift from the recorded spot (cm)")
    ax.set_ylabel("success rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("replay holds only within ~1 cm", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=7, loc="upper right", bbox_to_anchor=(1.0, 0.92))

    ax = fig.add_subplot(1, 2, 2, projection="polar")
    ang = [math.radians(a) for a in sorted(dirs)]
    far = [dirs[a] for a in sorted(dirs)]
    ax.bar(ang, far, width=math.radians(30), alpha=0.7)
    for a, f in zip(ang, far):
        ax.text(a, f + 0.8, f"{f:g}", fontsize=7, ha="center", va="center")
    ax.set_xticks([math.radians(a) for a in sorted(dirs)],
                  labels=["0° +x", "45°", "90° +y\nmoving jaw", "135°", "180°", "225°", "270° -y\nfixed jaw", "315°"],
                  fontsize=7)
    ax.set_rlim(0, 5)
    ax.set_rticks([1, 2, 3, 4, 5], labels=["", "", "", "", "5 cm"], fontsize=6)
    ax.set_rlabel_position(20)
    ax.set_title("farthest successful replay shift per direction", fontsize=9, pad=12)

    fig.suptitle("SO-101 sim: open-loop replay vs closed-loop re-planning (8 directions per shift)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "replay_vs_shift.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
