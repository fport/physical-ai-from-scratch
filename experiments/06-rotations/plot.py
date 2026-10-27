"""Plot lesson 06's sweeps from output.txt (no re-run needed).

Writes euler_breakdown.png next to this script:
  left:  cond(E) and the Euler rate needed for a 1 rad/s body rotation vs distance from pitch 90 deg
  right: naive Euler interpolation deviation from slerp vs start pitch

Run:  uv run python experiments/06-rotations/plot.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def section(text, title):
    """Return the lines of the output section whose header contains `title`."""
    block = text.split(title, 1)[1].split("\n== ", 1)[0]
    return block.splitlines()[1:]


def numeric_rows(lines):
    rows = []
    for line in lines:
        parts = line.split()
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            continue
    return [r for r in rows if r]


def main():
    text = open(os.path.join(HERE, "output.txt"), encoding="utf-8").read()

    cond_rows = [r for r in numeric_rows(section(text, "B. Sweep pitch")) if r[0] < 90.0]
    rate_rows = [r for r in numeric_rows(section(text, "Break it 1.")) if r[0] < 90.0]
    interp_rows = numeric_rows(section(text, "C. Sweep start pitch"))
    assert cond_rows and rate_rows and interp_rows, "could not parse output.txt"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4), dpi=150)

    ax1.plot([90 - r[0] for r in cond_rows], [r[3] for r in cond_rows], "o-", ms=4, label="cond(E)")
    ax1.plot([90 - r[0] for r in rate_rows], [r[1] for r in rate_rows], "s--", ms=4,
             label="Euler rate for 1 rad/s (rad/s)")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.invert_xaxis()
    ax1.set_xlabel("distance from pitch 90 deg (deg)")
    ax1.set_ylabel("value (log scale)")
    ax1.set_title("Euler-rate map near gimbal lock", fontsize=9)
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(frameon=False, fontsize=8)

    pitch = [r[0] for r in interp_rows]
    ax2.plot(pitch, [r[1] for r in interp_rows], "o-", ms=4, label="median deviation")
    ax2.plot(pitch, [r[2] for r in interp_rows], "s--", ms=4, label="95th percentile")
    ax2.plot(pitch, [r[3] for r in interp_rows], "^:", ms=4, label="max deviation")
    ax2.set_xlabel("start pitch (deg)")
    ax2.set_ylabel("deviation from slerp (deg)")
    ax2.set_title("Naive Euler interpolation of a 60 deg turn", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.legend(frameon=False, fontsize=8)

    fig.suptitle("Euler angles break down in a whole neighbourhood of pitch 90 deg", fontsize=10)
    fig.tight_layout()
    out = os.path.join(HERE, "euler_breakdown.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
