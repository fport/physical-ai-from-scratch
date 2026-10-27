"""Plot the timestep sweep of lesson 02 from output.txt (writes first_overlap_vs_dt.png).

Run:  uv run python experiments/02-timestep/plot.py
"""
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROW = re.compile(r"^\s+([\d.]+)\s+\d+\s+[\d.]+\s+\|\s+([\d.]+|-)\s+\(.{5}\)\s+(yes|no)\s+\S+\s+(WENT THROUGH|stayed)\s+\|",
                 re.M)


def main():
    text = open(os.path.join(HERE, "output.txt"), encoding="utf-8").read()
    mid = float(re.search(r"mid-plane ([\d.]+) cm", text).group(1))
    rows = [(float(dt), None if c == "-" else float(c), seen == "yes", res == "stayed")
            for dt, c, seen, res in ROW.findall(text)]

    fig, ax = plt.subplots(figsize=(9, 3.4), dpi=150)
    ax.axhspan(26, mid, color="tab:red", alpha=0.08)
    ax.axhline(mid, color="tab:red", lw=1)
    ax.text(0.45, mid - 0.25, "below the slab's mid-plane: the contact pushes the cube down", va="top",
            fontsize=8, color="tab:red")
    for dt, c, seen, ok in rows:
        if not seen:
            ax.plot(dt, 33.3, "v", color="gray", ms=6)
        else:
            ax.plot(dt, c, "o" if ok else "X", color="tab:green" if ok else "tab:red", ms=7)
    ax.annotate("touched on the right side,\nthen sank past the mid-plane", xy=(1.0, 32.2), xytext=(1.25, 33.4),
                fontsize=7, color="tab:red", arrowprops=dict(arrowstyle="-", color="tab:red", lw=0.6))
    ax.plot([], [], "o", color="tab:green", label="stayed on the table")
    ax.plot([], [], "X", color="tab:red", label="went through")
    ax.plot([], [], "v", color="gray", label="no step overlapped: contact never seen")
    ax.set_xscale("log")
    ax.set_xticks([r[0] for r in rows], [f"{r[0]:g}" for r in rows], fontsize=7)
    ax.minorticks_off()
    ax.set_ylim(26.5, 34)
    ax.set_xlabel("timestep dt (ms)")
    ax.set_ylabel("cube centre at first\noverlapping step (cm)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8, loc="lower left", ncol=3)
    ax.set_title("Default contact: the outcome is not monotonic in dt", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "first_overlap_vs_dt.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
