"""Plot lesson 37's delay results from output.txt (no re-run needed).

Left: overshoot vs measurement delay for the PD loop of part A, with the predicted delay margin.
Right: predicted delay margin vs Kp (break-it sweep) against the fixed 30 ms delay.
Writes delay_margin.png next to this script.

Run:  uv run python experiments/37-latency/plot.py
"""
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def parse(text):
    part_a, brk = [], []
    section = None
    d_max = None
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("[A]"):
            section = "A"
        elif s.startswith("[B]"):
            section = None
        elif s.startswith("[Break it]"):
            section = "break"
        m = re.search(r"delay margin PM/w_c = ([\d.]+) ms", s)
        if m:
            d_max = float(m.group(1))
        cells = [c.strip() for c in s.split("|")]
        if section == "A" and len(cells) == 4 and re.fullmatch(r"\d+", cells[0]):
            over = None if cells[1] == "diverges" else float(cells[1])
            part_a.append((int(cells[0]), over, cells[3] == "yes"))
        if section == "break" and len(cells) == 6 and re.fullmatch(r"\d+", cells[0]):
            brk.append((int(cells[0]), float(cells[2]), float(cells[3]), cells[5] == "yes"))
    return part_a, brk, d_max


def main():
    with open(os.path.join(HERE, "output.txt"), encoding="utf-8") as f:
        part_a, brk, d_max = parse(f.read())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4), dpi=150)

    finite = [(d, o) for d, o, _ in part_a if o is not None]
    ax1.plot([d for d, _ in finite], [o for _, o in finite], "o-", ms=4, color="C0", label="measured overshoot")
    div = [d for d, o, _ in part_a if o is None]
    top = max(o for _, o in finite) * 2
    ax1.plot(div, [top] * len(div), "x", color="C3", ms=6, label="diverges")
    ax1.axvline(d_max, color="k", ls="--", lw=1, label=f"predicted limit {d_max} ms")
    ax1.set_yscale("log")
    ax1.set_xlabel("measurement delay d (ms)")
    ax1.set_ylabel("overshoot (%)")
    ax1.set_title("Kp = 20: harmless until close to the limit", fontsize=9)
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(frameon=False, fontsize=7, loc="upper left")

    kps = [r[0] for r in brk]
    margins = [r[2] for r in brk]
    ax2.plot(kps, margins, "-", color="0.6", lw=1)
    for kp, _, dm, ok in brk:
        ax2.plot(kp, dm, "o" if ok else "x", color="C0" if ok else "C3", ms=6)
    ax2.axhline(30, color="k", ls="--", lw=1)
    ax2.text(kps[0], 31.5, "actual delay 30 ms", fontsize=7)
    ax2.plot([], [], "o", color="C0", label="stable at 30 ms")
    ax2.plot([], [], "x", color="C3", label="diverges at 30 ms")
    ax2.set_xscale("log")
    ax2.set_xlabel("Kp (zeta fixed at 0.7)")
    ax2.set_ylabel("predicted delay margin (ms)")
    ax2.set_title("Stiffer loop, smaller delay budget", fontsize=9)
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(frameon=False, fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "delay_margin.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
