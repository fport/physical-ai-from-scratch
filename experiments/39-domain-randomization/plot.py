"""Plot lesson 39's randomization-width sweep from output.txt (no re-run needed).

Left: success on the 32 held-out "real" plants vs randomization width w.
Right: RMS tracking error on the nominal plant and mean error on the real plants vs w.
The main sweep and the "wider than needed" rows are drawn together; the delay-only
exercise row is left out. Writes width_tradeoff.png next to this script.

Run:  uv run python experiments/39-domain-randomization/plot.py
"""
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def parse(text):
    rows, real_delay = [], None
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("--- exercise 3"):
            break
        m = re.search(r"delay ([\d.]+)-([\d.]+) ms", s)
        if s.startswith("real plants:") and m:
            real_delay = (float(m.group(1)), float(m.group(2)))
        cells = [c.strip() for c in s.split("|")]
        if len(cells) == 10 and re.fullmatch(r"[\d.]+", cells[0]):
            lo, hi = (float(x) for x in cells[1].split("-"))
            k, n = (int(x) for x in cells[9].split("/"))
            rows.append(dict(w=float(cells[0]), delay_hi=hi, nominal=float(cells[6]),
                             real_mean=float(cells[7]), success=k, n=n))
    return rows, real_delay


def main():
    with open(os.path.join(HERE, "output.txt"), encoding="utf-8") as f:
        rows, real_delay = parse(f.read())
    w = [r["w"] for r in rows]
    n = rows[0]["n"]
    covers = [r["w"] for r in rows if r["delay_hi"] >= real_delay[1]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4), dpi=150)
    for ax in (ax1, ax2):
        ax.axvspan(min(covers), max(w) + 0.1, color="0.92", zorder=0)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("randomization width w")
        ax.set_xlim(-0.1, max(w) + 0.1)

    ax1.plot(w, [r["success"] for r in rows], "o-", color="C0", ms=4)
    ax1.set_ylabel(f"real plants solved (of {n})")
    ax1.set_ylim(0, n)
    ax1.set_title("Too narrow is brittle, too wide is timid", fontsize=9)
    ax1.text(min(covers) + 0.03, 2, "training delays cover\nall real delays", fontsize=7, color="0.35")

    ax2.plot(w, [r["nominal"] for r in rows], "s--", color="C1", ms=4, label="nominal plant")
    ax2.plot(w, [r["real_mean"] for r in rows], "o-", color="C0", ms=4, label="real plants (mean)")
    ax2.set_ylabel("RMS tracking error (deg)")
    ax2.set_title("Width costs accuracy on the plant you know", fontsize=9)
    ax2.legend(frameon=False, fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "width_tradeoff.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
