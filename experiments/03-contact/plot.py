"""Plot two sweeps of lesson 03 from output.txt (writes contact_sweeps.png).

Left: steady penetration at rest vs solref timeconst. Right: rebound of a 10 cm drop vs dampratio.
Run:  uv run python experiments/03-contact/plot.py
"""
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def block(text, title):
    return text.split(title, 1)[1].split("\n\n", 1)[0]


def main():
    text = open(os.path.join(HERE, "output.txt"), encoding="utf-8").read()
    h = float(re.search(r"timestep ([\d.]+) s", text).group(1))
    s1 = re.findall(r"^\s+([\d.]+)\s+([\d.]+)\s+\d+\s+([\d.]+)\s+[\d.]+\s+[\d.]+\s+([\d.]+)$",
                    block(text, "sweep 1:"), re.M)
    tc, cube, sphere, eq = (list(map(float, c)) for c in zip(*s1))
    s3 = re.findall(r"^\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+[\d.]+$", block(text, "sweep 3:"), re.M)
    dr, peak, rebound = (list(map(float, c)) for c in zip(*s3))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4), dpi=150)
    a1.axvspan(min(tc) * 0.8, 2 * h, color="gray", alpha=0.15)
    a1.text(min(tc) * 0.9, max(sphere) * 0.5, f"below 2·dt = {2 * h:g} s:\nrefsafe raises\ntimeconst to {2 * h:g}",
            fontsize=7, va="top")
    a1.plot(tc, eq, "-", color="gray", lw=3, alpha=0.5, label="steady-state equation")
    a1.plot(tc, sphere, "o", label="sphere (1 contact)", ms=4)
    a1.plot(tc, cube, "s", label="cube (4 contacts)", ms=4)
    a1.set_xscale("log")
    a1.set_yscale("log")
    a1.set_xlim(min(tc) * 0.8, max(tc) * 1.25)
    a1.set_xlabel("solref timeconst (s)")
    a1.set_ylabel("penetration at rest (mm)")
    a1.grid(True, which="both", alpha=0.3)
    a1.legend(frameon=False, fontsize=7, loc="lower right")
    a1.set_title("Softer contact sinks deeper (dampratio 1)", fontsize=9)

    a2.plot(dr, rebound, "o-", ms=4)
    a2.set_xlabel("solref dampratio")
    a2.set_ylabel("rebound after a 10 cm drop (mm)")
    a2.grid(True, alpha=0.3)
    tc3 = re.search(r"sweep 3: dampratio \(timeconst ([\d.]+)\)", text).group(1)
    a2.set_title(f"dampratio sets the bounce (timeconst {tc3} s)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "contact_sweeps.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
