"""Lesson 11 figure: energy error of each integrator vs timestep, parsed from output.txt.

Writes energy_vs_timestep.png next to this script. Run:  python plot.py
"""
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_part_b(text):
    """Return dt list (ms) and {integrator: [final signed energy error %]} from block [B]."""
    block = text.split("[B]")[1].split("[C]")[0]
    header = next(line for line in block.splitlines() if line.startswith("integrator"))
    dts = [float(x) for x in re.findall(r"dt ([\d.]+) ms", header)]
    rows = {}
    for line in block.splitlines():
        name = line.split()[0] if line.strip() else ""
        if name in ("Euler", "RK4", "implicit", "implicitfast"):
            cells = re.findall(r"([\d.]+) / ([+-][\d.]+)", line)
            rows[name] = [float(final) for _, final in cells]
    return dts, rows


def main():
    text = open(os.path.join(HERE, "output.txt"), encoding="utf-8").read()
    dts, rows = parse_part_b(text)
    fig, ax = plt.subplots(figsize=(9, 3.4), dpi=150)
    styles = {"Euler": ("o-", "Euler (= implicitfast here)"), "implicit": ("s-", "implicit"),
              "RK4": ("^-", "RK4")}
    for name, (mark, label) in styles.items():
        ax.plot(dts, rows[name], mark, label=label, ms=4)
    ax.axhline(0.0, color="k", lw=0.6)
    ax.axvline(2.0, color="gray", lw=0.8, ls="--")
    ax.text(2.1, ax.get_ylim()[1] * 0.85, "default 2 ms", fontsize=8, color="gray")
    ax.set_xscale("log")
    ax.set_xticks(dts, [f"{dt:g}" for dt in dts])
    ax.minorticks_off()
    ax.set_xlabel("timestep (ms)")
    ax.set_ylabel("energy change after 10 s\n(% of releasable energy)")
    ax.set_title("Frictionless double pendulum: Euler gains energy, implicit loses it, RK4 keeps it",
                 fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "energy_vs_timestep.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
