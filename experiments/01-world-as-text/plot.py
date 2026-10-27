"""Plot the air-drag delay sweep of lesson 01 from output.txt (writes drag_delay_vs_mass.png).

Run:  uv run python experiments/01-world-as-text/plot.py
"""
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    text = open(os.path.join(HERE, "output.txt"), encoding="utf-8").read()
    rows = re.findall(r"light\s+(\d+) g · vacuum\s+([\d.]+) ms · air\s+([\d.]+) ms", text)
    mass = [float(r[0]) for r in rows]
    vac = [float(r[1]) for r in rows]
    air = [float(r[2]) for r in rows]
    keep = [i for i, a in enumerate(air) if a > 0]          # 1 kg vs 1 kg has zero delay: off a log axis

    fig, ax = plt.subplots(figsize=(9, 3.4), dpi=150)
    ax.plot([mass[i] for i in keep], [air[i] for i in keep], "o-", label='with air (density="1.2")')
    m0, d0 = mass[keep[0]], air[keep[0]]
    ax.plot([mass[i] for i in keep], [d0 * m0 / mass[i] for i in keep], ":", color="gray",
            label="guide: proportional to 1/mass")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([mass[i] for i in keep], [f"{mass[i]:g}" for i in keep])
    ax.minorticks_off()
    ax.set_xlabel("mass of the light cube (g); the heavy cube is always 1 kg")
    ax.set_ylabel("landing delay vs heavy cube (ms)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False)
    ax.set_title(f"In vacuum the delay is {max(vac):.2f} ms for every mass; with air it grows as the cube gets lighter",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "drag_delay_vs_mass.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
