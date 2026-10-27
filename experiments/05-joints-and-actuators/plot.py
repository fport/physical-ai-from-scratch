"""Plot the kp sweep of lesson 05 from output.txt (writes kp_sweep.png).

Left: steady sag under gravity vs kp. Right: 10-90 % rise time vs kp.
Run:  uv run python experiments/05-joints-and-actuators/plot.py
"""
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    text = open(os.path.join(HERE, "output.txt"), encoding="utf-8").read()
    sweep = text.split("sweep 1:", 1)[1].split("\n\n", 1)[0]
    rows = re.findall(r"^\s+(\d+)\s+[\d.]+\s+([\d.]+)\s+([\d.]+)\s+[\d.]+\s+([\d.]+)$", sweep, re.M)
    kp, sag, lin, rise = (list(map(float, c)) for c in zip(*rows))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4), dpi=150)
    a1.plot(kp, lin, "-", color="gray", lw=3, alpha=0.5, label="linear estimate tau_g / kp")
    a1.plot(kp, sag, "o", ms=4, label="simulated sag")
    a1.set_ylabel("steady sag below target (deg)")
    a1.set_title("Sag falls as 1/kp", fontsize=9)
    a2.plot(kp, [rise[0] * (kp[0] / k) ** 0.5 for k in kp], "-", color="gray", lw=3, alpha=0.5,
            label="guide: proportional to 1/sqrt(kp)")
    a2.plot(kp, rise, "o", ms=4, label="simulated rise time")
    a2.set_ylabel("rise time 10-90 % (ms)")
    a2.set_title("Rise time falls as 1/sqrt(kp)", fontsize=9)
    for ax in (a1, a2):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("kp (N·m/rad), dampratio 1")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "kp_sweep.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
