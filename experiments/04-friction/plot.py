"""Plot downhill speed vs ramp angle near the slip threshold (writes speed_vs_angle.png).

output.txt only holds the summary of each sweep, so this script re-runs the same `sweep()`
from experiment.py (μ = 0.8, 1 kg; deterministic, ~1 min) for three cases: aligned cube with the
pyramidal cone, and the 45°-rotated cube with the pyramidal and the elliptic cone.
Run:  uv run python experiments/04-friction/plot.py
"""
import importlib.util
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("friction_experiment", os.path.join(HERE, "experiment.py"))
exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp)

MU = 0.8


def main():
    fig, ax = plt.subplots(figsize=(9, 3.4), dpi=150)
    cases = [("aligned cube, pyramidal cone", "pyramidal", 0.0, "o-"),
             ("cube rotated 45°, pyramidal cone", "pyramidal", 45.0, "s-"),
             ("cube rotated 45°, elliptic cone", "elliptic", 45.0, "^-")]
    for label, cone, yaw, mark in cases:
        ref, angles, v = exp.sweep(MU, 1.0, cone=cone, yaw_deg=yaw)
        v2 = np.maximum(v[:, 1], 1e-3)                  # log axis: clip zero/negative speeds
        ax.plot(angles, v2, mark, ms=2.5, lw=1, label=label)
    ax.axvline(ref, color="k", lw=1, ls="--")
    ax.text(ref + 0.05, 0.15, f"Coulomb threshold:\narctan {MU} = {ref:.2f}°", fontsize=8)
    ax.axhline(exp.V_TH, color="gray", lw=0.8, ls=":")
    ax.text(ref - 1.4, exp.V_TH * 1.12, f"{exp.V_TH:g} cm/s (speed part of the slip criterion)", fontsize=7,
            color="gray")
    ax.set_yscale("log")
    ax.set_ylim(0.1, 600)
    ax.set_xlabel("ramp angle (°)")
    ax.set_ylabel("steady downhill speed\nover 1.5–2.0 s (cm/s)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=3)
    ax.set_title(f"μ = {MU}: soft friction creeps below the threshold; the pyramidal cone depends on direction",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "speed_vs_angle.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
