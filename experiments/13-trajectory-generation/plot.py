"""Lesson 13 figure: position, velocity and acceleration of the time scalings s(u).

The curves are the analytic profiles from the lesson's table (no simulation data needed).
Writes time_scalings.png next to this script. Run:  python plot.py
"""
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def trapezoid(u):
    """Accelerate for 1/3, cruise, decelerate for 1/3 (peak velocity 1.5, acceleration 4.5)."""
    a = 4.5
    s = np.where(u < 1 / 3, 0.5 * a * u ** 2,
                 np.where(u < 2 / 3, 0.25 + 1.5 * (u - 1 / 3), 1 - 0.5 * a * (1 - u) ** 2))
    v = np.where(u < 1 / 3, a * u, np.where(u < 2 / 3, 1.5, a * (1 - u)))
    acc = np.where(u < 1 / 3, a, np.where(u < 2 / 3, 0.0, -a))
    return s, v, acc


def main():
    u = np.linspace(0, 1, 2001)
    shapes = {
        "linear": (u, np.ones_like(u), np.zeros_like(u)),
        "trapezoid": trapezoid(u),
        "cubic": (3 * u ** 2 - 2 * u ** 3, 6 * u - 6 * u ** 2, 6 - 12 * u),
        "quintic": (10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5,
                    30 * u ** 2 - 60 * u ** 3 + 30 * u ** 4,
                    60 * u - 180 * u ** 2 + 120 * u ** 3),
    }
    # pad with rest before and after the move, so jumps at the ends are visible
    pad = np.linspace(-0.15, 0, 50)
    tail = np.linspace(1, 1.15, 50)
    t = np.concatenate([pad, u, tail])
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.4), dpi=150)
    titles = ["position s(u)", "velocity ds/du", "acceleration d²s/du²"]
    for name, curves in shapes.items():
        for i, (ax, c) in enumerate(zip(axes, curves)):
            before, after = (0.0, 1.0) if i == 0 else (0.0, 0.0)
            y = np.concatenate([np.full(50, before), c, np.full(50, after)])
            ax.plot(t, y, label=name, lw=1.3)
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("u = t / T")
        ax.grid(True, alpha=0.3)
    axes[2].set_ylim(-7.5, 7.5)
    axes[2].text(0.02, -7.0, "linear: infinite spikes at u = 0 and 1", fontsize=7)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Smoother shapes: the quintic starts and ends with zero velocity and zero acceleration",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "time_scalings.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
