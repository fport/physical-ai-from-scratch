"""Plot lesson 38's identification results from output.txt (no re-run needed).

Left: validation error before and after the fit vs excitation amplitude, with the share of
saturated steps. Right: validation error after the fit vs measurement noise, and the fit without
a delay parameter. Writes fit_error.png next to this script.

Run:  uv run python experiments/38-system-identification/plot.py
"""
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROW = re.compile(r"^(amp|sigma|no delay term)\s*([\d.]*)\s*(rad|mrad)?\s*\|(.*)$")


def parse(text):
    amp, noise, no_delay, nominal = [], [], None, None
    for line in text.splitlines():
        m = ROW.match(line.strip())
        if not m:
            continue
        cells = [c.strip() for c in m.group(4).split("|")]
        val_nom, val_fit = float(cells[6]), float(cells[7])
        nominal = val_nom
        if m.group(1) == "amp":
            sat = float(re.search(r"sat\s+([\d.]+)%", line).group(1))
            amp.append((float(m.group(2)), val_fit, sat))
        elif m.group(1) == "sigma":
            noise.append((float(m.group(2)), val_fit))
        else:
            no_delay = val_fit
    return amp, noise, no_delay, nominal


def main():
    with open(os.path.join(HERE, "output.txt"), encoding="utf-8") as f:
        amp, noise, no_delay, nominal = parse(f.read())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4), dpi=150)

    a = [r[0] for r in amp]
    ax1.plot(a, [nominal] * len(a), "--", color="0.5", lw=1, label="nominal model (before fit)")
    ax1.plot(a, [r[1] for r in amp], "o-", color="C0", ms=4, label="fitted model")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("excitation amplitude (rad)")
    ax1.set_ylabel("validation RMS error (mrad)")
    ax1.set_title("Excitation has a window (noise 2 mrad)", fontsize=9)
    ax1.grid(True, which="both", alpha=0.3)
    twin = ax1.twinx()
    twin.plot(a, [r[2] for r in amp], "s:", color="C3", ms=3, label="steps with a servo saturated")
    twin.set_ylabel("saturated steps (%)", color="C3")
    twin.set_ylim(0, 105)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = twin.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7, loc="lower left")

    s = [r[0] for r in noise]
    ax2.plot(s, [nominal] * len(s), "--", color="0.5", lw=1, label="nominal model (before fit)")
    ax2.plot(s, [r[1] for r in noise], "o-", color="C0", ms=4, label="fitted model, with delay term")
    ax2.plot([0], [no_delay], "x", color="C3", ms=7, label="fitted model, no delay term (noise 0)")
    ax2.set_xlabel("measurement noise sigma (mrad)")
    ax2.set_ylabel("validation RMS error (mrad)")
    ax2.set_title("Noise hurts slowly; a missing mechanism hurts at once", fontsize=9)
    ax2.set_ylim(0, nominal * 1.1)
    ax2.grid(True, alpha=0.3)
    ax2.legend(frameon=False, fontsize=7, loc="center right")

    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fit_error.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
