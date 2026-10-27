"""Plot lesson 09's finite-difference step sweep from output.txt (no re-run needed).

Writes fd_step_sweep.png next to this script:
  left:  error of the translational Jacobian columns (Jp) vs step eps, forward and central differences
  right: same for the rotational columns (Jr)

Run:  uv run python experiments/09-jacobians/plot.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def section(text, title):
    """Return the lines of the output section whose header contains `title`."""
    block = text.split(title, 1)[1].split("\n== ", 1)[0]
    return block.splitlines()[1:]


def main():
    text = open(os.path.join(HERE, "output.txt"), encoding="utf-8").read()
    rows = []
    for line in section(text, "B. Sweep finite-difference step"):
        parts = line.split()
        if len(parts) == 5:
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                continue
    assert rows, "could not parse output.txt"
    eps = [r[0] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), dpi=150, sharey=True)
    for ax, (fwd, cen), name in [(axes[0], (1, 3), "translational columns Jp"),
                                 (axes[1], (2, 4), "rotational columns Jr (round-off only)")]:
        ax.plot(eps, [r[fwd] for r in rows], "o-", ms=4, label="forward difference")
        ax.plot(eps, [r[cen] for r in rows], "s--", ms=4, label="central difference")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("finite-difference step eps (rad)")
        ax.set_title(name, fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("median max error vs mj_jacSite")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Smaller steps are not better: round-off on the left, truncation on the right", fontsize=10)
    fig.tight_layout()
    out = os.path.join(HERE, "fd_step_sweep.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
