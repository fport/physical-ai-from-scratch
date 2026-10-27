"""Plot lesson 08's sweeps from output.txt (no re-run needed).

Writes workspace_and_orientation.png next to this script:
  left:  sampled workspace volume vs number of joint samples (1 cm voxels, 2 cm voxels, convex hull)
  right: best approach-direction error vs required angle out of the arm plane (beta)

Run:  uv run python experiments/08-forward-kinematics/plot.py
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


def numeric_rows(lines):
    rows = []
    for line in lines:
        try:
            rows.append([float(p.rstrip("%")) for p in line.split()])
        except ValueError:
            continue
    return [r for r in rows if r]


def main():
    text = open(os.path.join(HERE, "output.txt"), encoding="utf-8").read()
    ws = [r for r in numeric_rows(section(text, "Workspace of the gripperframe")) if len(r) == 6]
    beta = [r for r in numeric_rows(section(text, "Sweep beta")) if len(r) == 3]
    assert ws and beta, "could not parse output.txt"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4), dpi=150)

    n = [r[0] for r in ws]
    ax1.plot(n, [r[1] for r in ws], "o-", ms=4, label="1 cm voxels")
    ax1.plot(n, [r[2] for r in ws], "s--", ms=4, label="2 cm voxels")
    ax1.plot(n, [r[3] for r in ws], "^:", ms=4, label="convex hull")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("number of random joint samples")
    ax1.set_ylabel("estimated volume (L)")
    ax1.set_title("SO-101 workspace estimate", fontsize=9)
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(frameon=False, fontsize=8)

    b = [r[0] for r in beta]
    ax2.plot(b, b, color="0.6", lw=1, label="error = beta")
    ax2.plot(b, [r[2] for r in beta], "o-", ms=4, label="best approach error")
    ax2.set_xlabel("approach angle out of the arm plane, beta (deg)")
    ax2.set_ylabel("orientation error (deg)")
    ax2.set_title("5 joints, fixed point: position always reached", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.legend(frameon=False, fontsize=8)

    fig.suptitle("Volume estimates depend on sample count and voxel size; out-of-plane orientations are lost",
                 fontsize=10)
    fig.tight_layout()
    out = os.path.join(HERE, "workspace_and_orientation.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
