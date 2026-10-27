"""Plot lesson 10's sweeps from output.txt (no re-run needed).

Writes damping_tradeoff.png next to this script:
  left:  success rate and median iterations vs solver (transpose, pseudo-inverse, DLS lambda)
  right: joint speed requested for 5 cm/s along the weakest direction while the arm stretches

Run:  uv run python experiments/10-inverse-kinematics/plot.py
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

    labels, success, iters = [], [], []
    for line in section(text, "A. Sweep damping"):
        if "%" not in line or "restarts" in line:
            continue
        head, rest = line.split("%", 1)
        name, rate = head.rsplit(None, 1)
        name = name.strip()
        labels.append({"transpose": "trans-\npose", "pinv": "pinv"}.get(name, name.replace("dls l=", "DLS\n")))
        success.append(float(rate))
        iters.append(float(rest.split()[0]))

    speed = []
    for line in section(text, "Break it. Request v"):
        parts = line.split()
        try:
            speed.append([float(p) for p in parts])
        except ValueError:
            continue
    assert labels and speed, "could not parse output.txt"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4), dpi=150)

    x = range(len(labels))
    ax1.bar(x, success, color="tab:blue", alpha=0.8)
    ax1.set_xticks(list(x), labels, fontsize=7)
    ax1.set_ylim(50, 100)
    ax1.set_ylabel("success rate (%)", color="tab:blue")
    ax1.set_title("500 reachable targets from q0 = 0", fontsize=9)
    ax1b = ax1.twinx()
    ax1b.plot(list(x), iters, "o-", color="tab:orange", ms=4)
    ax1b.set_ylabel("median iterations", color="tab:orange")
    ax1.grid(True, axis="y", alpha=0.3)

    cols = [str(r[0]) for r in speed]
    xs = range(len(cols))
    ax2.plot(list(xs), [r[1] for r in speed], "o-", ms=4, label="pseudo-inverse")
    ax2.plot(list(xs), [r[2] for r in speed], "s--", ms=4, label="DLS lambda = 0.03")
    ax2.plot(list(xs), [r[4] for r in speed], "^:", ms=4, label="DLS lambda = 0.1")
    ax2.set_xticks(list(xs), cols)
    ax2.set_yscale("log")
    ax2.set_xlabel("stretch s (1 = straight arm)")
    ax2.set_ylabel("joint speed |qdot| (rad/s)")
    ax2.set_title("Request 5 cm/s along the weakest direction", fontsize=9)
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(frameon=False, fontsize=8)

    fig.suptitle("Damping: more targets solved, bounded joint speeds near the straight arm, slower convergence", fontsize=10)
    fig.tight_layout()
    out = os.path.join(HERE, "damping_tradeoff.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
