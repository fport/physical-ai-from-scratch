"""Plot sections A and B of output.txt (success vs action design and vs policy rate) without re-running.

Run:  uv run python experiments/34-observation-action-design/plot.py   -> writes success_vs_action_and_rate.png
"""
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROW = re.compile(r"^\s+(.+?)\s*(\d+/\d+/\d+)\s+(\d+)/300 \[\s*([\d.]+),\s*([\d.]+)\]\s+([\d.]+)\s+(\d+)\s+([\d.]+)\s+\d+")


def parse(text, section, nxt):
    """Rows of dicts from one section of output.txt."""
    block = text.split(f"[{section}]")[1].split(f"[{nxt}]")[0]
    rows = []
    for line in block.splitlines():
        m = ROW.match(line)
        if m:
            rows.append({"label": m[1].strip(), "pooled": int(m[3]) / 3, "lo": float(m[4]), "hi": float(m[5]),
                         "tau_rate": int(m[7])})
    return rows


def main():
    text = open(os.path.join(HERE, "output.txt"), encoding="utf-8").read()
    actions, rates = parse(text, "A", "B"), parse(text, "B", "C")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), dpi=150, gridspec_kw={"width_ratios": [1, 1.4]})

    ax = axes[0]
    names = ["absolute\ntarget (PD)", "delta\ntarget (PD)", "torque"]
    y = [r["pooled"] for r in actions]
    err = [[r["pooled"] - r["lo"] for r in actions], [r["hi"] - r["pooled"] for r in actions]]
    ax.bar(range(3), y, yerr=err, capsize=4, color=["C0", "C0", "C1"], alpha=0.8)
    for i, r in enumerate(actions):
        ax.text(i, 8, f"torque rate\n{r['tau_rate']} N·m/s", ha="center", fontsize=7, color="white")
    ax.set_xticks(range(3), names, fontsize=8)
    ax.set_ylim(0, 105)
    ax.set_ylabel("success (%), Wilson 95%")
    ax.set_title("what the action means (50 Hz)", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    hz = [float(r["label"].split("/")[1]) for r in rates]
    y = [r["pooled"] for r in rates]
    err = [[r["pooled"] - r["lo"] for r in rates], [r["hi"] - r["pooled"] for r in rates]]
    ax.errorbar(hz, y, yerr=err, fmt="o-", ms=5, capsize=4)
    ax.set_xscale("log", base=2)
    ax.set_xticks(hz, [f"{h:g}" for h in hz])
    ax.set_ylim(0, 105)
    ax.set_xlabel("policy rate (Hz), same 300k policy steps")
    ax.set_title("how often the policy decides (absolute targets)", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle("SO-101 reach, PPO fixed, 3 seeds x 100 episodes per point", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "success_vs_action_and_rate.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
