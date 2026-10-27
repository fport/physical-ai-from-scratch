"""Shared test helpers."""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?", re.I)


def assert_build_output(slug):
    """Run experiments/<slug>/build.py and compare with its committed build_output.txt.

    Text must match exactly; numbers must match to 1e-6 relative, except round-off-sized values
    (both below 1e-9 in magnitude), which only need to stay that small: they differ across CPUs and libms.
    """
    got = subprocess.run([sys.executable, os.path.join(ROOT, "experiments", slug, "build.py")],
                         capture_output=True, text=True, check=True).stdout.splitlines()
    want = open(os.path.join(ROOT, "experiments", slug, "build_output.txt"), encoding="utf-8").read().splitlines()
    assert len(got) == len(want), "different number of output lines"
    for g, w in zip(got, want):
        assert NUM.sub("#", g) == NUM.sub("#", w), f"\n got: {g}\nwant: {w}"
        for a, b in zip(map(float, NUM.findall(g)), map(float, NUM.findall(w))):
            assert (abs(a) < 1e-9 and abs(b) < 1e-9) or abs(a - b) <= 1e-6 * max(1.0, abs(b)), f"\n got: {g}\nwant: {w}"
