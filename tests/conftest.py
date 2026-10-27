"""Shared test helpers."""
import os
import platform
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?", re.I)
# build_output.txt files are recorded on an Apple Silicon Mac (the lessons name the hardware).
RECORDED_HERE = platform.system() == "Darwin" and platform.machine() == "arm64"


def assert_build_output(slug):
    """Run experiments/<slug>/build.py and compare with its committed build_output.txt.

    The text must match exactly everywhere. On the recording platform numbers must also match to
    1e-6 relative, except round-off-sized values (both below 1e-9 in magnitude), which only need to
    stay that small. On other CPUs contact-rich simulations, chaotic rollouts and local optimizers
    can land on slightly different numbers, so there only the structure is checked; the physai tests
    against MuJoCo still run in full.
    """
    got = subprocess.run([sys.executable, os.path.join(ROOT, "experiments", slug, "build.py")],
                         capture_output=True, text=True, check=True).stdout.splitlines()
    want = open(os.path.join(ROOT, "experiments", slug, "build_output.txt"), encoding="utf-8").read().splitlines()
    assert len(got) == len(want), "different number of output lines"
    for g, w in zip(got, want):
        assert " ".join(NUM.sub("#", g).split()) == " ".join(NUM.sub("#", w).split()), f"\n got: {g}\nwant: {w}"
        if not RECORDED_HERE:
            continue
        for a, b in zip(map(float, NUM.findall(g)), map(float, NUM.findall(w))):
            assert (abs(a) < 1e-9 and abs(b) < 1e-9) or abs(a - b) <= 1e-6 * max(1.0, abs(b)), f"\n got: {g}\nwant: {w}"
