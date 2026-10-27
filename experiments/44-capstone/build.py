"""Lesson 44 · Build it: the capstone's contracts, assembled from earlier physai modules, checked against the run.

Step 1  Perception (physai.vision, lessons 19-21): the camera's estimate of each requested cube, calibrated and
        with an unmodelled tilt, vs the Experiment's medians.
Step 2  Reach (physai.foundation on physai.kinematics, lessons 09, 10, 42): the workspace radius and IK residual of
        lesson 16's three grasp points for every cube of the 5 layouts.
Step 3  The harness (physai.capstone): contracts from each logged episode, first-violation attribution, compared
        with the Experiment's layer column.
Step 4  The failure budget and what 30 episodes can certify (physai.evaluation's Wilson interval, lesson 25).

No language model and no renderer are needed: the Experiment's output.txt is the data.
Run:  uv run python experiments/44-capstone/build.py      (~2 s, deterministic)
"""
import os
import re
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "..")]
from common.menagerie import path  # noqa: E402
from physai.capstone import ORDER, attribute, estimate_xy, even_split, failure_budget, tilt  # noqa: E402
from physai.evaluation import wilson  # noqa: E402
from physai.foundation import ik_position  # noqa: E402
from physai.kinematics import chain_from_mujoco  # noqa: E402
from physai.vision import intrinsics_from_fovy  # noqa: E402

OUT = open(os.path.join(HERE, "output.txt"), encoding="utf-8").read()
LAYOUTS = [{c: np.array([float(x), float(y)]) / 100 for c, x, y in re.findall(r"(\w+) \(([-\d.]+), ([-\d.]+)\)", line)}
           for line in re.findall(r"^layout \d: (.*)$", OUT, re.M)]
WANT = {"plain": "red", "filler": "green", "terse": "blue", "asr": "red", "negation": "blue", "turkish": "red"}
CAM, FOVY, W, H = np.array([0.20, 0.0, 0.60]), 50.0, 640, 480      # lesson 21's overhead camera
TABLE_TOP, HALF, PAN_AXIS = 0.001, 0.015, np.array([0.0388, 0.0])

print(f"MuJoCo {mujoco.__version__} | NumPy {np.__version__}")
print(f"{len(LAYOUTS)} layouts read from output.txt, e.g. layout 1: "
      + ", ".join(f"{c} ({p[0] * 100:.1f}, {p[1] * 100:.1f})" for c, p in LAYOUTS[1].items()) + " cm")

print("\n== Step 1. Perception: estimate of the requested cube (plain, filler, terse x 5 layouts) ==")
m_cam = mujoco.MjModel.from_xml_string(f'<mujoco><worldbody><camera name="top" pos="{CAM[0]} {CAM[1]} {CAM[2]}" '
                                       f'fovy="{FOVY}"/></worldbody></mujoco>')
d_cam = mujoco.MjData(m_cam)
mujoco.mj_forward(m_cam, d_cam)
R_model = d_cam.cam_xmat[0].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])   # MuJoCo camera axes -> OpenCV axes
K = intrinsics_from_fovy(FOVY, W, H)
print(f"K: f = {K[0, 0]:.2f} px, centre ({K[0, 2]:.1f}, {K[1, 2]:.1f}); camera 0.6 m above the table, looking down")
reported = dict(re.findall(r"^([\d.]+)\s+([\d.]+)\s+\d+/15", OUT, re.M))
print(f"{'tilt deg':>8s} {'median err mm':>14s} {'max err mm':>11s} {'> 8 mm':>7s} {'Experiment median mm':>21s}")
for t in (0.0, 0.5, 1.0, 1.5, 2.0):
    errs = []
    for lay in LAYOUTS:
        for cmd in ("plain", "filler", "terse"):
            p = np.r_[lay[WANT[cmd]], TABLE_TOP + 2 * HALF]          # centre of the top face
            est = estimate_xy(p, K, tilt(t) @ R_model, R_model, CAM, TABLE_TOP + 2 * HALF)
            errs.append(np.linalg.norm(est - p[:2]) * 1000)
    errs = np.array(errs)
    print(f"{t:8.1f} {np.median(errs):14.1f} {errs.max():11.1f} {int((errs > 8).sum()):4d}/15 "
          f"{reported[f'{t:.1f}']:>21s}")
print(f"small-angle estimate h * alpha at 1 deg: {(CAM[2] - TABLE_TOP - 2 * HALF) * np.radians(1) * 1000:.1f} mm")

print("\n== Step 2. Reach: workspace radius and IK residual of the three grasp points ==")
m = mujoco.MjModel.from_xml_path(path("robotstudio_so101", "so101.xml"))
chain = chain_from_mujoco(m, "gripperframe")
LO, HI = m.jnt_range[:5, 0], m.jnt_range[:5, 1]
HOME = np.array([0.0, -1.2, 0.5, 1.2, 1.68])
rej = {(li, c) for li, trace in re.findall(r"^\S+ \w+\s+L(\d) sig \w+\s+(.*?)  =>", OUT, re.M)
       for c in re.findall(r"pick\((\w+) cube\)\[REJ:IK\]", trace.split("place(")[0])}   # before anything moved
print(f"{'layout':>6s} {'cube':>6s} {'radius cm':>10s} {'reach miss mm':>14s} {'ours':>7s} {'Experiment':>11s}")
for li, lay in enumerate(LAYOUTS):
    for c, xy in lay.items():
        r = np.linalg.norm(xy - PAN_AXIS)
        miss = max(min(ik_position(chain, np.r_[xy, z], q0, LO, HI)[1] for q0 in (HOME, np.zeros(5)))
                   for z in (TABLE_TOP + HALF - 0.004, TABLE_TOP + HALF + 0.055, TABLE_TOP + HALF + 0.07))
        ok = 0.12 <= r <= 0.30 and miss <= 0.005
        exp_rej = (str(li), c) in rej
        print(f"{li:6d} {c:>6s} {r * 100:10.1f} {miss * 1000:14.2f} {'accept' if ok else 'reject':>7s} "
              f"{'REJ:IK' if exp_rej else 'ok':>11s}")

print("\n== Step 3. The harness: contracts from each logged episode, first broken layer ==")
CALL = re.compile(r"(\w+)\(([^)]*)\)(?:\[([\w:]+)\])?")


def broken_contracts(cmd, calls, perc_err, success):
    """Which interface contracts an episode broke, read from its tool calls and the measured perception error."""
    want = WANT[cmd]
    picks = [(arg, mark) for name, arg, mark in calls if name == "pick" and want in arg and mark != "ERR"]
    broken = set()
    held = any(mark == "" for _, mark in picks)
    placed_after = any(name == "place" and mark == "" for name, _, mark in
                       calls[next((i for i, (n, a, mk) in enumerate(calls) if n == "pick" and want in a and mk == ""),
                                  len(calls)):])
    if not picks or (held and not placed_after):
        broken.add("language")
    if perc_err > 8.0:
        broken.add("perception")
    if any(mark.startswith("REJ") for _, mark in picks):
        broken.add("IK/limits")
    if any(mark == "EMPTY" for _, mark in picks):
        broken.add("grasp")
    if not success:
        broken.add("place")                            # nothing earlier broke, yet not done: the last layer
    return broken


firsts, agree, total = {}, 0, 0
for model, cmd, li, calls, success, layer, err in re.findall(
        r"^(\S+) (\w+)\s+L(\d) sig \w+\s+(.*?)  => claim \w+, success (\w+)(?:, layer ([\w/]+))?, perc err ([\d.]+) mm",
        OUT, re.M):
    first = attribute(broken_contracts(cmd, CALL.findall(calls), float(err), success == "True")) \
        if success == "False" else None
    firsts.setdefault(model, []).append(first)
    if success == "False":
        total += 1
        agree += first == layer
for t, cmd, li, calls, layer, err in re.findall(
        r"^  tilt ([\d.]+) (\w+)\s+L(\d) (.*?)  => claim \w+, layer ([\w/]+), perc err ([\d.]+) mm", OUT, re.M):
    total += 1
    agree += attribute(broken_contracts(cmd, CALL.findall(calls), float(err), False)) == layer
print(f"failed episodes in output.txt: {total}; our first broken layer = the Experiment's on {agree}/{total}")
for model, f in firsts.items():
    print(f"{model}: " + ", ".join(f"{layer} {sum(x == layer for x in f)}" for layer in ORDER)
          + f"; success {sum(x is None for x in f)}/{len(f)}")

print("\n== Step 4. Failure budget and what 30 episodes certify ==")
rows = failure_budget(firsts["qwen2.5:7b"], 30)
for layer, reached, passed in rows:
    print(f"{layer:<11s} reached {reached:2d} passed {passed:2d}  p = {passed / reached:.3f}")
prod = np.prod([p / r for _, r, p in rows])
lo, hi = wilson(26, 30)
print(f"product of layer pass rates {prod:.3f} = end-to-end 26/30 = {26 / 30:.3f}; Wilson 95% [{lo:.2f}, {hi:.2f}]")
print(f"even split of 0.90 over 5 layers: {even_split(0.90, 5):.3f} per layer; "
      f"a layer that passed 30/30 has Wilson lower bound {wilson(30, 30)[0]:.3f}")
print(f"clean episodes needed before the lower bound reaches {even_split(0.90, 5):.3f}: "
      f"{next(n for n in range(1, 1000) if wilson(n, n)[0] >= even_split(0.90, 5))}")
