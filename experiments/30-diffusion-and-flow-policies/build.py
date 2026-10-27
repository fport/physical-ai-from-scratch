"""Lesson 30 · Build it: why MSE averages and how flow matching samples, checked at every step.

Step 1  The lesson's 200 two-sided demonstrations; the least-squares answer is their mean, and it collides.
Step 2  The flow-matching training pair, and the velocity field it regresses (1-D Monte Carlo check).
Step 3  Euler sampling along the exact field of the 200 demonstrations: collisions vs the number of steps.
Step 4  What the exact field does at many steps: it returns the training demonstrations (memorization).

The demonstrations, the obstacle test and the trials are the Experiment's (seed 0). Instead of training a
network, Steps 3-4 integrate the exact minimizer of the flow-matching loss on those demonstrations, so this
script needs no PyTorch and runs in seconds; the Experiment's learned network is the comparison.
The functions live in physai/imitation.py (the lesson shows them step by step).
Run:  uv run python experiments/30-diffusion-and-flow-policies/build.py      (~2 s, deterministic)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "..")]
from physai.imitation import empirical_velocity, euler_sample, flow_matching_pair  # noqa: E402

print(f"NumPy {np.__version__}")
OBS_R, T_DEMO, SCALE = 0.30, 40, 2.0          # the Experiment's obstacle radius, waypoints, network scaling
S51 = np.linspace(0, 1, 51)


def demo(rng, side, x0):                      # the Experiment's demonstrator
    a = 0.55 + rng.normal(0, 0.04)
    s = np.linspace(0, 1, T_DEMO + 1)
    return np.stack([x0 * (1 - s) + side * a * np.sin(np.pi * s) ** 1.2, -1 + 2 * s], 1)


def make_data(rng, n_demo, left_frac):        # the Experiment's data: observation = start, action = path - start
    n_left = int(round(n_demo * left_frac))
    sides = np.array([-1] * n_left + [1] * (n_demo - n_left))
    rng.shuffle(sides)
    x0s = rng.uniform(-0.05, 0.05, n_demo)
    paths = np.stack([demo(rng, s, x0) for s, x0 in zip(sides, x0s)])
    return paths, sides, paths[:, 0, :], (paths[:, 1:, :] - paths[:, :1, :]).reshape(n_demo, -1)


def evaluate(predict, rng, n=200):
    """The Experiment's trials: n random starts; a path collides if any of 51 points per segment is inside."""
    xs = rng.uniform(-0.05, 0.05, n)
    fulls = [np.vstack([[x0, -1.0], np.array([x0, -1.0]) + r.reshape(-1, 2)]) for x0, r in zip(xs, predict(xs))]
    hit, left, right, absx = 0, 0, 0, []
    for f in fulls:
        pts = f[:-1, None, :] + S51[None, :, None] * (f[1:] - f[:-1])[:, None, :]
        h = bool((np.linalg.norm(pts, axis=2) < OBS_R).any())
        side = f[np.argmin(np.abs(f[:, 1])), 0]
        hit, left, right = hit + h, left + (not h and side < 0), right + (not h and side >= 0)
        absx.append(abs(f[20, 0]))
    return hit, left, right, float(np.mean(absx))


paths, sides, X, Y = make_data(np.random.default_rng(0), 200, 0.5)

print("\n== Step 1. 200 demonstrations, half left, half right: the least-squares answer ==")
F = np.c_[np.ones(len(X)), X[:, 0]]                         # linear in the start x0: E[path | x0] for this data
coef = np.linalg.lstsq(F, Y, rcond=None)[0]
mse_predict = lambda xs: np.c_[np.ones(len(xs)), xs] @ coef  # noqa: E731
print(f"demos pass the obstacle (y = 0) at x = {paths[sides < 0, 20, 0].mean():+.3f} / {paths[sides > 0, 20, 0].mean():+.3f}")
print(f"least-squares path from x0 = 0 passes at x = {mse_predict(np.array([0.0]))[0, 38]:+.3f}; "
      f"max |intercept - mean of the 200 demos| = {np.abs(coef[0] - Y.mean(0) + coef[1] * X[:, 0].mean()).max():.1e}")
h, le, ri, ax = evaluate(mse_predict, np.random.default_rng(2))
print(f"200 trials: collided {h}/200, left {le}, right {ri}")

print("\n== Step 2. The flow-matching pair and the field it regresses ==")
a1, a0 = np.array([[1.1], [-1.1]]), np.array([[0.3], [0.3]])
for t in (0.0, 0.5, 1.0):
    at, v = flow_matching_pair(a0, a1, np.full(2, t))
    print(f"t = {t:.1f}: a_t = {at.ravel()}, target a1 - a0 = {v.ravel()}")
rng = np.random.default_rng(0)
data1 = np.array([[-1.1], [1.1]])                           # two modes at x = -/+0.55, scaled by 2 like the Experiment
a1 = data1[rng.integers(0, 2, 4_000_000)]
a0 = rng.normal(size=a1.shape)
at, target = flow_matching_pair(a0, a1, np.full(len(a1), 0.5))
print("t = 0.5     a    mean target near a (Monte Carlo)   empirical_velocity")
for a in (-0.8, -0.2, 0.0, 0.2, 0.8):
    near = np.abs(at[:, 0] - a) < 0.02
    print(f"         {a:+.1f}    {target[near, 0].mean():+28.3f}   {empirical_velocity(np.array([[a]]), 0.5, data1)[0, 0]:+18.3f}")

print("\n== Step 3. Euler sampling along the exact field of the 200 demonstrations ==")
data = Y * SCALE
print("steps  collided  left  right  mean |x| at y=0   (Experiment's network: collided, mean |x|)")
net = {1: (200, 0.161), 2: (95, 0.372), 4: (17, 0.494), 8: (7, 0.533), 16: (5, 0.546), 32: (3, 0.554)}
for n in (1, 2, 4, 8, 16, 32):
    noise = np.random.default_rng(3)
    sample = lambda xs: euler_sample(lambda a, t: empirical_velocity(a, t, data), noise.normal(size=(len(xs), 80)), n) / SCALE  # noqa: E731,B023
    h, le, ri, ax = evaluate(sample, np.random.default_rng(2))
    print(f"{n:>5}  {h:>4}/200  {le:>4}  {ri:>5}  {ax:15.3f}   ({net[n][0]}/200, {net[n][1]:.3f})")
one = euler_sample(lambda a, t: empirical_velocity(a, t, data), np.random.default_rng(3).normal(size=(5, 80)), 1)
print(f"1 step: max |sample - mean of the demos| = {np.abs(one - data.mean(0)).max():.1e}")

print("\n== Step 4. Many steps: the exact field returns the training demonstrations ==")
for n in (8, 32, 128):
    s = euler_sample(lambda a, t: empirical_velocity(a, t, data), np.random.default_rng(3).normal(size=(200, 80)), n)
    nearest = np.sqrt(((s[:, None, :] - data[None]) ** 2).mean(-1)).min(1) / SCALE
    print(f"{n:>4} steps: RMS distance of a sample to its nearest demonstration, median {np.median(nearest):.4f}, max {nearest.max():.4f}")
print("demonstrations differ from each other by (median RMS, nearest pair) "
      f"{np.median(np.sort(np.sqrt(((data[:, None] - data[None]) ** 2).mean(-1)), 1)[:, 1]) / SCALE:.4f}")
