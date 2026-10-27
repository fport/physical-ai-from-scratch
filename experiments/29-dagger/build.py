"""Lesson 29 · Build it: distribution shift and DAgger on a small lane-keeping task, checked at every step.

Step 1  The toy task: a car on a curved lane, a feedback expert, noise-free demonstrations, gusts at test time.
Step 2  Plain BC with a nearest-neighbour learner: success for 20 / 40 / 60 demos, drift away from the data.
Step 3  DAgger: the learner drives, the expert labels, everything is kept (4 iterations x 10 episodes).
Step 4  Break it: DAgger without aggregation.

A smaller version of the Experiment: the same loop and the same drift measure, but a 3-number state and a
1-nearest-neighbour learner instead of the SO-101 pick and an MLP, so that the whole script runs in seconds.
The functions live in physai/imitation.py (the lesson shows them step by step).
Run:  uv run python experiments/29-dagger/build.py      (~3 s, deterministic)
"""
import os
import sys

import numpy as np
import scipy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "..")]
from physai.imitation import dagger, knn_policy, nearest_distance  # noqa: E402

print(f"NumPy {np.__version__} | SciPy {scipy.__version__}")
DT, T, HALF = 0.05, 200, 0.5            # 20 Hz, 10 s per episode, lane half-width 0.5 m
GUST = 0.04                             # rad, std of a random heading change per step (test and DAgger rollouts)
N_TEST = 50


def lane(x):
    """Centre line y = sin(x / 2) and its heading."""
    return np.sin(0.5 * x), np.arctan(0.5 * np.cos(0.5 * x))


def expert(S):
    """Steer towards the lane's heading and back to its centre (rad/s), for any state (x, y, heading)."""
    x, y, psi = np.atleast_2d(S).T
    yc, pc = lane(x)
    return np.clip(2.0 * (pc - psi) - 3.0 * (y - yc), -2.0, 2.0)


def drive(policy, s0, rng=None):
    """One episode at 1 m/s; rng=None means no gusts. Returns the visited states and whether it stayed in lane."""
    s, states = np.array(s0, float), []
    for _ in range(T):
        states.append(s.copy())
        x, y, psi = s
        psi = psi + policy(s)[0] * DT + (rng.normal(0, GUST) if rng is not None else 0.0)
        s = np.array([x + np.cos(psi) * DT, y + np.sin(psi) * DT, psi])
        if abs(s[1] - lane(s[0])[0]) > HALF:
            return np.array(states), False
    return np.array(states), True


def starts(n, rng):
    return np.c_[np.zeros(n), rng.uniform(-0.05, 0.05, n), np.full(n, lane(0.0)[1])]


def evaluate(policy):
    rng = np.random.default_rng(99)                          # the same 50 test episodes for every policy
    runs = [drive(policy, s0, rng) for s0 in starts(N_TEST, rng)]
    return sum(ok for _, ok in runs), [S for S, _ in runs]


def windows(runs, frames):
    """Mean distance of visited positions (x, y) to the nearest demonstrated position, in 2.5 s windows [m]."""
    d = [nearest_distance(S[:, :2], frames[:, :2]) for S in runs]
    return [np.mean(np.concatenate([di[a:b] for di in d if len(di) > a])) for a, b in ((0, 50), (50, 100), (100, 150), (150, 200))
            if any(len(di) > a for di in d)]


def fmt_w(w):
    return " ".join(f"{v:6.3f}" for v in w) + "      -" * (4 - len(w))


demos = [drive(expert, s0)[0] for s0 in starts(60, np.random.default_rng(0))]

print("\n== Step 1. The task and the expert ==")
k, _ = evaluate(expert)
D20 = np.concatenate(demos[:20])
print(f"lane y = sin(x/2), half-width {HALF} m, 1 m/s, {T} steps of {DT} s; gust std {GUST} rad per step")
print(f"20 noise-free demonstrations: {len(D20)} states, lateral offset from the centre at most "
      f"{np.abs(D20[:, 1] - lane(D20[:, 0])[0]).max() * 100:.1f} cm")
print(f"expert with gusts: {k}/{N_TEST} episodes stay in the lane")

print("\n== Step 2. Plain BC: a 1-nearest-neighbour learner on the demonstrations ==")
pts = np.random.default_rng(1).normal(size=(200, 3))
brute = np.abs(pts[:, None, :] - D20[None, :, :]).max(-1).min(1)
print(f"nearest_distance vs brute force on 200 points: max diff {np.abs(nearest_distance(pts, D20) - brute).max():.1e}")
print("  demos  success   dist to demos [m] in 0-2.5 / 2.5-5 / 5-7.5 / 7.5-10 s")
for n in (20, 40, 60):
    X = np.concatenate(demos[:n])
    k, runs = evaluate(knn_policy(X, expert(X)))
    print(f"{n:>7}  {k:>4}/{N_TEST}   {fmt_w(windows(runs, D20))}")
_, runs = evaluate(expert)
print(f" expert  {N_TEST:>4}/{N_TEST}   {fmt_w(windows(runs, D20))}")

print("\n== Step 3. DAgger: 10 learner episodes per iteration, labelled by the expert, all data kept ==")
rng = np.random.default_rng(1)
plan = [starts(10, rng) for _ in range(4)]
collect = np.random.default_rng(2)
pols = dagger(knn_policy, expert, lambda p, s0: drive(p, s0, collect)[0], D20, expert(D20), plan)
print(" iter  expert-labelled episodes  success   dist to demos [m] per window")
for i, p in enumerate(pols):
    k, runs = evaluate(p)
    print(f"{i:>5}  {20 + 10 * i:>24}  {k:>4}/{N_TEST}   {fmt_w(windows(runs, D20))}")

print("\n== Step 4. Break it: train each iteration only on its newest 10 learner episodes ==")
collect = np.random.default_rng(2)
pols = dagger(knn_policy, expert, lambda p, s0: drive(p, s0, collect)[0], D20, expert(D20), plan, aggregate=False)
print("success per iteration 0..4: " + " ".join(f"{evaluate(p)[0]}/{N_TEST}" for p in pols))
