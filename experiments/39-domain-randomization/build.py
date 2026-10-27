"""Lesson 39 · Build it: a domain-randomization sampler and the cross-entropy method, checked against the Experiment.

The plant and its batched MuJoCo rollout are the Experiment's (one SO-101-like joint through a measurement delay);
what you write is the sampler, the coverage check, the averaged cost and the optimizer.

Step 1  The sampler: log-uniform scales and a uniform delay, checked against their formulas.
Step 2  Coverage: which of the 32 held-out real plants each width can draw at all.
Step 3  The averaged cost: re-score the Experiment's tuned controllers on the nominal and the real plants.
Step 4  CEM: a toy quadratic, then domain randomization at w = 0.5, which reproduces the Experiment's row.
Use it  SciPy's loguniform distribution, and MuJoCo model fields as the randomization knobs.

The functions live in physai/sim2real.py. Run:  uv run python experiments/39-domain-randomization/build.py  (~5 s)
"""
import importlib.util
import math
import os
import sys

import mujoco
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "..")]
from physai.sim2real import cem, dr_sample, expected_cost, inside_support  # noqa: E402

spec = importlib.util.spec_from_file_location("exp39", os.path.join(HERE, "experiment.py"))
exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp)                       # the Experiment's plant, reference and batched rollout

REAL = exp.real_plants(exp.N_TEST)
TEST = exp.Batch(exp.N_TEST)
THR = math.radians(2.0)

print(f"MuJoCo {mujoco.__version__} | NumPy {np.__version__}")

print("\n== Step 1. The sampler at w = 0.5, 100,000 draws ==")
p = dr_sample(np.random.default_rng(0), 100_000, 0.5)
for k in ("mass", "damping", "friction"):
    print(f"{k:<9s} min {p[k].min():.4f} max {p[k].max():.4f} (2^-0.5 = {2 ** -0.5:.4f}, 2^0.5 = {2 ** 0.5:.4f}), "
          f"median {np.median(p[k]):.4f}, share above 1: {np.mean(p[k] > 1):.3f}")
print(f"delay     min {p['delay'].min() * 1e3:.2f} ms, max {p['delay'].max() * 1e3:.2f} ms (formula 5-25 ms)")
same = dr_sample(np.random.default_rng(7), 10, 0.5)
ref = exp.sample_plants(np.random.default_rng(7), 10, 0.5)
print(f"same draws as the Experiment's sampler (seed 7): {all(np.array_equal(same[k], ref[k]) for k in ref)}")

print("\n== Step 2. Coverage: real plants inside the training support (of 32) ==")
print(f"{'w':>5s} {'mass':>5s} {'damping':>8s} {'friction':>9s} {'delay':>6s} {'all four':>9s}")
for w in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
    ins = inside_support(REAL, w)
    every = ins["mass"] & ins["damping"] & ins["friction"] & ins["delay"]
    print(f"{w:5.2f} " + " ".join(f"{int(ins[k].sum()):>{n}d}" for k, n in
                                  (("mass", 5), ("damping", 8), ("friction", 9), ("delay", 6))) + f" {int(every.sum()):9d}")
print(f"largest real friction scale {REAL['friction'].max():.2f}, largest real delay {REAL['delay'].max() * 1e3:.1f} ms")

print("\n== Step 3. The averaged cost: the Experiment's tuned controllers, re-scored ==")
print(f"{'tuned at':<9s} {'Kp':>7s} {'Kd':>6s} {'kg':>6s} {'nominal deg':>12s} {'real mean deg':>14s} {'real < 2 deg':>13s}")
for w, gains in ((0.0, (117.36, 0.914, 0.679)), (0.5, (55.43, 3.244, 0.563))):
    g = np.array([gains])
    nominal = expected_cost(TEST.rollout, g, exp.nominal_plants(exp.N_TEST))[0]   # 32 identical copies
    per_plant = TEST.rollout(np.tile(g, (exp.N_TEST, 1)), REAL)
    print(f"w = {w:<5.2f} {gains[0]:7.2f} {gains[1]:6.3f} {gains[2]:6.3f} {math.degrees(nominal):12.3f} "
          f"{math.degrees(expected_cost(TEST.rollout, g, REAL)[0]):14.3f} {int((per_plant < THR).sum()):>10d}/32")

print("\n== Step 4. CEM: a toy quadratic, then domain randomization at w = 0.5 ==")
target = np.array([1.0, -2.0, 0.5])
toy = lambda x, rng: ((x - target) ** 2).sum(axis=1) + 0.01 * rng.standard_normal(len(x))  # noqa: E731
mu = cem(toy, np.zeros(3), np.ones(3) * 2, -5.0, 5.0, np.random.default_rng(0))
print(f"toy: minimum at {target}, CEM finds [{', '.join(f'{v:.3f}' for v in mu)}]")
TRAIN = exp.Batch(exp.POP * exp.K_SETS)


def dr_cost(x, rng, w=0.5):
    """x: (log Kp, log Kd, kg) per candidate; every candidate is scored on the same K freshly drawn plants."""
    gains = np.column_stack([np.exp(x[:, 0]), np.exp(x[:, 1]), x[:, 2]])
    return expected_cost(TRAIN.rollout, gains, dr_sample(rng, exp.K_SETS, w))


mu0, sd0 = np.array([math.log(20.0), 0.0, 0.0]), np.array([1.5, 1.5, 1.0])
lo, hi = np.array([0.0, math.log(0.01), -3.0]), np.array([math.log(1000.0), math.log(30.0), 3.0])
mu = cem(dr_cost, mu0, sd0, lo, hi, np.random.default_rng(7))
g = np.array([[math.exp(mu[0]), math.exp(mu[1]), mu[2]]])
real = TEST.rollout(np.tile(g, (exp.N_TEST, 1)), REAL)
print(f"w = 0.5: Kp {g[0, 0]:.2f}, Kd {g[0, 1]:.3f}, kg {g[0, 2]:.3f}; nominal "
      f"{math.degrees(expected_cost(TEST.rollout, g, exp.nominal_plants(exp.N_TEST))[0]):.3f} deg, real mean {math.degrees(real.mean()):.3f} deg, "
      f"{int((real < THR).sum())}/32 below 2 deg")

print("\n== Use it. SciPy's log-uniform distribution; MuJoCo model fields as the knobs ==")
ks = stats.kstest(p["mass"], stats.loguniform(2 ** -0.5, 2 ** 0.5).cdf)
print(f"Kolmogorov-Smirnov, our mass scales vs scipy.stats.loguniform(2^-0.5, 2^0.5): D = {ks.statistic:.4f}, "
      f"p = {ks.pvalue:.2f}")
m = TEST.m
print(f"randomized fields in the batched model: body_mass {m.body_mass[1:].shape}, dof_damping {m.dof_damping.shape}, "
      f"dof_frictionloss {m.dof_frictionloss.shape} (one entry per copy, written before each rollout)")
