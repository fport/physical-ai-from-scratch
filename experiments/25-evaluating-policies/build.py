"""Lesson 25 · Build it: success-rate statistics from scratch, against SciPy and the Experiment's tables.

Step 1  Wald and Wilson intervals vs scipy.stats.binomtest.
Step 2  Exact coverage by a binomial sum (table B's "exact" columns).
Step 3  What a 9/10 means, and how many trials a claim needs (tables C and D).
Step 4  Fisher's exact test by hand vs scipy.stats.fisher_exact.
Step 5  Exact power of the comparison A vs B (table E, which resampled the pools).

The functions live in physai/evaluation.py (the lesson shows them step by step).
Run:  uv run python experiments/25-evaluating-policies/build.py      (~2 s, deterministic)
"""
import os
import sys

import numpy as np
import scipy
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "..")]
from physai.evaluation import binom_pmf, coverage, fisher_p, power, trials_needed, wald, wilson  # noqa: E402

print(f"NumPy {np.__version__} | SciPy {scipy.__version__}")
PA, PB = 221 / 300, 263 / 300          # the pool rates of policies A and B (Experiment, part A)


def fmt(iv):
    return f"[{iv[0]:.3f}, {iv[1]:.3f}]"


print("\n== Step 1. Wald and Wilson intervals (z = 1.96) vs scipy.stats.binomtest ==")
print(f"{'k/n':>7} {'Wald':>16} {'Wilson':>16} {'scipy Wilson':>16}")
for k, n in ((9, 10), (10, 10), (18, 20), (221, 300)):
    ci = stats.binomtest(k, n).proportion_ci(method="wilson")
    print(f"{f'{k}/{n}':>7} {fmt(wald(k, n)):>16} {fmt(wilson(k, n)):>16} {fmt((ci.low, ci.high)):>16}")
z = stats.norm.ppf(0.975)
diff = max(abs(a - b) for k in range(21) for a, b in zip(wilson(k, 20, z), stats.binomtest(k, 20).proportion_ci(method="wilson")))
print(f"scipy uses z = {z:.6f}; with that z, max difference over k = 0..20 of n = 20: {diff:.1e}")

print("\n== Step 2. Exact coverage at p = 221/300 by a binomial sum ==")
pmf_err = max(abs(binom_pmf(k, 50, PA) - stats.binom.pmf(k, 50, PA)) for k in range(51))
print(f"binom_pmf vs scipy.stats.binom.pmf (n = 50): max difference {pmf_err:.1e}")
print(f"{'n':>4} {'Wilson coverage':>16} {'Wald coverage':>14}")
for n in (5, 10, 20, 50, 100):
    print(f"{n:4d} {coverage(wilson, n, PA):16.3f} {coverage(wald, n, PA):14.3f}")

print("\n== Step 3. A 9/10 claim, and trials for a Wilson half-width target ==")
for name, p in (("A", PA), ("B", PB)):
    print(f"policy {name} (p = {p:.3f}): P(>= 9/10) = {binom_pmf(9, 10, p) + binom_pmf(10, 10, p):.3f}")
print(f"{'observed rate':>14}{'+-0.10':>8}{'+-0.05':>8}{'+-0.02':>8}")
for p in (0.5, 0.7, 0.9, 0.95):
    print(f"{p:>14.2f}" + "".join(f"{trials_needed(p, h):>8}" for h in (0.10, 0.05, 0.02)))

print("\n== Step 4. Fisher's exact test by hand vs scipy.stats.fisher_exact ==")
diffs = [abs(fisher_p(a, 20, b, 20) - stats.fisher_exact([[a, 20 - a], [b, 20 - b]])[1])
         for a in range(21) for b in range(21)]
print(f"all {len(diffs)} tables k_A/20 vs k_B/20: max p-value difference {max(diffs):.1e}")
print(f"14/20 vs 18/20: p = {fisher_p(14, 20, 18, 20):.3f} | 74/100 vs 88/100: p = {fisher_p(74, 100, 88, 100):.3f}")

print("\n== Step 5. Exact power of Fisher's test, A (p = 0.737) vs B (p = 0.877), n trials each ==")
print(f"{'n':>4} {'P(Fisher p < 0.05)':>19} {'P(B scores <= A)':>17}")
for n in (10, 20, 50, 100):
    pa = [binom_pmf(k, n, PA) for k in range(n + 1)]
    pb = [binom_pmf(k, n, PB) for k in range(n + 1)]
    worse = sum(pa[a] * pb[b] for a in range(n + 1) for b in range(a + 1))
    print(f"{n:4d} {power(PA, PB, n):19.3f} {worse:17.3f}")
