"""Lesson 25 · Evaluating robot policies.

A "policy" with an unknown success rate: the scripted SO-101 pick of lesson 16 aiming at a noisy estimate of the
cube position (true position + Gaussian aim noise, as if it came from an imperfect perception module).
Success is physical (no weld): cube bottom >= 4 cm above the floor at the end and both jaws touching it.

  A. Two pools of 300 simulated trials each: policy A (aim noise 9 mm per axis) and policy B (6 mm).
     The 300-trial rate stands in for the long-run rate p. A looser criterion ("the jaw closed on the cube")
     is counted on the same trials.
  B. Sweep the number of trials n: draw 2000 n-trial evaluations from pool A (with replacement) ->
     Wilson 95% interval width, how often it covers the pool rate, and the same for the Wald interval.
  C. What a "9/10" claim means: its Wilson interval and how often policies A and B produce >= 9/10.
  D. How many trials a claim needs (Wilson half-width target).
  E. Comparing two policies: for n trials each, how often a Fisher exact test (p < 0.05) separates A from B.

Helpers (wilson, wald, trials_needed, fisher_p) and the simulation harness `Scene` are imported by lessons 26-29.

Run:  python experiment.py      (deterministic; ~1 min CPU on Apple M5, longer wall time on a loaded machine)
"""
import math
import os
import sys
import time

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(EXP, "16-grasping-and-pick-place"))
sys.path.insert(0, EXP)
import experiment as pick16  # noqa: E402  (lesson 16: build, reset, ids, scripted_pick, jaw_contacts)


# ------------------------------------------------------------------ helpers (imported by later lessons)
def wilson(k, n, z=1.96):
    """Wilson score interval for k successes in n Bernoulli trials (Wilson 1927)."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def wald(k, n, z=1.96):
    """Textbook normal-approximation interval p +- z sqrt(p(1-p)/n). Shown only to see it fail."""
    p = k / n
    h = z * math.sqrt(p * (1 - p) / n)
    return max(0.0, p - h), min(1.0, p + h)


def trials_needed(p, half_width, z=1.96):
    """Smallest n whose Wilson interval at observed rate p has half-width <= half_width."""
    n = 1
    while True:
        k = round(p * n)
        lo, hi = wilson(k, n, z)
        if (hi - lo) / 2 <= half_width:
            return n
        n += 1


def fisher_p(k1, n1, k2, n2):
    """Two-sided Fisher exact test p-value for k1/n1 vs k2/n2 (scipy)."""
    from scipy.stats import fisher_exact
    return float(fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def binom_pmf(k, n, p):
    return math.comb(n, k) * p ** k * (1 - p) ** (n - k)


# ------------------------------------------------------------------ simulation harness (reused by lessons 26-29)
class Scene:
    """One lesson-16 scene (scene.xml + free 3 cm cube) reused for every trial; place() resets it."""

    def __init__(self):
        self.m, self.d = pick16.build(pick16.CUBE_HOME)
        self.I = pick16.ids(self.m)
        self.cq = self.m.joint(self.m.body("cube").jntadr[0]).qposadr[0]

    def place(self, xy):
        """Start state: arm at rest in HOME_Q, jaw open, cube at (x, y) on the floor."""
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[self.cq:self.cq + 3] = [xy[0], xy[1], pick16.CUBE_HALF]
        self.d.qpos[self.cq + 3:self.cq + 7] = [1, 0, 0, 0]
        pick16.reset(self.m, self.d)

    def obs_from_qpos(self, q):
        """5 arm joint angles, jaw angle, cube position (x, y, z)."""
        return np.concatenate([q[self.I["qadr"]], [q[self.I["grip_q"]]], q[self.cq:self.cq + 3]])

    def obs(self):
        return self.obs_from_qpos(self.d.qpos)

    def apply(self, a):
        self.d.ctrl[self.I["act"]] = a[:5]
        self.d.ctrl[self.I["grip"]] = a[5]

    def step(self, n):
        for _ in range(n):
            mujoco.mj_step(self.m, self.d)

    def success(self):
        """Physical test, no weld: cube bottom >= 4 cm above the floor and both jaws touch the cube."""
        nf, nm = pick16.jaw_contacts(self.m, self.d, self.I)
        bottom = self.d.xpos[self.I["cube_body"]][2] - pick16.CUBE_HALF
        return bool(bottom >= pick16.SUCCESS_LIFT_M and nf > 0 and nm > 0)


# ------------------------------------------------------------------ the noisy-aim scripted policy
def run_pool(scene, sigma, n, seed):
    """n trials: cube uniform in the lesson-26 area, the pick aims at cube + N(0, sigma^2) per axis."""
    rng = np.random.default_rng(seed)
    succ, closed = [], []
    for _ in range(n):
        xy = np.array([rng.uniform(0.16, 0.28), rng.uniform(-0.08, 0.08)])
        aim = xy + rng.normal(0.0, sigma, 2)
        scene.place(xy)
        r = pick16.scripted_pick(scene.m, scene.d, np.array([aim[0], aim[1], pick16.CUBE_HALF]))
        succ.append(r["success"])
        closed.append(bool(np.isfinite(r["close_s"])))   # the jaw settled with both jaws on the cube
    return np.array(succ), np.array(closed)


def main():
    t0 = time.process_time()
    import scipy
    print(f"mujoco {mujoco.__version__} · numpy {np.__version__} · scipy {scipy.__version__}")

    N_POOL = 300
    scene = Scene()
    print(f"\n[A] Pools of {N_POOL} simulated trials, cube uniform in x 16-28 cm, y -8..8 cm")
    pools = {}
    print(f"{'policy':>8}{'aim noise':>11}{'success':>10}{'rate':>7}{'Wilson 95%':>16}{'jaw closed on cube':>21}")
    for name, sigma, seed in (("A", 0.009, 1), ("B", 0.006, 2)):
        s, c = run_pool(scene, sigma, N_POOL, seed)
        pools[name] = s
        lo, hi = wilson(s.sum(), N_POOL)
        print(f"{name:>8}{f'{sigma * 1000:.0f} mm':>11}{f'{s.sum()}/{N_POOL}':>10}{s.mean():>7.3f}"
              f"{f'[{lo:.3f}, {hi:.3f}]':>16}{f'{c.sum()}/{N_POOL} ({c.mean():.3f})':>21}")
    pA, pB = pools["A"].mean(), pools["B"].mean()
    print("(the pool rate is used as the long-run rate p below)")

    rng = np.random.default_rng(25)
    R = 2000
    print(f"\n[B] Number of trials n: {R} evaluations of n trials drawn from pool A (p = {pA:.3f})")
    print(f"{'n':>5}{'mean Wilson width':>19}{'Wilson covers p':>17}{'Wald covers p':>15}"
          f"{'exact Wilson cov.':>19}{'exact Wald cov.':>17}{'k/n range (5-95%)':>20}")
    for n in (5, 10, 20, 50, 100):
        ks = np.array([pools["A"][rng.choice(N_POOL, n, replace=True)].sum() for _ in range(R)])
        wi = [wilson(k, n) for k in ks]
        wa = [wald(k, n) for k in ks]
        width = np.mean([h - l for l, h in wi])
        cw = np.mean([l <= pA <= h for l, h in wi])
        ca = np.mean([l <= pA <= h for l, h in wa])
        # exact coverage under a binomial(n, pA) model
        ew = sum(binom_pmf(k, n, pA) for k in range(n + 1) if wilson(k, n)[0] <= pA <= wilson(k, n)[1])
        ea = sum(binom_pmf(k, n, pA) for k in range(n + 1) if wald(k, n)[0] <= pA <= wald(k, n)[1])
        q5, q95 = np.percentile(ks / n, [5, 95])
        print(f"{n:>5}{width:>19.3f}{cw:>17.3f}{ca:>15.3f}{ew:>19.3f}{ea:>17.3f}{f'{q5:.2f}-{q95:.2f}':>20}")

    print("\n[C] A '9/10' claim")
    lo, hi = wilson(9, 10)
    print(f"Wilson 95% for 9/10: [{lo:.3f}, {hi:.3f}];  for 10/10: [{wilson(10, 10)[0]:.3f}, 1.000];  "
          f"Wald for 10/10: [{wald(10, 10)[0]:.3f}, {wald(10, 10)[1]:.3f}];  Wilson for 18/20: "
          f"[{wilson(18, 20)[0]:.3f}, {wilson(18, 20)[1]:.3f}]")
    for name, p in (("A", pA), ("B", pB)):
        exact = sum(binom_pmf(k, 10, p) for k in (9, 10))
        sim = np.mean([pools[name][rng.choice(N_POOL, 10, replace=True)].sum() >= 9 for _ in range(R)])
        print(f"policy {name} (p = {p:.3f}): P(>= 9/10) = {exact:.3f} binomial, {sim:.3f} drawn from the pool")

    print("\n[D] Trials needed for a Wilson 95% half-width target")
    print(f"{'observed rate':>14}{'+-0.10':>8}{'+-0.05':>8}{'+-0.02':>8}")
    for p in (0.5, 0.7, 0.9, 0.95):
        print(f"{p:>14.2f}" + "".join(f"{trials_needed(p, h):>8}" for h in (0.10, 0.05, 0.02)))

    print(f"\n[E] A (p = {pA:.3f}) vs B (p = {pB:.3f}): n trials each, {R // 2} evaluation pairs drawn from the pools")
    print(f"{'n':>5}{'Fisher p < 0.05':>17}{'Wilson intervals disjoint':>27}{'B looks worse or equal':>24}")
    for n in (10, 20, 50, 100):
        sig, disj, wrong = 0, 0, 0
        for _ in range(R // 2):
            kA = pools["A"][rng.choice(N_POOL, n, replace=True)].sum()
            kB = pools["B"][rng.choice(N_POOL, n, replace=True)].sum()
            sig += fisher_p(kA, n, kB, n) < 0.05
            a, b = wilson(kA, n), wilson(kB, n)
            disj += a[1] < b[0] or b[1] < a[0]
            wrong += kB <= kA
        print(f"{n:>5}{sig / (R // 2):>17.3f}{disj / (R // 2):>27.3f}{wrong / (R // 2):>24.3f}")
    print(f"CPU time {time.process_time() - t0:.0f} s", file=sys.stderr)


if __name__ == "__main__":
    main()
