"""Lesson 33 · PPO for continuous control.

Train ppo.py on the lesson-32 SO-101 reach task (position actions, 50 Hz, 1.6 s episodes).
  A. Reference config, 3 seeds: success over 100 fixed evaluation episodes (deterministic policy)
     with a Wilson 95% interval; learning curve; wall-clock per run.
  B. Sweep the discount gamma (= the planning horizon in seconds), 3 seeds each.
  C. Break it: remove the clip (eps = 100) and reuse each batch for 30 epochs, 3 seeds.

Runs are independent processes (fixed seeds, 1 torch thread each), so results do not depend on
scheduling. Wall-clock times do: other jobs were running on the same Mac.

Run:  python experiment.py           (15 runs; 222 s wall-clock for output.txt on a loaded Apple M5, about
                                      4-6 min under heavier load; the last line prints the wall-clock time)
      python experiment.py --quick   (seed 0 only, 5 runs: reproduces the seed-0 entries of every table,
                                      NOT the pooled numbers)
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import mujoco
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "32-rl-environment"))
from ppo import PPOConfig, evaluate, train, wilson  # noqa: E402
from reach_env import ReachEnv  # noqa: E402

BASE = dict(total_steps=300_000, lr=1e-3, gamma=0.95, lam=0.95, clip=0.2, epochs=10, log_std_init=-1.0)
SEEDS = (0, 1, 2)
GAMMAS = (0.9, 0.95, 0.99, 0.995)
N_EVAL = 100


def make_env(i):
    return ReachEnv()


def job(args):
    tag, overrides, seed = args
    cfg = PPOConfig(seed=seed, **{**BASE, **overrides})
    pi, log = train(make_env, cfg)
    ev = evaluate(make_env, pi, N_EVAL)
    curve = {r["steps"]: r["succ"] for r in log["iters"]}
    kl = [r["kl"] for r in log["iters"]]
    return tag, seed, ev, log["seconds"], curve, kl


def fmt_rate(k, n):
    lo, hi = wilson(k, n)
    return f"{k:>4}/{n:<4} {100*k/n:5.1f}%  [{100*lo:5.1f}, {100*hi:5.1f}]"


def main():
    global SEEDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="seed 0 only; does not reproduce the pooled table")
    if ap.parse_args().quick:
        SEEDS = (0,)
        print("QUICK MODE: seed 0 only; pooled numbers and means do NOT reproduce output.txt")
    print(f"mujoco {mujoco.__version__} | numpy {np.__version__} | torch {torch.__version__} | "
          f"Apple M5, CPU, 1 torch thread per run")
    print(f"reference config: {BASE}, n_envs 16, rollout 64 -> 1024 samples/iter, minibatch 256, "
          f"vf_coef 0.5, ent_coef 0, Adam eps 1e-5, grad-norm 0.5")
    jobs = [("ref", {}, s) for s in SEEDS]
    jobs += [(f"gamma={g}", {"gamma": g}, s) for g in GAMMAS if g != BASE["gamma"] for s in SEEDS]
    jobs += [("noclip", {"clip": 100.0, "epochs": 30}, s) for s in SEEDS]
    t0 = time.perf_counter()
    with Pool(min(6, os.cpu_count(), len(jobs))) as p:
        res = p.map(job, jobs)
    wall = time.perf_counter() - t0
    by = {}
    for tag, seed, ev, sec, curve, kl in res:
        by.setdefault(tag, []).append((seed, ev, sec, curve, kl))

    print(f"\n[A] Reference config, success over {N_EVAL} evaluation episodes (seeds 10000..10099), "
          f"deterministic mean action")
    print(f"  {'seed':>4}  {'success':<31}{'median final d':>15}{'eval return':>13}{'train s':>9}")
    K = 0
    for seed, ev, sec, _, _ in by["ref"]:
        K += ev["k"]
        print(f"  {seed:>4}  {fmt_rate(ev['k'], ev['n']):<31}{ev['final_d']*1000:>12.1f} mm{ev['ret']:>13.2f}{sec:>9.0f}")
    print(f"  pooled {fmt_rate(K, N_EVAL*len(SEEDS))}")
    print("\n  learning curve: training success (stochastic policy, last 50 episodes) vs env steps")
    marks = [50_176, 100_352, 150_528, 200_704, 250_880, 299_008]
    print("  " + f"{'seed':>4}" + "".join(f"{m//1000:>8}k" for m in marks))
    for seed, _, _, curve, _ in by["ref"]:
        print("  " + f"{seed:>4}" + "".join(f"{curve[m]:>9.2f}" for m in marks))

    print(f"\n[B] Sweep gamma (horizon 1/(1-gamma) steps x 20 ms), {len(SEEDS)} seed{'s' if len(SEEDS) > 1 else ''} each")
    print(f"  {'gamma':>6}{'horizon':>10}  {'success per seed':<20}{'pooled success':<34}{'median final d (mm)':>20}")
    for g in GAMMAS:
        rows = by["ref"] if g == BASE["gamma"] else by[f"gamma={g}"]
        ks = [ev["k"] for _, ev, _, _, _ in rows]
        fds = [ev["final_d"] * 1000 for _, ev, _, _, _ in rows]
        print(f"  {g:>6}{0.02/(1-g):>8.1f} s  {'/'.join(str(k) for k in ks):<20}"
              f"{fmt_rate(sum(ks), N_EVAL*len(rows)):<34}{' / '.join(f'{f:.1f}' for f in fds):>20}")

    print(f"\n[C] Break it: clip eps = 100 (no trust region) and 30 epochs per batch, {len(SEEDS)} seed{'s' if len(SEEDS) > 1 else ''}")
    print(f"  {'seed':>4}  {'success':<31}{'median final d':>15}  approx KL per update at iter 1 / 10 / 50 / 100 / 292")
    for seed, ev, _, _, kl in by["noclip"]:
        print(f"  {seed:>4}  {fmt_rate(ev['k'], ev['n']):<31}{ev['final_d']*1000:>12.1f} mm  "
              + " / ".join(f"{kl[i]:.3g}" for i in (0, 9, 49, 99, len(kl) - 1)))
    ref_kl = [np.mean([r[4][i] for r in by["ref"]]) for i in (0, 9, 49, 99, -1)]
    print(f"  reference (eps 0.2, 10 epochs), mean of {len(SEEDS)} seed{'s' if len(SEEDS) > 1 else ''}: " + " / ".join(f"{k:.3g}" for k in ref_kl))
    print(f"\n{len(jobs)} training runs, {wall:.0f} s wall-clock in total "
          f"(6 worker processes, shared machine)")


if __name__ == "__main__":
    main()
