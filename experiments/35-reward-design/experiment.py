"""Lesson 35 · Reward design and reward hacking.

Same PPO (lesson 33 reference config), same SO-101 reach task (absolute joint targets, 50 Hz),
same 300k-step budget; only the reward (and in one case the termination rule) changes.
  A. Reward shapes: dense -d, sparse 1[d < 2 cm], sparse + potential-based shaping
     F = gamma * Phi(s') - Phi(s) with Phi = -20 d, gamma = 0.95 (= the PPO discount).
  B. Reward hacking: a positive "closeness" reward 1 - tanh(d / 5 cm) while the episode ends on
     success. Compared with -d under the same termination rule, and with a scripted IK controller
     scored by the same written reward.
  C. Sweep the action-rate penalty weight w in r = -d - w * ||a_t - a_{t-1}||^2
     -> success vs smoothness (torque rate) and energy (mean |mechanical power|).
Each cell: 3 seeds x 100 evaluation episodes (fixed seeds, deterministic policy).

Run:  python experiment.py      (33 runs; 1904 s wall-clock on a heavily loaded Apple M5; the script
                                 prints its own wall-clock time at the end)
      python experiment.py --quick
                                 seed 0 only (11 runs, roughly a third of the time). Reproduces the
                                 first entry of every "success per seed" field, NOT the pooled columns.
      python experiment.py --only close+end,pbrs [--seeds 0,1,2]
                                 selected cells only; with all three seeds each line matches output.txt.
                                 Tags: sparse pbrs close+end dense+end close w=0.0 w=0.03 w=0.1 w=0.3
                                 w=1.0 w=3.0 (dense = w=0.0)
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import mujoco
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "32-rl-environment"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "33-ppo"))
from ppo import PPOConfig, evaluate, train, wilson  # noqa: E402
from reach_env import Q_HOME, ReachEnv  # noqa: E402

BASE = dict(total_steps=300_000, lr=1e-3, gamma=0.95, lam=0.95, log_std_init=-1.0)
GAMMA = BASE["gamma"]
SEEDS = (0, 1, 2)
N_EVAL = 100
WEIGHTS = (0.0, 0.03, 0.1, 0.3, 1.0, 3.0)


# ---------------------------------------------------------------- rewards (info comes from ReachEnv.step)
def r_dense(info):
    return -info["dist"]


def r_sparse(info):
    return float(info["dist"] < 0.02)


def r_pbrs(info):
    phi, phi_prev = -20.0 * info["dist"], -20.0 * info["dist_prev"]
    return r_sparse(info) + GAMMA * phi - phi_prev


def r_closeness(info):
    return 1.0 - np.tanh(info["dist"] / 0.05)


def r_action_rate(w):
    return lambda info: -info["dist"] - w * float(np.sum((info["action"] - info["prev_action"]) ** 2))


def env_kwargs(tag):
    if tag.startswith("w="):
        return {"reward_fn": r_action_rate(float(tag[2:]))}
    return {"dense": {"reward_fn": r_dense},
            "sparse": {"reward_fn": r_sparse},
            "pbrs": {"reward_fn": r_pbrs},
            "close+end": {"reward_fn": r_closeness, "terminate_on_success": True},
            "dense+end": {"reward_fn": r_dense, "terminate_on_success": True},
            "close": {"reward_fn": r_closeness}}[tag]


def job(args):
    tag, seed = args
    kw = env_kwargs(tag)
    pi, log = train(lambda i: ReachEnv(**kw), PPOConfig(seed=seed, **BASE))
    ev = evaluate(lambda i: ReachEnv(record=True, **kw), pi, N_EVAL)
    ev["d_noterm"] = evaluate(lambda i: ReachEnv(reward_fn=kw["reward_fn"]), pi, N_EVAL)["final_d"]  # same policy, no termination
    return tag, seed, ev, log["seconds"]


def scripted_return(kw, n=N_EVAL):
    env = ReachEnv(**kw)
    rets = []
    for e in range(n):
        env.reset(seed=10_000 + e)
        R, done = 0.0, False
        while not done:
            m, d = env.model, env.data
            jacp = np.zeros((3, m.nv))
            mujoco.mj_jacSite(m, d, jacp, None, env.site)
            J = jacp[:, env.vadr]
            err = env.target - d.site_xpos[env.site]
            dq = J.T @ np.linalg.solve(J @ J.T + 0.05 ** 2 * np.eye(3), err)
            _, r, term, trunc, info = env.step(np.clip((d.qpos[env.qadr] + dq - Q_HOME) / env.scale, -1, 1))
            R += r
            done = term or trunc
        rets.append(R)
    return float(np.mean(rets))


def summarize(rows):
    ks = [ev["k"] for _, ev, _ in rows]
    K, n = sum(ks), N_EVAL * len(rows)
    lo, hi = wilson(K, n)
    return {"ks": "/".join(str(k) for k in ks), "pooled": f"{K:>3}/{n} [{100*lo:4.1f}, {100*hi:5.1f}]",
            "ret": np.mean([ev["ret"] for _, ev, _ in rows]),
            "d": np.mean([ev["final_d"] for _, ev, _ in rows]) * 1000,
            "len": np.mean([ev["len"] for _, ev, _ in rows]),
            "tr": np.mean([ev["tau_rate"] for _, ev, _ in rows]),
            "pw": np.mean([ev["power"] for _, ev, _ in rows]),
            "d_noterm": np.mean([ev["d_noterm"] for _, ev, _ in rows]) * 1000}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="seed 0 only; does not reproduce the pooled table")
    ap.add_argument("--only", default="", help="comma-separated cell tags, e.g. close+end,pbrs")
    ap.add_argument("--seeds", default="0,1,2", help="seeds for --only (default 0,1,2)")
    args = ap.parse_args()
    seeds = (0,) if args.quick else tuple(int(x) for x in args.seeds.split(","))
    print(f"mujoco {mujoco.__version__} | numpy {np.__version__} | torch {torch.__version__} | "
          f"Apple M5, CPU, 1 torch thread per run")
    if args.quick:
        print("QUICK MODE: seed 0 only; pooled columns and means do NOT reproduce output.txt")
    print(f"PPO: {BASE} (+ lesson-33 defaults), {len(seeds)} seed{'s' if len(seeds) > 1 else ''} x {N_EVAL} "
          f"evaluation episodes per cell")
    t_start = time.perf_counter()
    tags = ["sparse", "pbrs", "close+end", "dense+end", "close"] + [f"w={w}" for w in WEIGHTS]
    only = [t for t in args.only.split(",") if t]
    if only:
        if set(only) - set(tags):
            raise SystemExit(f"unknown tag(s) {sorted(set(only) - set(tags))}")
        tags = only
    jobs = [(t, s) for t in tags for s in SEEDS if s in seeds]
    t0 = time.perf_counter()
    with Pool(min(6, os.cpu_count(), len(jobs))) as p:
        res = p.map(job, jobs)
    wall = time.perf_counter() - t0
    by = {}
    for tag, seed, ev, sec in res:
        by.setdefault(tag, []).append((seed, ev, sec))
    if only:
        print(f"\nSelected cells, seeds {'/'.join(map(str, seeds))}")
        print(f"  {'reward':<12}{'success per seed':<18}{'pooled success, Wilson 95%':<28}{'written return':>15}"
              f"{'ep. length':>11}{'median final d mm':>18}{'tau rate N m/s':>16}{'power W':>9}{'d no-term mm':>13}")
        for t in only:
            s = summarize(by[t])
            print(f"  {t:<12}{s['ks']:<18}{s['pooled']:<28}{s['ret']:>15.2f}{s['len']:>11.1f}{s['d']:>18.1f}"
                  f"{s['tr']:>16.0f}{s['pw']:>9.2f}{s['d_noterm']:>13.1f}")
        print(f"\n{len(jobs)} training runs, {wall:.0f} s wall-clock in total (shared machine)")
        return
    by["dense"] = by["w=0.0"]

    print("\n[A] Reward shape (no termination; success = within 2 cm at the 1.6 s time limit)")
    print(f"  {'reward':<30}{'success per seed':<18}{'pooled success, Wilson 95%':<28}{'median final d mm':>18}")
    for tag, label in [("dense", "dense  -d"), ("sparse", "sparse 1[d < 2 cm]"),
                       ("pbrs", "sparse + 0.95*Phi(s') - Phi(s)")]:
        s = summarize(by[tag])
        print(f"  {label:<30}{s['ks']:<18}{s['pooled']:<28}{s['d']:>18.1f}")

    print("\n[B] Reward hacking: episode ends when d < 2 cm (success = episode ended by reaching)")
    print(f"  {'reward':<30}{'success per seed':<18}{'pooled success, Wilson 95%':<28}{'written return':>15}"
          f"{'ep. length':>11}{'median final d mm':>18}")
    for tag, label in [("close+end", "1 - tanh(d/5cm), ends on success"), ("dense+end", "-d, ends on success"),
                       ("close", "1 - tanh(d/5cm), no termination")]:
        s = summarize(by[tag])
        print(f"  {label:<30}{s['ks']:<18}{s['pooled']:<28}{s['ret']:>15.2f}{s['len']:>11.1f}{s['d']:>18.1f}")
    s = summarize(by["close+end"])
    print(f"  close+end policy run WITHOUT termination: median final distance {s['d_noterm']:.1f} mm")
    kw = env_kwargs("close+end")
    print(f"  scripted IK (reaches every target) scored by the same written reward: return "
          f"{scripted_return(kw):.2f}, ends on success")

    print("\n[C] Action-rate penalty r = -d - w ||a_t - a_(t-1)||^2 (no termination)")
    print(f"  {'w':>6}  {'success per seed':<18}{'pooled success, Wilson 95%':<28}{'median final d mm':>18}"
          f"{'tau rate N m/s':>16}{'power W':>9}")
    for w in WEIGHTS:
        s = summarize(by[f"w={w}"])
        print(f"  {w:>6}  {s['ks']:<18}{s['pooled']:<28}{s['d']:>18.1f}{s['tr']:>16.0f}{s['pw']:>9.2f}")
    print(f"\n{len(jobs)} training runs, {wall:.0f} s wall-clock in total (6 worker processes, shared machine)")
    if args.quick:
        print(f"total wall-clock {time.perf_counter() - t_start:.0f} s")


if __name__ == "__main__":
    main()
