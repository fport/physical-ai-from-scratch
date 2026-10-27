"""Lesson 34 · Observation and action design.

Same PPO (lesson 33, reference config), same SO-101 reach task, same 300k-step budget; only the
interface between policy and robot changes.
  A. Action parameterization at 50 Hz: absolute joint target (servo PD), delta target, raw torque.
  B. Policy rate (decimation): frame_skip 1..16 with absolute targets; gamma and lambda are set from
     a fixed 0.4 s horizon, gamma = lambda = 1 - dt / 0.4 s, so only the rate changes.
  C. Observation content: full / without joint velocities / absolute goal instead of goal - gripper.
  D. Break it: action scale 0.25 rad (the Go2 training value) on an arm whose targets need up to 1 rad.
Each cell: 3 seeds x 100 evaluation episodes (fixed seeds, deterministic policy). Smoothness and
energy are measured at the physics rate (5 ms) during evaluation:
  torque rate = RMS over steps of ||d tau / dt|| (N m/s), power = mean sum_i |tau_i * qdot_i| (W).

Run:  python experiment.py      (30 runs; 1190 s wall-clock on a heavily loaded Apple M5; the script
                                 prints its own wall-clock time at the end)
      python experiment.py --quick
                                 seed 0 only (10 runs, roughly a third of the time). Reproduces the
                                 first entry of every "success per seed" field, NOT the pooled columns.
      python experiment.py --only A:torque,D:0.25 [--seeds 0,1,2]
                                 selected cells only; with all three seeds each line matches output.txt
                                 (except the train-s timing column). Tags: A:position A:delta A:torque
                                 B:1 B:2 B:8 B:16 C:no_vel C:abs_goal D:0.25
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
SEEDS = (0, 1, 2)
N_EVAL = 100
HORIZON_S = 0.4


def job(args):
    tag, env_kw, ppo_kw, seed = args
    cfg = PPOConfig(seed=seed, **{**BASE, **ppo_kw})
    pi, log = train(lambda i: ReachEnv(**env_kw), cfg)
    ev = evaluate(lambda i: ReachEnv(record=True, **env_kw), pi, N_EVAL)
    return tag, seed, ev, log["seconds"]


def scripted_ik_success(scale, n=N_EVAL):
    """Upper bound for a given action scale: damped least-squares IK through the same clipped interface."""
    env = ReachEnv(action_scale=scale)
    k = 0
    for e in range(n):
        env.reset(seed=10_000 + e)
        done = False
        while not done:
            m, d = env.model, env.data
            jacp = np.zeros((3, m.nv))
            mujoco.mj_jacSite(m, d, jacp, None, env.site)
            J = jacp[:, env.vadr]
            err = env.target - d.site_xpos[env.site]
            dq = J.T @ np.linalg.solve(J @ J.T + 0.05 ** 2 * np.eye(3), err)
            a = np.clip((d.qpos[env.qadr] + dq - Q_HOME) / env.scale, -1, 1)
            _, _, term, trunc, info = env.step(a)
            done = term or trunc
        k += int(info["is_success"])
    return k


def summarize(rows):
    ks = [ev["k"] for _, ev, _ in rows]
    K, n = sum(ks), N_EVAL * len(rows)
    lo, hi = wilson(K, n)
    return {"ks": "/".join(str(k) for k in ks), "pooled": f"{K:>3}/{n} [{100*lo:4.1f}, {100*hi:5.1f}]",
            "d": np.mean([ev["final_d"] for _, ev, _ in rows]) * 1000,
            "tr": np.mean([ev["tau_rate"] for _, ev, _ in rows]),
            "pw": np.mean([ev["power"] for _, ev, _ in rows]),
            "sec": np.mean([s for _, _, s in rows])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="seed 0 only; does not reproduce the pooled table")
    ap.add_argument("--only", default="", help="comma-separated cell tags, e.g. A:torque,D:0.25")
    ap.add_argument("--seeds", default="0,1,2", help="seeds for --only (default 0,1,2)")
    args = ap.parse_args()
    seeds = (0,) if args.quick else tuple(int(x) for x in args.seeds.split(","))
    print(f"mujoco {mujoco.__version__} | numpy {np.__version__} | torch {torch.__version__} | "
          f"Apple M5, CPU, 1 torch thread per run")
    if args.quick:
        print("QUICK MODE: seed 0 only; pooled columns and means do NOT reproduce output.txt")
    print(f"PPO: {BASE} (+ lesson-33 defaults), {len(seeds)} seed{'s' if len(seeds) > 1 else ''} x {N_EVAL} evaluation episodes per cell")
    t_start = time.perf_counter()
    jobs = []
    for mode in ("position", "delta", "torque"):
        jobs += [(f"A:{mode}", {"action_mode": mode}, {}, s) for s in SEEDS]
    for fs in (1, 2, 8, 16):
        g = 1 - 0.005 * fs / HORIZON_S
        jobs += [(f"B:{fs}", {"frame_skip": fs}, {"gamma": g, "lam": g}, s) for s in SEEDS]
    for om in ("no_vel", "abs_goal"):
        jobs += [(f"C:{om}", {"obs_mode": om}, {}, s) for s in SEEDS]
    jobs += [("D:0.25", {"action_scale": 0.25}, {}, s) for s in SEEDS]
    jobs = [j for j in jobs if j[3] in seeds]
    only = [t for t in args.only.split(",") if t]
    if only:
        unknown = set(only) - {j[0] for j in jobs}
        if unknown:
            raise SystemExit(f"unknown tag(s) {sorted(unknown)}")
        jobs = [j for j in jobs if j[0] in only]
    t0 = time.perf_counter()
    with Pool(min(6, os.cpu_count(), len(jobs))) as p:
        res = p.map(job, jobs)
    wall = time.perf_counter() - t0
    by = {}
    for tag, seed, ev, sec in res:
        by.setdefault(tag, []).append((seed, ev, sec))

    hdr = f"{'success per seed':<18}{'pooled success, Wilson 95%':<28}{'final d mm':>11}{'tau rate N m/s':>16}{'power W':>9}{'train s':>9}"

    def line(label, s):
        return f"  {label:<26}{s['ks']:<18}{s['pooled']:<28}{s['d']:>11.1f}{s['tr']:>16.0f}{s['pw']:>9.2f}{s['sec']:>9.0f}"

    if only:
        print(f"\nSelected cells, seeds {'/'.join(map(str, seeds))}")
        print(f"  {'cell':<26}" + hdr)
        for t in only:
            print(line(t, summarize(by[t])))
        print(f"\n{len(jobs)} training runs, {wall:.0f} s wall-clock in total (shared machine)")
        return

    print("\n[A] Action parameterization (50 Hz policy, frame_skip 4)")
    print(f"  {'action':<26}" + hdr)
    labels = {"position": "abs target q0+1.0*a (PD)", "delta": "delta target +0.1*a (PD)", "torque": "torque 2.94*a N m"}
    for mode in ("position", "delta", "torque"):
        print(line(labels[mode], summarize(by[f"A:{mode}"])))

    print(f"\n[B] Policy rate, absolute targets, gamma = lambda = 1 - dt/{HORIZON_S} s, same 300k policy steps")
    print(f"  {'frame_skip / Hz / gamma':<26}" + hdr)
    for fs in (1, 2, 4, 8, 16):
        rows = by["A:position"] if fs == 4 else by[f"B:{fs}"]
        g = 1 - 0.005 * fs / HORIZON_S
        print(line(f"{fs:>2} / {200/fs:5.1f} / {g:.4f}", summarize(rows)))

    print("\n[C] Observation content (absolute targets, 50 Hz)")
    print(f"  {'observation':<26}" + hdr)
    print(line("full (q, qdot, goal-ee, a)", summarize(by["A:position"])))
    print(line("no joint velocities", summarize(by["C:no_vel"])))
    print(line("absolute goal, not goal-ee", summarize(by["C:abs_goal"])))

    print("\n[D] Break it: action scale 0.25 rad (absolute targets, 50 Hz)")
    print(f"  {'scale':<26}" + hdr)
    print(line("1.0 rad", summarize(by["A:position"])))
    print(line("0.25 rad", summarize(by["D:0.25"])))
    for sc in (1.0, 0.25):
        print(f"  scripted IK through the same interface, scale {sc}: {scripted_ik_success(sc)}/{N_EVAL}")
    print(f"\n{len(jobs)} training runs, {wall:.0f} s wall-clock in total (6 worker processes, shared machine)")
    if args.quick:
        print(f"total wall-clock {time.perf_counter() - t_start:.0f} s")


if __name__ == "__main__":
    main()
