"""Lesson 32 · A MuJoCo RL environment.

  A. API checks: spaces, seeding (same seed -> same target), terminated vs truncated counts.
  B. Baselines on the SO-101 reach task, 200 episodes each with fixed seeds:
     random policy, zero action (hold the home pose), scripted IK controller.
     -> mean return, median final distance, success rate with a Wilson 95% interval.
  C. Sweep frame_skip (physics steps per policy step) -> env steps per CPU second and simulated seconds per
     CPU second, random actions, one env, one thread (process CPU time, approximate).
  D. Break it: shorten the time limit (truncation) for the scripted controller -> success.

Run:  python experiment.py           (prints its own wall-clock time at the end; about 1-2 min on a loaded Apple M5)
      python experiment.py --quick   (50 episodes per row, 1 s per throughput row: a smoke test that
                                      does NOT reproduce output.txt)
"""
import argparse
import time

import mujoco
import numpy as np
import torch
import gymnasium

from reach_env import Q_HOME, ReachEnv, wilson

N_EPISODES = 200
CPU_SECONDS = 3.0


def run_policy(env, policy, n=None, seed0=0):
    n = N_EPISODES if n is None else n
    rets, final_d, succ, n_term, n_trunc = [], [], 0, 0, 0
    for k in range(n):
        obs, _ = env.reset(seed=seed0 + k)
        R, done = 0.0, False
        while not done:
            obs, r, term, trunc, info = env.step(policy(env, obs))
            R += r
            done = term or trunc
        n_term += term
        n_trunc += trunc
        rets.append(R)
        final_d.append(info["dist"])
        succ += int(info["is_success"])
    return np.mean(rets), np.std(rets), np.median(final_d), succ, n_term, n_trunc


def random_policy(env, obs):
    return env.action_space.sample()


def zero_policy(env, obs):
    return np.zeros(4)


def scripted_ik(env, obs, gain=1.0, damping=0.05):
    """Damped least-squares IK step toward the target, sent as an absolute joint target."""
    m, d = env.model, env.data
    jacp = np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, jacp, None, env.site)
    J = jacp[:, env.vadr]
    e = env.target - d.site_xpos[env.site]
    dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(3), gain * e)
    q_target = d.qpos[env.qadr] + dq
    return np.clip((q_target - Q_HOME) / env.scale, -1, 1)


def main():
    global N_EPISODES, CPU_SECONDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="50 episodes, 1 s per throughput row (not the table)")
    if ap.parse_args().quick:
        N_EPISODES, CPU_SECONDS = 50, 1.0
        print("QUICK MODE: 50 episodes per row, 1 s per throughput row; does not reproduce output.txt")
    t_start = time.perf_counter()
    print(f"mujoco {mujoco.__version__} | numpy {np.__version__} | gymnasium {gymnasium.__version__} | "
          f"torch {torch.__version__} | Apple M5, CPU, 1 thread")

    env = ReachEnv()
    print("\n[A] Environment")
    print(f"  model timestep {env.model.opt.timestep*1000:.0f} ms, frame_skip {env.frame_skip} -> policy period "
          f"{env.dt*1000:.0f} ms ({1/env.dt:.0f} Hz), {env.max_steps} steps per episode ({env.max_steps*env.dt:.1f} s)")
    print(f"  observation_space {env.observation_space.shape}, action_space {env.action_space.shape} "
          f"in [{env.action_space.low[0]:.0f}, {env.action_space.high[0]:.0f}]")
    t1 = env.reset(seed=7)[1]["dist"], env.target.copy()
    env.reset(seed=8)
    t2 = env.reset(seed=7)[1]["dist"], env.target.copy()
    print(f"  reset(seed=7) twice -> same target: {np.allclose(t1[1], t2[1])} "
          f"(target {np.round(t1[1], 3)} m, start distance {t1[0]:.3f} m)")
    term_env = ReachEnv(terminate_on_success=True)
    _, _, _, s, nt, ntr = run_policy(term_env, scripted_ik, n=50)
    print(f"  terminate_on_success=True, scripted IK, 50 episodes: terminated {nt}, truncated {ntr}")

    print(f"\n[B] Baselines, {N_EPISODES} episodes each, success = gripper within 2 cm of target at the time limit")
    print(f"  {'policy':<14}{'return':>16}{'median final d':>16}{'success':>10}{'Wilson 95%':>16}")
    for name, pol in [("random", random_policy), ("zero action", zero_policy), ("scripted IK", scripted_ik)]:
        env = ReachEnv()
        env.action_space.seed(0)
        mr, sr, fd, k, _, _ = run_policy(env, pol)
        lo, hi = wilson(k, N_EPISODES)
        print(f"  {name:<14}{mr:>9.2f} ± {sr:<5.2f}{fd*1000:>13.1f} mm{k:>6}/{N_EPISODES}"
              f"{lo*100:>8.1f}-{hi*100:.1f}%")

    print(f"\n[C] Throughput vs frame_skip (random actions, 1 env, 1 thread, {CPU_SECONDS:.0f} s of process CPU time per row;\n"
          "    CPU time, not wall-clock, so other load on the machine distorts it less)")
    print(f"  {'frame_skip':>10}{'policy Hz':>11}{'env steps/CPU s':>17}{'physics steps/CPU s':>21}{'sim s / CPU s':>16}")
    fss, us = (1, 2, 4, 8, 16, 32), []
    for fs in fss:
        env = ReachEnv(frame_skip=fs, episode_seconds=1e9)
        env.reset(seed=0)
        env.action_space.seed(0)
        n, t0 = 0, time.process_time()
        while time.process_time() - t0 < CPU_SECONDS:
            for _ in range(100):
                env.step(env.action_space.sample())
            n += 100
        el = time.process_time() - t0
        us.append(el / n * 1e6)
        print(f"  {fs:>10}{1/env.dt:>11.1f}{n/el:>17.0f}{n*fs/el:>21.0f}{n*env.dt/el:>16.1f}")
    b, a = np.polyfit(fss, us, 1)
    print(f"  linear fit, CPU time per env.step = {a:.1f} us + {b:.1f} us x frame_skip "
          f"(fixed Python/env overhead + physics)")

    print(f"\n[D] Break it: time limit (truncation) vs scripted IK success, {N_EPISODES} episodes, frame_skip 4")
    print(f"  {'time limit':>10}{'steps':>7}{'success':>10}{'Wilson 95%':>16}{'median final d':>16}")
    for T in (0.08, 0.16, 0.24, 0.32, 0.48, 1.6):
        env = ReachEnv(episode_seconds=T)
        _, _, fd, k, _, _ = run_policy(env, scripted_ik)
        lo, hi = wilson(k, N_EPISODES)
        print(f"  {T:>8.2f} s{env.max_steps:>7}{k:>6}/{N_EPISODES}{lo*100:>8.1f}-{hi*100:.1f}%{fd*1000:>13.1f} mm")
    print(f"\ntotal wall-clock {time.perf_counter() - t_start:.0f} s (shared machine; approximate)")


if __name__ == "__main__":
    main()
