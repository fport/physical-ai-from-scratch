"""Lesson 36 · Legged locomotion and massively parallel training (CPU part).

The Go2 (MuJoCo Menagerie `unitree_go2`, 2 ms timestep) stands with the training PD rule
(Kp 20, Kd 0.5 around the home joint angles, torque capped at the motor limits), modelled as MuJoCo
position actuators so open-loop rollouts can run in C++.
  A. Contact basics: feet in contact, sum of contact normal forces vs m g, base height.
  B. Check: mujoco.rollout reproduces the Python mj_step loop exactly.
  C. Sweep the number of parallel environments (nbatch) x threads (nthread) with mujoco.rollout
     -> total physics steps/s. Baseline: a Python loop over one MjData.
  D. Break it: the same batch started from a settled stance vs released from the keyframe pose.
  E. Arithmetic: physics steps in one run of the Go2 training config, divided by the best rate in C.
Timing depends on machine load (other jobs were running on this 10-core Apple M5: 4 performance + 6
efficiency cores), so the throughput numbers are approximate; each cell is the best of several timings.

Run:  python experiment.py           (prints its own wall-clock time at the end; a few minutes on a loaded Apple M5)
      python experiment.py --quick   (1 timing round, no nbatch 4096: a smoke test; its rates are noisier
                                      and do NOT reproduce output.txt)
"""
import argparse
import os
import sys
import time

import mujoco
import numpy as np
from mujoco import rollout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.menagerie import path  # noqa: E402

KP, KD = 20.0, 0.5
REPEATS = 3
ROUNDS = 5
# default_joint_angles of the Go2 training config (go2_config.py), Menagerie order FL, FR, RL, RR x (hip, thigh, calf)
Q_STAND = np.array([0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5])


def load_go2():
    m = mujoco.MjModel.from_xml_path(path("unitree_go2"))
    for i in range(m.nu):                       # motor -> PD position servo, keep the torque limit
        tau_max = m.actuator_ctrlrange[i, 1]
        j = m.actuator_trnid[i, 0]
        m.actuator_gainprm[i, :3] = [KP, 0, 0]
        m.actuator_biasprm[i, :3] = [0, -KP, -KD]
        m.actuator_biastype[i] = mujoco.mjtBias.mjBIAS_AFFINE
        m.actuator_ctrlrange[i] = m.jnt_range[j]
        m.actuator_forcelimited[i] = 1
        m.actuator_forcerange[i] = [-tau_max, tau_max]
    return m


def main():
    global REPEATS, ROUNDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="1 timing round, no nbatch 4096 (not the table)")
    quick = ap.parse_args().quick
    if quick:
        REPEATS, ROUNDS = 1, 1
        print("QUICK MODE: 1 timing round, no nbatch 4096; does not reproduce output.txt")
    t_start = time.perf_counter()
    m = load_go2()
    d = mujoco.MjData(m)
    key = m.key("home").id
    q_home = Q_STAND

    def reset(data):
        mujoco.mj_resetDataKeyframe(m, data, key)          # base pose from the Menagerie keyframe
        data.qpos[7:] = Q_STAND
        data.ctrl[:] = Q_STAND
    reset(d)
    print(f"mujoco {mujoco.__version__} | numpy {np.__version__} | Apple M5 (4P + 6E cores), "
          f"os.cpu_count() = {os.cpu_count()} | load average at start {os.getloadavg()[0]:.0f}")
    print(f"Go2: nq {m.nq}, nv {m.nv}, nu {m.nu}, timestep {m.opt.timestep*1000:.0f} ms, "
          f"cone elliptic, impratio {m.opt.impratio:.0f}, PD Kp {KP} Kd {KD}")

    # ------------------------------------------------------------ A. contact basics
    d.ctrl[:] = q_home
    for _ in range(1000):                       # 2 s of standing
        mujoco.mj_step(m, d)
    floor = m.geom("floor").id
    feet = {m.geom(n).id: n for n in ("FL", "FR", "RL", "RR")}
    fz, in_contact, f6 = 0.0, set(), np.zeros(6)
    for c in range(d.ncon):
        con = d.contact[c]
        if floor in (con.geom1, con.geom2):
            mujoco.mj_contactForce(m, d, c, f6)
            fz += f6[0]                          # normal component in the contact frame
            other = con.geom2 if con.geom1 == floor else con.geom1
            in_contact.add(feet.get(other, m.geom(other).name or str(other)))
    nstate = mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    x_stand = np.zeros(nstate)                   # settled standing state, start of every rollout in C
    mujoco.mj_getState(m, d, x_stand, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    mass = m.body_subtreemass[1]
    print("\n[A] Standing 2 s with PD 20 / 0.5 around the training default joint angles")
    print(f"  base height {d.qpos[2]*100:.1f} cm | floor contacts {d.ncon} on {sorted(in_contact)} | "
          f"sum of normal forces {fz:.1f} N vs m g = {mass:.2f} kg x 9.81 = {mass*9.81:.1f} N")

    # ------------------------------------------------------------ B. rollout == Python loop
    nstate = mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    reset(d)
    x0 = np.zeros(nstate)
    mujoco.mj_getState(m, d, x0, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    ctrl = np.tile(q_home, (1, 500, 1))
    st, _ = rollout.rollout(m, mujoco.MjData(m), x0[None], ctrl)
    d2 = mujoco.MjData(m)
    mujoco.mj_setState(m, d2, x0, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    d2.ctrl[:] = q_home
    for _ in range(500):
        mujoco.mj_step(m, d2)
    x1 = np.zeros(nstate)
    mujoco.mj_getState(m, d2, x1, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    print(f"\n[B] rollout vs Python loop after 500 steps: max |state difference| = {np.max(np.abs(st[0, -1] - x1)):.1e}")

    # ------------------------------------------------------------ C. throughput sweep
    rng = np.random.default_rng(0)

    def init_states(n):
        """Settled stance + small random joint velocities, so every robot and every step costs about the same."""
        X = np.tile(x_stand, (n, 1))
        X[:, 1 + m.nq + 6:1 + m.nq + m.nv] += rng.uniform(-0.05, 0.05, size=(n, m.nv - 6))  # [time, qpos, qvel, ...]
        return X

    print(f"\n[C] Total physics steps/s = nbatch x nstep / wall time, Go2 standing (from the settled stance of A),\n"
          f"    ~5e4 steps per call, best of {ROUNDS} round-robin rounds")
    d3 = mujoco.MjData(m)
    mujoco.mj_setState(m, d3, x_stand, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    d3.ctrl[:] = q_home
    base = 0.0
    for _ in range(REPEATS):                     # best of REPEATS 1-second windows
        n, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < 1.0:
            for _ in range(500):
                mujoco.mj_step(m, d3)
            n += 500
        base = max(base, n / (time.perf_counter() - t0))
    print(f"  Python loop, 1 MjData, 1 thread: {base:,.0f} steps/s")
    threads, batches = (1, 2, 4, 8), ((1, 8, 64, 512) if quick else (1, 8, 64, 512, 4096))
    cells = {}
    for nb in batches:
        nstep = max(25, 50_000 // nb)
        X, U = init_states(nb), np.tile(q_home, (nb, max(25, 50_000 // nb), 1))
        for nt in threads:
            cells[nb, nt] = (X, U, [mujoco.MjData(m) for _ in range(nt)], [])
    for _ in range(ROUNDS):                      # round-robin, so a load burst hits different cells each round
        for (nb, nt), (X, U, datas, el) in cells.items():
            t0, c0 = time.perf_counter(), time.process_time()
            rollout.rollout(m, datas, X, U)
            el.append((time.perf_counter() - t0, time.process_time() - c0))
    print("  wall-clock steps/s:")
    print(f"  {'nbatch':>7}" + "".join(f"{'nthread ' + str(t):>14}" for t in threads))
    best = (0, None)
    for nb in batches:
        row = []
        for nt in threads:
            X, U, _, el = cells[nb, nt]
            rate = U.shape[0] * U.shape[1] / min(w for w, _ in el)
            row.append(rate)
            if rate > best[0]:
                best = (rate, (nb, nt))
        print(f"  {nb:>7}" + "".join(f"{r:>14,.0f}" for r in row))
    print("  steps per CPU-second (process CPU time summed over all threads; flat = no threading overhead):")
    print(f"  {'nbatch':>7}" + "".join(f"{'nthread ' + str(t):>14}" for t in threads))
    for nb in batches:
        print(f"  {nb:>7}" + "".join(
            f"{cells[nb, nt][1].shape[0] * cells[nb, nt][1].shape[1] / min(c for _, c in cells[nb, nt][3]):>14,.0f}"
            for nt in threads))
    print(f"  best: {best[0]:,.0f} steps/s at nbatch {best[1][0]}, nthread {best[1][1]} "
          f"= {best[0]/base:.1f}x the Python loop")

    # ------------------------------------------------------------ D. break it: what the robots are doing
    def init_drop(n):
        """Keyframe base pose with joints perturbed by +-0.1 rad: every robot starts by settling onto its feet."""
        X = np.tile(x0, (n, 1))
        X[:, 1 + 7:1 + m.nq] += rng.uniform(-0.1, 0.1, size=(n, m.nq - 7))
        return X

    print(f"\n[D] Break it. Same rollouts, different start: 512 robots x 25 steps (50 ms), nthread 8, best of {REPEATS}")
    print(f"  {'start':<44}{'steps/s':>10}{'solver iters/step':>19}{'contacts':>10}")
    for label, X in (("settled stance (+ small joint velocities)", init_states(512)),
                     ("released from keyframe pose, joints +-0.1 rad", init_drop(512))):
        U = np.tile(q_home, (512, 25, 1))
        datas = [mujoco.MjData(m) for _ in range(8)]
        el = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            rollout.rollout(m, datas, X, U)
            el.append(time.perf_counter() - t0)
        it, nc, de = [], [], mujoco.MjData(m)          # solver statistics from a Python replay of 32 robots
        for i in range(32):
            mujoco.mj_setState(m, de, X[i], mujoco.mjtState.mjSTATE_FULLPHYSICS)
            de.ctrl[:] = q_home
            for _ in range(25):
                mujoco.mj_step(m, de)
                it.append(de.solver_niter[0])
                nc.append(de.ncon)
        print(f"  {label:<44}{512*25/min(el):>10,.0f}{np.mean(it):>19.1f}{np.mean(nc):>10.1f}")

    # ------------------------------------------------------------ E. arithmetic
    n_env, n_steps, iters, dec = 4096, 24, 1500, 4
    total = n_env * n_steps * iters * dec
    print("\n[E] Go2 training config (num_envs 4096, num_steps_per_env 24, max_iterations 1500, decimation 4)")
    print(f"  policy steps {n_env*n_steps*iters:,} -> physics steps {total:,}")
    print(f"  load average at end {os.getloadavg()[0]:.0f} (other jobs share the 10 cores)")
    print(f"  at the best rate in C: {total/best[0]/3600:.1f} h of physics alone on this CPU "
          f"(no policy, no learning; our 2 ms step vs 5 ms in that config)")
    print(f"\ntotal wall-clock {time.perf_counter() - t_start:.0f} s (shared machine; approximate)")


if __name__ == "__main__":
    main()
