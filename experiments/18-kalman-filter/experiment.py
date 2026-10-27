"""Lesson 18 · The Kalman filter.

Same signal as lesson 17: the SO-101 shoulder_pan joint in MuJoCo, read through a
12-bit encoder with 10 mrad white noise at 500 Hz (same seed -> same noise).
A linear Kalman filter with a constant-velocity model estimates position AND
velocity from that one noisy reading.

Sweep: process-noise intensity q (rad^2/s^3).  Measured: position / velocity RMS
error, delay, and whether the filter's own uncertainty is honest (2-sigma coverage).
Break it: wrong measurement noise R, and an unmodeled acceleration (a step command).

Run:  python experiment.py
"""
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.menagerie import path  # noqa: E402

DT = 0.002
T_END = 10.0
COUNTS = 4096
SIGMA = 0.01
SEED = 0
K0 = 500                      # samples skipped before scoring (start-up transient)


# ---------- signal (identical to lesson 17) ----------
def reference(t):
    return 0.5 * np.sin(2 * np.pi * 0.2 * t) + 0.2 * np.sin(2 * np.pi * 0.5 * t)


def simulate(ref_fn):
    spec = mujoco.MjSpec.from_file(path("robotstudio_so101"))
    spec.option.timestep = DT
    m = spec.compile()
    d = mujoco.MjData(m)
    n = int(round(T_END / DT))
    q, v = np.zeros(n), np.zeros(n)
    ctrl = np.array([0.0, -0.5, 0.8, 0.6, 0.0, 0.0])
    d.qpos[:6] = ctrl
    mujoco.mj_forward(m, d)
    for i in range(n):
        ctrl[0] = ref_fn(d.time)
        d.ctrl[:] = ctrl
        q[i], v[i] = d.qpos[0], d.qvel[0]
        mujoco.mj_step(m, d)
    return q, v


def encoder(q, rng):
    step = 2 * np.pi / COUNTS
    return np.round(q / step) * step + rng.normal(0, SIGMA, q.shape)


# ---------- lesson 17 baselines ----------
def moving_average(x, n):
    c = np.cumsum(np.insert(x, 0, 0.0))
    idx = np.arange(len(x))
    lo = np.maximum(0, idx + 1 - n)
    return (c[idx + 1] - c[lo]) / (idx + 1 - lo)


# ---------- Kalman filter ----------
def kalman_cv(z, q_psd, r):
    """Constant-velocity KF. State x = [q, qdot]. Returns estimates and position std."""
    F = np.array([[1.0, DT], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = q_psd * np.array([[DT**3 / 3, DT**2 / 2], [DT**2 / 2, DT]])
    x = np.array([z[0], 0.0])
    P = np.diag([r, 1.0])
    xs, sd = np.zeros((len(z), 2)), np.zeros(len(z))
    for k in range(len(z)):
        # predict
        x = F @ x
        P = F @ P @ F.T + Q
        # update
        S = H @ P @ H.T + r
        K = (P @ H.T) / S
        x = x + (K * (z[k] - H @ x)).ravel()
        P = (np.eye(2) - K @ H) @ P
        xs[k], sd[k] = x, np.sqrt(P[0, 0])
    return xs, sd, K.ravel()


def delay_ms(y, ref, max_lag=400):
    errs = [np.mean((y[K0 + L:] - ref[K0:len(ref) - L]) ** 2) for L in range(max_lag)]
    return 1000 * DT * int(np.argmin(errs))


def rms(e):
    return np.sqrt(np.mean(e[K0:] ** 2))


def main():
    print(f"mujoco {mujoco.__version__} · numpy {np.__version__} · seed {SEED} · dt {DT*1000:.0f} ms")
    q, v = simulate(reference)
    z = encoder(q, np.random.default_rng(SEED))
    step = 2 * np.pi / COUNTS
    R = SIGMA**2 + step**2 / 12             # white noise + quantization variance
    print(f"measurement noise R = {R:.3e} rad^2 (sigma {1000*np.sqrt(R):.2f} mrad)")

    print("\n[1] lesson 17 baselines on the same data")
    ma5, ma65 = moving_average(z, 5), moving_average(z, 65)
    print(f"    MA N=5  : pos RMS {1000*rms(ma5-q):.2f} mrad, delay {delay_ms(ma5, q):.1f} ms")
    print(f"    MA N=65 : vel RMS {rms(np.gradient(ma65, DT)-v):.3f} rad/s (best velocity in lesson 17)")

    print("\n[2] sweep process noise q (constant-velocity model)")
    print(f"{'q':>8} {'pos RMS mrad':>13} {'delay ms':>9} {'vel RMS rad/s':>14} {'K_pos':>7} {'K_vel':>7} "
          f"{'2sigma cover %':>15}")
    best = None
    for qp in [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]:
        xs, sd, K = kalman_cv(z, qp, R)
        ep, ev = rms(xs[:, 0] - q), rms(xs[:, 1] - v)
        cover = 100 * np.mean(np.abs(xs[K0:, 0] - q[K0:]) <= 2 * sd[K0:])
        if best is None or ep < best[1]:
            best = (qp, ep, ev)
        print(f"{qp:>8g} {1000*ep:>13.2f} {delay_ms(xs[:, 0], q):>9.1f} {ev:>14.3f} {K[0]:>7.4f} {K[1]:>7.3f} "
              f"{cover:>15.1f}")
    q_best = best[0]
    print(f"    best position RMS at q = {q_best:g}: {1000*best[1]:.2f} mrad, velocity RMS {best[2]:.3f} rad/s")

    print("\n[3] break it: wrong R (true R fixed, filter uses R_used), q =", q_best)
    print(f"{'R_used/R':>9} {'pos RMS mrad':>13} {'vel RMS rad/s':>14} {'2sigma cover %':>15}")
    for f in [0.01, 0.1, 1, 10, 100]:
        xs, sd, _ = kalman_cv(z, q_best, f * R)
        cover = 100 * np.mean(np.abs(xs[K0:, 0] - q[K0:]) <= 2 * sd[K0:])
        print(f"{f:>9g} {1000*rms(xs[:, 0]-q):>13.2f} {rms(xs[:, 1]-v):>14.3f} {cover:>15.1f}")

    print("\n[4] break it: unmodeled acceleration, a 0.3 rad step command at t = 5 s")
    q2, v2 = simulate(lambda t: reference(t) + (0.3 if t >= 5.0 else 0.0))
    z2 = encoder(q2, np.random.default_rng(SEED))
    win = slice(int(5.0 / DT), int(5.5 / DT))          # 0.5 s after the step
    print(f"    true peak velocity after the step: {np.max(np.abs(v2[win])):.2f} rad/s")
    print(f"{'q':>8} {'max pos err mrad':>17} {'max vel err rad/s':>18} {'pos RMS (smooth part) mrad':>27}")
    smooth = slice(K0, int(5.0 / DT))
    for qp in [q_best, 100 * q_best, 10000 * q_best]:
        xs, _, _ = kalman_cv(z2, qp, R)
        e = xs[:, 0] - q2
        print(f"{qp:>8g} {1000*np.max(np.abs(e[win])):>17.1f} {np.max(np.abs(xs[win, 1]-v2[win])):>18.2f} "
              f"{1000*np.sqrt(np.mean(e[smooth]**2)):>27.2f}")


if __name__ == "__main__":
    main()
