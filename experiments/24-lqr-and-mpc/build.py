"""Lesson 24 · Build it: linearization, LQR by Riccati iteration and the MPPI update, from scratch.

Step 1  Linearize the cart-pole by central differences of mj_step vs mujoco.mjd_transitionFD.
Step 2  Finite-horizon LQR: the backward Riccati recursion, gains for growing horizons.
Step 3  Infinite-horizon LQR by iterating to a fixed point vs scipy.linalg.solve_discrete_are.
Step 4  Use the gain: release angles from the Experiment's region-of-attraction table.
Step 5  The MPPI update inside a receding-horizon loop: swing-up from hanging down.

The functions live in physai/optimal.py (the lesson shows them step by step).
Run:  uv run python experiments/24-lqr-and-mpc/build.py      (~5 s, deterministic)
"""
import os
import sys

import mujoco
import numpy as np
import scipy
from mujoco import rollout
from scipy.linalg import solve_discrete_are

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "..")]
from physai.optimal import dlqr, linearize, lqr_finite, mppi_update  # noqa: E402

print(f"MuJoCo {mujoco.__version__} | NumPy {np.__version__} | SciPy {scipy.__version__}")
np.set_printoptions(precision=4, suppress=True)
XML = """
<mujoco>
  <option timestep="0.01" integrator="Euler"/>
  <worldbody>
    <body name="cart">
      <joint name="slider" type="slide" axis="1 0 0" range="-1.5 1.5" limited="true"/>
      <geom type="box" size="0.15 0.1 0.05" mass="1"/>
      <body name="pole">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.6" size="0.02" mass="0.2"/>
      </body>
    </body>
  </worldbody>
  <actuator><motor joint="slider" gear="1" ctrllimited="true" ctrlrange="-100 100"/></actuator>
</mujoco>"""
m = mujoco.MjModel.from_xml_string(XML)
d = mujoco.MjData(m)
Q, R = np.diag([1.0, 10.0, 0.1, 0.1]), np.array([[0.01]])


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def mj_step_fn(x, u):
    """One MuJoCo step from state x = (x_cart, theta, xdot, thetadot) with force u: the next state."""
    mujoco.mj_resetData(m, d)
    d.qpos[:], d.qvel[:], d.ctrl[:] = x[:2], x[2:], u
    mujoco.mj_step(m, d)
    return np.r_[d.qpos, d.qvel]


print("\n== Step 1. Linearize at the upright: central differences of mj_step vs mjd_transitionFD ==")
A, B = linearize(mj_step_fn, np.zeros(4), np.zeros(1))
mujoco.mj_resetData(m, d)
mujoco.mj_forward(m, d)
A_mj, B_mj = np.zeros((4, 4)), np.zeros((4, 1))
mujoco.mjd_transitionFD(m, d, 1e-6, True, A_mj, B_mj, None, None)
print("theta row of A:", A[1], "| B:", B.ravel())
print(f"max |A - A_mj| = {np.abs(A - A_mj).max():.1e} | max |B - B_mj| = {np.abs(B - B_mj).max():.1e}")
print(f"open-loop max |eig(A)| = {np.abs(np.linalg.eigvals(A)).max():.4f}")

print("\n== Step 2. Finite-horizon LQR (Qf = Q): the first gain K_0 for growing horizons ==")
for N in (10, 50, 200, 1000):
    K0 = lqr_finite(A, B, Q, R, Q, N)[0][0]
    print(f"N = {N:4d} steps ({N * m.opt.timestep:5.1f} s)  K_0 = {K0.ravel()}")

print("\n== Step 3. Infinite horizon: iterate Riccati to a fixed point vs solve_discrete_are ==")
K, P, steps = dlqr(A, B, Q, R)
P_ref = solve_discrete_are(A, B, Q, R)
print(f"converged after {steps} iterations | max |P - P_scipy| / max |P| = {np.abs(P - P_ref).max() / np.abs(P_ref).max():.1e}")
print("K =", K.ravel())
print("closed-loop |eig(A - B K)| =", np.sort(np.abs(np.linalg.eigvals(A - B @ K))))


def lqr_trial(theta0, F, T=5.0):
    """Release the pole at rest at theta0, cart at 0, u = clip(-K x, +-F) for T s (the Experiment's rule)."""
    dd = mujoco.MjData(m)
    dd.qpos[1] = theta0
    for _ in range(int(round(T / m.opt.timestep))):
        x = np.r_[dd.qpos, dd.qvel]
        x[1] = wrap(x[1])
        dd.ctrl[0] = np.clip(-(K @ x)[0], -F, F)
        mujoco.mj_step(m, dd)
        if abs(wrap(dd.qpos[1])) > np.pi / 2:
            return "fell"
        if abs(dd.qpos[0]) > 1.49:
            return "track end"
    return "ok" if abs(wrap(dd.qpos[1])) < np.radians(2) else "not settled"


print("\n== Step 4. Use the gain: release at rest, u = clip(-K x, +-F), 5 s ==")
print(f"{'theta0 (deg)':>12} | {'F=5 N':>9} | {'F=10 N':>9} | {'F=50 N':>9}")
for a in (15, 20, 35, 50, 55):
    print(f"{a:12d} | " + " | ".join(f"{lqr_trial(np.radians(a), F):>9}" for F in (5, 10, 50)))

print("\n== Step 5. MPPI swing-up from hanging down (|F| <= 20 N, 64 samples, sigma 8 N, lambda 0.5, 50 Hz) ==")
SPEC = mujoco.mjtState.mjSTATE_FULLPHYSICS
datas = [mujoco.MjData(m) for _ in range(8)]


def stage_cost(S, U):
    x, th, xd, thd = S[..., 1], S[..., 2], S[..., 3], S[..., 4]
    return (5 * (1 - np.cos(th)) + 0.5 * x ** 2 + 0.01 * thd ** 2 + 0.01 * xd ** 2
            + 1e-4 * U ** 2 + 100 * (np.abs(x) > 1.4))


def mpc_trial(horizon_s, seed, T=6.0, sub=2, n_samples=64, sigma=8.0, lam=0.5, F=20.0):
    rng = np.random.default_rng(seed)
    H = int(round(horizon_s / (m.opt.timestep * sub)))
    dd = mujoco.MjData(m)
    dd.qpos[1] = np.pi
    st, U, cost, log = np.zeros(mujoco.mj_stateSize(m, SPEC)), np.zeros(H), 0.0, []
    for _ in range(int(round(T / (m.opt.timestep * sub)))):
        mujoco.mj_getState(m, dd, st, SPEC)
        Uk = np.clip(U + rng.normal(0.0, sigma, (n_samples, H)), -F, F)    # noisy variations of the plan
        ctrl = np.repeat(Uk, sub, axis=1)[:, :, None]
        S, _ = rollout.rollout(m, datas, st, ctrl)                          # MuJoCo simulates each one
        U = np.clip(mppi_update(Uk, stage_cost(S, ctrl[..., 0]).sum(axis=1), lam)[0], -F, F)
        u, U = U[0], np.r_[U[1:], U[-1]]                                    # apply the first, shift the rest
        for _ in range(sub):
            dd.ctrl[0] = u
            mujoco.mj_step(m, dd)
            cost += stage_cost(np.r_[0.0, dd.qpos, dd.qvel], u) * m.opt.timestep
        log.append(np.degrees(wrap(dd.qpos[1])))
    worst = np.abs(log[-50:]).max()                                         # worst |theta| in the last second
    return worst, cost


res = np.array([mpc_trial(0.5, s) for s in range(8)])
print(f"horizon 0.5 s, 8 seeds: held (|theta| < 10 deg for the last 1 s) {int((res[:, 0] < 10).sum())}/8 | "
      f"median worst |theta| {np.median(res[:, 0]):.1f} deg | median cost {np.median(res[:, 1]):.2f}")
