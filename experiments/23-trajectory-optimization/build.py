"""Lesson 23 · Build it: direct collocation and direct shooting from scratch, solved with SciPy's SLSQP.

Step 1  Pendulum dynamics and an RK4 rollout vs a MuJoCo pendulum.
Step 2  The cost and its analytic gradient vs finite differences.
Step 3  Trapezoidal defects: their Jacobian vs finite differences, and how they shrink with h.
Step 4  Direct collocation for N = 10, 20, 40, replayed in MuJoCo.
Step 5  Direct shooting for N = 10, replayed in MuJoCo.

The functions live in physai/optimal.py (the lesson shows them step by step).
Run:  uv run python experiments/23-trajectory-optimization/build.py      (~3 s, deterministic)
"""
import os
import sys

import mujoco
import numpy as np
import scipy
from scipy.optimize import approx_fprime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "..")]
from physai.optimal import (collocate, effort_cost, pendulum_f, pendulum_jac, rk4_rollout, shoot,  # noqa: E402
                            trapezoid_defects, trapezoid_jacobian)

print(f"MuJoCo {mujoco.__version__} | NumPy {np.__version__} | SciPy {scipy.__version__}")
UMAX, T, W = 5.0, 4.0, 0.05
X0, XG = np.array([0.0, 0.0]), np.array([np.pi, 0.0])
XML = """
<mujoco>
  <option timestep="0.001" integrator="RK4" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="pole">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="0.1"/>
      <inertial pos="0 0 -1" mass="1" diaginertia="1e-9 1e-9 1e-9"/>
    </body>
  </worldbody>
  <actuator><motor joint="hinge" ctrllimited="true" ctrlrange="-5 5"/></actuator>
</mujoco>"""
MODEL = mujoco.MjModel.from_xml_string(XML)


def replay(U):
    """u(t) linear between knots, applied open-loop in MuJoCo (RK4, 1 ms). Returns the final (theta, omega)."""
    d = mujoco.MjData(MODEL)
    tk = np.linspace(0, T, len(U))
    for _ in range(int(round(T / MODEL.opt.timestep))):
        d.ctrl[0] = np.interp(d.time + MODEL.opt.timestep / 2, tk, U)
        mujoco.mj_step(MODEL, d)
    return np.array([d.qpos[0], d.qvel[0]])


print("\n== Step 1. RK4 rollout of the pendulum vs MuJoCo (u(t) = 4 sin(1.5 t) through 11 knots) ==")
U_test = 4 * np.sin(1.5 * np.linspace(0, T, 11))
ours, mj = rk4_rollout(pendulum_f, X0, U_test, T)[0], replay(U_test)
print(f"final state ours (theta, omega) = ({ours[0]:.6f}, {ours[1]:.6f}) | MuJoCo = ({mj[0]:.6f}, {mj[1]:.6f})")
print(f"max difference {np.abs(ours - mj).max():.1e}")

print("\n== Step 2. Cost gradient vs finite differences (N = 10, random torques) ==")
U_r = np.random.default_rng(0).uniform(-UMAX, UMAX, 11)
J, g = effort_cost(U_r, T / 10, W)
g_fd = approx_fprime(U_r, lambda U: effort_cost(U, T / 10, W)[0], 1e-6)
print(f"J = {J:.4f} | max |grad - finite difference| = {np.abs(g - g_fd).max():.1e}")

print("\n== Step 3. Trapezoidal defects ==")
rng = np.random.default_rng(1)
X_r, U_r = rng.normal(size=(11, 2)), rng.uniform(-UMAX, UMAX, 11)
z = np.r_[X_r.ravel(), U_r]
Jd = trapezoid_jacobian(pendulum_jac, X_r, U_r, T / 10)
Jd_fd = approx_fprime(z, lambda z: trapezoid_defects(pendulum_f, z[:22].reshape(11, 2), z[22:], T / 10).ravel(), 1e-7)
print(f"Jacobian {Jd.shape[0]} x {Jd.shape[1]}, nonzeros {np.count_nonzero(Jd)} | "
      f"max |analytic - finite difference| = {np.abs(Jd - Jd_fd).max():.1e}")
print("defects of the true motion (u linear through the knots of 4 sin(1.5 t), RK4 with 200 sub-steps per interval):")
for N in (10, 20, 40):
    U = 4 * np.sin(1.5 * np.linspace(0, T, N + 1))
    Xk = [X0]
    for k in range(N):
        Xk.append(rk4_rollout(pendulum_f, Xk[-1], U[k:k + 2], T / N, sub=200)[0])
    dmax = np.abs(trapezoid_defects(pendulum_f, np.array(Xk), U, T / N)).max()
    print(f"  N = {N:2d}  h = {T / N:.2f} s  max |defect| = {dmax:.2e}")

print("\n== Step 4. Direct collocation, SLSQP from rest at the bottom, replayed in MuJoCo ==")
print(f"{'N':>3} {'converged':>9} {'iters':>5} {'cost':>7} {'solver residual':>15} {'replay err theta (deg)':>22}")
for N in (10, 20, 40):
    r, X, U, res = collocate(pendulum_f, pendulum_jac, X0, XG, N, T, UMAX, W, np.zeros((N + 1, 2)), np.zeros(N + 1))
    err = np.degrees(abs(replay(U)[0] - np.pi))
    print(f"{N:3d} {str(bool(r.success)):>9} {r.nit:5d} {r.fun:7.2f} {res:15.1e} {err:22.2f}")

print("\n== Step 5. Direct shooting, N = 10, SLSQP from zero torque, replayed in MuJoCo ==")
r, U, res = shoot(pendulum_f, X0, XG, 10, T, UMAX, W, np.zeros(11))
xT = replay(U)
print(f"converged {bool(r.success)} | iterations {r.nit} | cost {r.fun:.2f} | solver residual {res:.1e} | "
      f"replay error {np.degrees(abs(xT[0] - np.pi)):.2f} deg, {np.degrees(abs(xT[1])):.2f} deg/s")
