"""Lesson 23 · Trajectory optimization: direct shooting vs direct collocation.

System: torque-limited pendulum swing-up (|u| <= 5 N m, m g l = 9.81 N m, so it must pump).
Both methods optimize the same control parameterization: u(t) piecewise linear through N+1 knots
on a T = 4 s horizon, cost = effort + smoothness, goal x(T) = (pi, 0) as a hard constraint.

  shooting     decision variables = knot torques; the dynamics are integrated (RK4, 20 sub-steps
               per interval); constraint = terminal state.
  collocation  decision variables = knot states AND torques; dynamics enforced only at the knots
               through trapezoidal defects; constraint = all defects + boundary states.

Every solution is then replayed open-loop in a MuJoCo pendulum (RK4, 1 ms) to measure how far the
REAL trajectory ends from the goal: that is the constraint violation that matters.

Sweep   number of knots N -> cost, solver iterations, reported vs replayed goal error.
Break   collocation with poor initial guesses (N = 40) -> local minima, measured.

Run:  python experiment.py      (deterministic, fixed seeds)
"""
import platform

import mujoco
import numpy as np
import scipy
from scipy.optimize import minimize

print(f"MuJoCo {mujoco.__version__} | NumPy {np.__version__} | SciPy {scipy.__version__} "
      f"| Python {platform.python_version()} | {platform.machine()}")

G, L, M, B = 9.81, 1.0, 1.0, 0.1       # gravity, length, point mass, joint damping
UMAX, T = 5.0, 4.0                     # torque limit (N m), horizon (s)
W_SMOOTH = 0.05                        # weight of the smoothness term
I = M * L * L
X0, XG = np.array([0.0, 0.0]), np.array([np.pi, 0.0])   # hanging down -> upright, at rest
SUB = 20                               # RK4 sub-steps per interval (shooting)


def f(x, u):
    """Pendulum dynamics, vectorized over leading dims. x = (theta, omega), theta = 0 hanging."""
    th, w = x[..., 0], x[..., 1]
    return np.stack([w, (u - B * w - M * G * L * np.sin(th)) / I], axis=-1)


def cost_and_grad(U, h):
    """J = int u^2 dt (trapezoid) + W_SMOOTH * sum (du)^2 / h  (approximates int udot^2 dt)."""
    du = np.diff(U)
    J = h * np.sum((U[:-1] ** 2 + U[1:] ** 2) / 2) + W_SMOOTH * np.sum(du ** 2) / h
    g = 2 * h * U
    g[0] *= 0.5
    g[-1] *= 0.5
    gs = 2 * W_SMOOTH * du / h
    g[:-1] -= gs
    g[1:] += gs
    return J, g


# ---------------------------------------------------------------- direct shooting
def rollout_batch(Ubatch, N):
    """Integrate many control sequences at once (rows of Ubatch). Returns final states."""
    h = T / N / SUB
    x = np.tile(X0, (Ubatch.shape[0], 1))
    for k in range(N):
        for s in range(SUB):
            a = s / SUB
            u0 = Ubatch[:, k] * (1 - a) + Ubatch[:, k + 1] * a
            um = Ubatch[:, k] * (1 - a - 0.5 / SUB) + Ubatch[:, k + 1] * (a + 0.5 / SUB)
            u1 = Ubatch[:, k] * (1 - a - 1.0 / SUB) + Ubatch[:, k + 1] * (a + 1.0 / SUB)
            k1 = f(x, u0)
            k2 = f(x + h / 2 * k1, um)
            k3 = f(x + h / 2 * k2, um)
            k4 = f(x + h * k3, u1)
            x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def shooting(N, U0):
    h = T / N
    eps = 1e-6

    def term(U):
        return rollout_batch(U[None], N)[0] - XG

    def term_jac(U):            # forward differences, all perturbations in one batched rollout
        batch = np.vstack([U, U + eps * np.eye(N + 1)])
        xs = rollout_batch(batch, N)
        return ((xs[1:] - xs[0]) / eps).T

    r = minimize(lambda U: cost_and_grad(U, h)[0], U0, jac=lambda U: cost_and_grad(U, h)[1],
                 bounds=[(-UMAX, UMAX)] * (N + 1), method="SLSQP",
                 constraints=[{"type": "eq", "fun": term, "jac": term_jac}],
                 options=dict(maxiter=500, ftol=1e-9))
    viol = np.max(np.abs(term(r.x)))
    return r, r.x, viol


# ---------------------------------------------------------------- direct collocation
def collocation(N, X_init, U_init):
    h, n = T / N, N + 1

    def unpack(z):
        return z[:2 * n].reshape(n, 2), z[2 * n:]

    def obj(z):
        return cost_and_grad(unpack(z)[1], h)[0]

    def obj_grad(z):
        g = np.zeros_like(z)
        g[2 * n:] = cost_and_grad(unpack(z)[1], h)[1]
        return g

    def defects(z):
        X, U = unpack(z)
        F = f(X, U)
        d = X[1:] - X[:-1] - h / 2 * (F[1:] + F[:-1])      # trapezoidal rule
        return np.concatenate([d.ravel(), X[0] - X0, X[-1] - XG])

    def defects_jac(z):
        X, U = unpack(z)
        J = np.zeros((2 * N + 4, 3 * n))
        dfu = np.array([0.0, 1.0 / I])
        for k in range(N):
            r = slice(2 * k, 2 * k + 2)
            for j, sign in ((k, -1.0), (k + 1, 1.0)):
                dfx = np.array([[0.0, 1.0], [-M * G * L * np.cos(X[j, 0]) / I, -B / I]])
                J[r, 2 * j:2 * j + 2] = sign * np.eye(2) - h / 2 * dfx
                J[r, 2 * n + j] = -h / 2 * dfu
        J[2 * N:2 * N + 2, 0:2] = np.eye(2)
        J[2 * N + 2:, 2 * n - 2:2 * n] = np.eye(2)
        return J

    z0 = np.concatenate([X_init.ravel(), U_init])
    r = minimize(obj, z0, jac=obj_grad, method="SLSQP",
                 bounds=[(None, None)] * (2 * n) + [(-UMAX, UMAX)] * n,
                 constraints=[{"type": "eq", "fun": defects, "jac": defects_jac}],
                 options=dict(maxiter=500, ftol=1e-9))
    X, U = unpack(r.x)
    return r, U, np.max(np.abs(defects(r.x))), X


# ---------------------------------------------------------------- MuJoCo replay
XML = f"""
<mujoco>
  <option timestep="0.001" integrator="RK4" gravity="0 0 -{G}"/>
  <worldbody>
    <body name="pole">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="{B}"/>
      <inertial pos="0 0 -{L}" mass="{M}" diaginertia="1e-9 1e-9 1e-9"/>
      <geom type="capsule" fromto="0 0 0 0 0 -{L}" size="0.02" mass="0"/>
    </body>
  </worldbody>
  <actuator><motor joint="hinge" gear="1" ctrllimited="true" ctrlrange="-{UMAX} {UMAX}"/></actuator>
</mujoco>"""
MODEL = mujoco.MjModel.from_xml_string(XML)


def replay(U):
    """Apply u(t) (linear between knots) open-loop in MuJoCo. Returns final (theta, omega)."""
    d = mujoco.MjData(MODEL)
    N = len(U) - 1
    tk = np.linspace(0, T, N + 1)
    steps = int(round(T / MODEL.opt.timestep))
    for _ in range(steps):
        d.ctrl[0] = np.interp(d.time + MODEL.opt.timestep / 2, tk, U)   # midpoint of the 1 ms step
        mujoco.mj_step(MODEL, d)
    return np.array([d.qpos[0], d.qvel[0]])


def err_str(xT):
    return np.degrees(abs(xT[0] - XG[0])), np.degrees(abs(xT[1]))


def swings(U):
    """Number of sign changes of the torque: how many 'pumps' the solution uses."""
    s = np.sign(U[np.abs(U) > 0.05 * UMAX])
    return int(np.sum(s[1:] != s[:-1]))


print(f"\nPendulum m={M} kg, l={L} m, b={B} N m s, |u| <= {UMAX} N m (m g l = {M * G * L:.2f} N m), "
      f"T = {T} s, W_SMOOTH = {W_SMOOTH}")
print("Initial guess for the sweep: zero torque, state resting at the bottom.")
print("'solver viol' = max constraint residual the solver sees; 'replay err' = goal error of the "
      "same u(t) replayed in MuJoCo.")
print(f"\n{'method':>11} | {'N':>3} | {'converged':>9} | {'iters':>5} | {'cost':>7} | "
      f"{'solver viol':>11} | {'replay err theta (deg)':>22} | {'replay err omega (deg/s)':>24} | {'pumps':>5}")
for N in [5, 10, 20, 40, 80]:
    for name in ["shooting", "collocation"]:
        if name == "shooting":
            r, U, viol = shooting(N, np.zeros(N + 1))
        else:
            r, U, viol, _ = collocation(N, np.zeros((N + 1, 2)), np.zeros(N + 1))
        e_th, e_w = err_str(replay(U))
        print(f"{name:>11} | {N:3d} | {str(bool(r.success)):>9} | {r.nit:5d} | {r.fun:7.2f} | "
              f"{viol:11.1e} | {e_th:22.2f} | {e_w:24.2f} | {swings(U):5d}")

# ---------------------------------------------------------------- Break it: initial guesses
N = 40
tt = np.linspace(0, 1, N + 1)
inits = {"rest at bottom, u=0": (np.zeros((N + 1, 2)), np.zeros(N + 1)),
         "straight line theta 0->pi, u=0": (np.outer(tt, XG), np.zeros(N + 1))}
rng = np.random.default_rng(0)                      # one generator; draws 0..7 below
for s in range(8):
    Xr = np.outer(tt, XG) + rng.normal(0.0, 1.0, (N + 1, 2))
    inits[f"line + noise, draw {s}"] = (Xr, rng.uniform(-UMAX, UMAX, N + 1))

print(f"\nBreak it · collocation, N = {N}, same problem, different initial guesses")
print(f"{'initial guess':>30} | {'converged':>9} | {'iters':>5} | {'cost':>7} | "
      f"{'replay err theta (deg)':>22} | {'pumps':>5} | {'theta range (deg)':>17}")
for name, (Xi, Ui) in inits.items():
    r, U, viol, X = collocation(N, Xi, Ui)
    e_th, _ = err_str(replay(U))
    lo, hi = np.degrees(X[:, 0].min()), np.degrees(X[:, 0].max())
    print(f"{name:>30} | {str(bool(r.success)):>9} | {r.nit:5d} | {r.fun:7.2f} | {e_th:22.2f} | "
          f"{swings(U):5d} | {lo:7.0f} .. {hi:5.0f}")
