"""Lesson 18 · Build it: a constant-velocity Kalman filter from scratch, checked at every step.

Step 1  The model F, Q, H, R; Q against SciPy's matrix exponential (Van Loan).
Step 2  Predict: one step and 1 s without readings vs the closed form.
Step 3  Update: one reading vs the information form of the same Bayes rule.
Step 4  The lesson's encoder signal: our filter vs the Experiment's filter and table.
Step 5  The steady-state gain vs scipy.linalg.solve_discrete_are.
Step 6  Honest uncertainty: claimed sigma vs actual error, coverage, and the q_c/R ratio.

The functions live in physai/estimation.py (the lesson shows them step by step).
The signal (MuJoCo shoulder_pan + encoder model) is taken from experiment.py, so both tell one story.
Run:  uv run python experiments/18-kalman-filter/build.py      (~1 s, deterministic)
"""
import importlib.util
import os
import sys

import mujoco
import numpy as np
import scipy
from scipy.linalg import expm, solve_discrete_are

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "..")]
from physai.estimation import (coverage, cv_model, kalman_filter, kf_predict, kf_update,  # noqa: E402
                               moving_average, steady_state_gain)

spec = importlib.util.spec_from_file_location("kf_experiment", os.path.join(HERE, "experiment.py"))
exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp)
DT, K0 = exp.DT, exp.K0

print(f"MuJoCo {mujoco.__version__} | NumPy {np.__version__} | SciPy {scipy.__version__}")


def fmt(M, p=4):
    """Matrix on one line, rows separated by ';' (p significant digits)."""
    rows = np.atleast_2d(M)
    return "[" + "; ".join(" ".join(f"{v:.{p}g}" for v in row) for row in rows) + "]"


print("\n== Step 1. The model at dt = 2 ms, q_c = 1 rad^2/s^3, R = 1e-4 rad^2 ==")
F, Q, H, R = cv_model(DT, 1.0, 1e-4)
print(f"F = {fmt(F)}   H = {fmt(H)}   R = {fmt(R)}")
print(f"Q = {fmt(Q)}")
A = np.array([[0.0, 1.0], [0.0, 0.0]])                 # continuous model: d/dt [q, qd] = A x + [0, 1] a
W = np.array([[0.0, 0.0], [0.0, 1.0]])                 # white acceleration of intensity q_c = 1 enters qd
C = expm(np.block([[-A, W], [np.zeros((2, 2)), A.T]]) * DT)   # Van Loan's method
Phi, Q_vl = C[2:, 2:].T, C[2:, 2:].T @ C[:2, 2:]
print(f"max |F - expm(A dt)| = {np.abs(F - Phi).max():.1e}, max |Q - Q_VanLoan| / max |Q| = "
      f"{np.abs(Q - Q_vl).max() / np.abs(Q).max():.1e}")

print("\n== Step 2. Predict: one step from x = (0.1 rad, 0.5 rad/s), P = diag(1e-4, 1) ==")
x0, P0 = np.array([0.1, 0.5]), np.diag([1e-4, 1.0])
x1, P1 = kf_predict(x0, P0, F, Q)
x_hand = np.array([0.1 + 0.5 * DT, 0.5])
P_hand = np.array([[1e-4 + DT**2 + DT**3 / 3, DT + DT**2 / 2], [DT + DT**2 / 2, 1.0 + DT]])
print(f"x- = {fmt(x1, 6)}   P- = {fmt(P1, 6)}")
print(f"max |x- - hand| = {np.abs(x1 - x_hand).max():.1e}, max |P- - hand| = {np.abs(P1 - P_hand).max():.1e}")
x, P = x0, P0
for _ in range(500):                                   # 1 s of forecasts without a single reading
    x, P = kf_predict(x, P, F, Q)
T = 500 * DT
Fn = np.array([[1.0, T], [0.0, 1.0]])
P_closed = Fn @ P0 @ Fn.T + np.array([[T**3 / 3, T**2 / 2], [T**2 / 2, T]])
print(f"after 500 predicts (1 s): position sigma {np.sqrt(P0[0, 0]):.3f} -> {np.sqrt(P[0, 0]):.3f} rad, "
      f"max |P - closed form| / max |P| = {np.abs(P - P_closed).max() / np.abs(P).max():.1e}")

print("\n== Step 3. Update: the one-step forecast meets a reading z = 0.12 rad ==")
x2, P2, K = kf_update(x1, P1, 0.12, H, R)
P_info = np.linalg.inv(np.linalg.inv(P1) + H.T @ np.linalg.inv(R) @ H)    # information form
x_info = P_info @ (np.linalg.inv(P1) @ x1 + (H.T @ np.linalg.inv(R)).ravel() * 0.12)
print(f"K = {fmt(K.ravel())}   x+ = {fmt(x2)}   sigma_q {np.sqrt(P1[0, 0]) * 1000:.2f} -> "
      f"{np.sqrt(P2[0, 0]) * 1000:.2f} mrad")
print(f"scalar check K_q = P00/(P00 + R) = {P1[0, 0] / (P1[0, 0] + R[0, 0]):.4f}")
print(f"max |x+ - info form| = {np.abs(x2 - x_info).max():.1e}, max |P+ - info form| = {np.abs(P2 - P_info).max():.1e}")

print("\n== Step 4. The lesson's encoder signal (Experiment setup, seed 0) ==")
q, v = exp.simulate(exp.reference)
z = exp.encoder(q, np.random.default_rng(exp.SEED))
r = exp.SIGMA**2 + (2 * np.pi / exp.COUNTS) ** 2 / 12


def rms(e):
    return np.sqrt(np.mean(e[K0:] ** 2))


print(f"R = {r:.3e} rad^2, {len(z)} readings, scored after the first {K0}")
print(f"{'filter':>14} {'pos RMS mrad':>13} {'vel RMS rad/s':>14} {'K_q':>7} {'K_qd':>7} {'max |ours - Exp.|':>18}")
for q_c in (0.1, 1.0):
    xs, Ps, K = kalman_filter(z, DT, q_c, r)
    ref = exp.kalman_cv(z, q_c, r)[0]
    print(f"{f'KF q_c = {q_c:g}':>14} {1000 * rms(xs[:, 0] - q):>13.2f} {rms(xs[:, 1] - v):>14.3f} "
          f"{K[0, 0]:>7.4f} {K[1, 0]:>7.3f} {np.abs(xs - ref).max():>18.1e}")
ma5 = moving_average(z, 5)
print(f"{'MA N = 5':>14} {1000 * rms(ma5 - q):>13.2f} {'':>14} {'':>7} {'':>7} "
      f"{np.abs(ma5 - exp.moving_average(z, 5)).max():>18.1e}")

print("\n== Step 5. Steady-state gain: run, Riccati iteration, scipy.linalg.solve_discrete_are ==")
print(f"{'q_c':>7} {'K_q':>7} {'K_qd':>8} {'|K_iter - K_dare|':>18} {'|K_run - K_dare|':>17}")
QCS = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
for q_c in QCS:
    F, Q, H, R = cv_model(DT, q_c, r)
    P_dare = solve_discrete_are(F.T, H.T, Q, R)        # dual problem: forecast covariance P-
    K_dare = P_dare @ H.T @ np.linalg.inv(H @ P_dare @ H.T + R)
    K_iter, _ = steady_state_gain(F, Q, H, R)
    K_run = kalman_filter(z, DT, q_c, r)[2]
    print(f"{q_c:>7g} {K_dare[0, 0]:>7.4f} {K_dare[1, 0]:>8.3f} {np.abs(K_iter - K_dare).max():>18.1e} "
          f"{np.abs(K_run - K_dare).max():>17.1e}")

print("\n== Step 6. Honest uncertainty: claimed sigma vs actual error ==")
print(f"{'q_c':>7} {'claimed sigma mrad':>19} {'actual RMS mrad':>16} {'2-sigma coverage %':>19}")
for q_c in QCS:
    xs, Ps, _ = kalman_filter(z, DT, q_c, r)
    sd = np.sqrt(Ps[:, 0, 0])
    print(f"{q_c:>7g} {1000 * sd[-1]:>19.2f} {1000 * rms(xs[:, 0] - q):>16.2f} "
          f"{100 * coverage(xs[K0:, 0] - q[K0:], sd[K0:]):>19.1f}")
xs1, Ps1, _ = kalman_filter(z, DT, 1.0, r)
xs100, Ps100, _ = kalman_filter(z, DT, 100.0, 100 * r, P0=100 * np.diag([r, 1.0]))
print(f"q_c and R (and P0) x 100: max |x change| = {np.abs(xs100 - xs1).max():.1e} rad, "
      f"P ratio {np.median(Ps100[:, 0, 0] / Ps1[:, 0, 0]):.3f}")
