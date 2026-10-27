"""Trajectory optimization and optimal feedback (lessons 23-24 · Build it).

Lesson 23: a torque-limited pendulum swing-up by direct collocation (trapezoidal defects) and direct shooting,
both handed to SciPy's SLSQP. Lesson 24: linearization by finite differences, discrete LQR by Riccati
iteration, and the MPPI update. Plain NumPy (SciPy only as the NLP solver).
"""
import numpy as np
from scipy.optimize import minimize


# ---------------------------------------------------------------- lesson 23: the pendulum and its NLP
def pendulum_f(x, u, m=1.0, l=1.0, b=0.1, g=9.81):
    """Pendulum dynamics, vectorized over leading dims. x = (theta, omega), theta = 0 hanging down."""
    th, w = x[..., 0], x[..., 1]
    return np.stack([w, (u - b * w - m * g * l * np.sin(th)) / (m * l * l)], axis=-1)


def pendulum_jac(x, u, m=1.0, l=1.0, b=0.1, g=9.81):
    """df/dx (2 x 2) and df/du (2,) of pendulum_f at one knot."""
    inertia = m * l * l
    return np.array([[0.0, 1.0], [-m * g * l * np.cos(x[0]) / inertia, -b / inertia]]), np.array([0.0, 1 / inertia])


def rk4_rollout(f, x0, U, T, sub=20):
    """Final state for knot torques U (rows: one control sequence each), u linear between knots, RK4."""
    U = np.atleast_2d(U)
    N = U.shape[1] - 1
    h = T / N / sub
    x = np.tile(x0, (U.shape[0], 1))
    for k in range(N):
        for s in range(sub):
            a = s / sub
            u0 = U[:, k] * (1 - a) + U[:, k + 1] * a                         # start of the sub-step
            um = U[:, k] * (1 - a - 0.5 / sub) + U[:, k + 1] * (a + 0.5 / sub)  # its midpoint
            u1 = U[:, k] * (1 - a - 1.0 / sub) + U[:, k + 1] * (a + 1.0 / sub)  # its end
            k1 = f(x, u0)
            k2 = f(x + h / 2 * k1, um)
            k3 = f(x + h / 2 * k2, um)
            k4 = f(x + h * k3, u1)
            x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def effort_cost(U, h, w_smooth):
    """J = int u^2 dt (trapezoid) + w_smooth * sum (du)^2 / h, and its gradient dJ/dU."""
    du = np.diff(U)
    J = h * np.sum((U[:-1] ** 2 + U[1:] ** 2) / 2) + w_smooth * np.sum(du ** 2) / h
    g = 2 * h * U
    g[0] *= 0.5
    g[-1] *= 0.5
    gs = 2 * w_smooth * du / h
    g[:-1] -= gs
    g[1:] += gs
    return J, g


def trapezoid_defects(f, X, U, h):
    """x_{k+1} - x_k - h/2 (f_k + f_{k+1}) for every interval: an N x 2 array, zero when the knots obey the rule."""
    F = f(X, U)
    return X[1:] - X[:-1] - h / 2 * (F[1:] + F[:-1])


def trapezoid_jacobian(jac, X, U, h):
    """Derivative of the flattened defects w.r.t. z = [X.ravel(), U]: (2N) x 3(N+1), banded."""
    n = len(U)
    J = np.zeros((2 * (n - 1), 3 * n))
    for k in range(n - 1):
        r = slice(2 * k, 2 * k + 2)
        for j, sign in ((k, -1.0), (k + 1, 1.0)):
            dfx, dfu = jac(X[j], U[j])
            J[r, 2 * j:2 * j + 2] = sign * np.eye(2) - h / 2 * dfx
            J[r, 2 * n + j] = -h / 2 * dfu
    return J


def collocate(f, jac, x0, xg, N, T, umax, w_smooth, X_init, U_init, maxiter=500):
    """Direct collocation: unknowns are knot states and torques, constraints are the trapezoidal defects."""
    h, n = T / N, N + 1

    def unpack(z):
        return z[:2 * n].reshape(n, 2), z[2 * n:]

    def cons(z):
        X, U = unpack(z)
        return np.concatenate([trapezoid_defects(f, X, U, h).ravel(), X[0] - x0, X[-1] - xg])

    def cons_jac(z):
        X, U = unpack(z)
        B = np.zeros((4, 3 * n))
        B[0:2, 0:2], B[2:4, 2 * n - 2:2 * n] = np.eye(2), np.eye(2)    # boundary states
        return np.vstack([trapezoid_jacobian(jac, X, U, h), B])

    def grad(z):
        g = np.zeros_like(z)
        g[2 * n:] = effort_cost(unpack(z)[1], h, w_smooth)[1]
        return g

    r = minimize(lambda z: effort_cost(unpack(z)[1], h, w_smooth)[0], np.r_[X_init.ravel(), U_init], jac=grad,
                 method="SLSQP", bounds=[(None, None)] * (2 * n) + [(-umax, umax)] * n,
                 constraints=[{"type": "eq", "fun": cons, "jac": cons_jac}],
                 options=dict(maxiter=maxiter, ftol=1e-9))
    X, U = unpack(r.x)
    return r, X, U, float(np.abs(cons(r.x)).max())


def shoot(f, x0, xg, N, T, umax, w_smooth, U_init, sub=20, eps=1e-6, maxiter=500):
    """Direct shooting: unknowns are the knot torques, the constraint is the RK4 rollout's final state."""
    h = T / N

    def term(U):
        return rk4_rollout(f, x0, U, T, sub)[0] - xg

    def term_jac(U):                     # forward differences, all perturbations in one batched rollout
        xs = rk4_rollout(f, x0, np.vstack([U, U + eps * np.eye(N + 1)]), T, sub)
        return ((xs[1:] - xs[0]) / eps).T

    r = minimize(lambda U: effort_cost(U, h, w_smooth)[0], U_init, jac=lambda U: effort_cost(U, h, w_smooth)[1],
                 method="SLSQP", bounds=[(-umax, umax)] * (N + 1),
                 constraints=[{"type": "eq", "fun": term, "jac": term_jac}],
                 options=dict(maxiter=maxiter, ftol=1e-9))
    return r, r.x, float(np.abs(term(r.x)).max())


# ---------------------------------------------------------------- lesson 24: LQR and MPPI
def linearize(step, x, u, eps=1e-6):
    """A = d step / d x, B = d step / d u by central differences of any step(x, u) -> next state."""
    A = np.column_stack([(step(x + e, u) - step(x - e, u)) / (2 * eps) for e in eps * np.eye(len(x))])
    B = np.column_stack([(step(x, u + e) - step(x, u - e)) / (2 * eps) for e in eps * np.eye(len(u))])
    return A, B


def riccati_step(A, B, Q, R, P):
    """One backward step of the Riccati recursion: (P one step earlier, the gain K for that step)."""
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return Q + A.T @ P @ (A - B @ K), K


def lqr_finite(A, B, Q, R, Qf, N):
    """Finite-horizon discrete LQR: gains K_0 .. K_{N-1} (u_t = -K_t dx_t) and the cost-to-go matrix P_0."""
    P, gains = Qf, []
    for _ in range(N):
        P, K = riccati_step(A, B, Q, R, P)
        gains.append(K)
    return gains[::-1], P


def dlqr(A, B, Q, R, tol=1e-12, max_iter=100_000):
    """Infinite-horizon discrete LQR: iterate the Riccati recursion until P stops changing. Returns (K, P, steps)."""
    P = Q.copy()
    for i in range(1, max_iter + 1):
        P_new, K = riccati_step(A, B, Q, R, P)
        if np.abs(P_new - P).max() <= tol * np.abs(P_new).max():
            return K, P_new, i
        P = P_new
    raise RuntimeError("Riccati iteration did not converge")


def mppi_update(samples, costs, lam):
    """MPPI: weights exp(-(S_k - min S) / lam), normalized; returns (new plan = weighted mean of samples, weights)."""
    w = np.exp(-(costs - costs.min()) / lam)
    w = w / w.sum()
    return w @ samples, w
