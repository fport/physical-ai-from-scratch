"""Time scalings for point-to-point moves (lesson 13 · Build it).

A move q(t) = q0 + s(t / T) dq is fixed by its shape s(u) on u in [0, 1] with s(0) = 0 and s(1) = 1;
the time derivatives follow from the chain rule. Plain NumPy, checked against the lesson's formulas.
"""
from math import factorial

import numpy as np
from numpy.polynomial import polynomial as P


def poly_from_boundary(conds, degree):
    """Coefficients c (s = sum_k c_k u^k) that meet every condition (u, derivative order, value)."""
    A = [[factorial(k) / factorial(k - r) * u ** (k - r) if k >= r else 0.0 for k in range(degree + 1)]
         for u, r, _ in conds]
    return np.linalg.solve(np.array(A), np.array([v for _, _, v in conds], dtype=float))


def poly_eval(c, u, order=0):
    """The order-th derivative of the polynomial with coefficients c, at u."""
    return P.polyval(u, P.polyder(c, order))


REST = [(0, 0, 0.0), (1, 0, 1.0), (0, 1, 0.0), (1, 1, 0.0)]      # s(0) = 0, s(1) = 1, at rest at both ends
CUBIC = poly_from_boundary(REST, 3)
QUINTIC = poly_from_boundary(REST + [(0, 2, 0.0), (1, 2, 0.0)], 5)  # ... and zero acceleration at both ends


def cubic(u):
    """Cubic time scaling 3u^2 - 2u^3: s, ds/du, d2s/du2."""
    return poly_eval(CUBIC, u), poly_eval(CUBIC, u, 1), poly_eval(CUBIC, u, 2)


def quintic(u):
    """Quintic (minimum-jerk) time scaling 10u^3 - 15u^4 + 6u^5: s, ds/du, d2s/du2."""
    return poly_eval(QUINTIC, u), poly_eval(QUINTIC, u, 1), poly_eval(QUINTIC, u, 2)


def trapezoid(u, ta=1 / 3):
    """Trapezoidal velocity: accelerate for ta, cruise, decelerate for ta. Returns s, ds/du, d2s/du2."""
    a = 1 / (ta * (1 - ta))                        # the acceleration that makes s(1) = 1
    if u < ta:
        return 0.5 * a * u ** 2, a * u, a
    if u <= 1 - ta:
        return 0.5 * a * ta ** 2 + a * ta * (u - ta), a * ta, 0.0
    return 1 - 0.5 * a * (1 - u) ** 2, a * (1 - u), -a


def joint_reference(shape, t, T, q0, dq):
    """Reference angle, speed and acceleration at time t of the move q0 -> q0 + dq lasting T seconds."""
    s, s1, s2 = shape(min(max(t / T, 0.0), 1.0))
    if not 0.0 < t < T:                            # at rest before and after the move
        s1 = s2 = 0.0
    return q0 + s * dq, s1 / T * dq, s2 / T ** 2 * dq
