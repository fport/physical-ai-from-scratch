"""Lesson 18: physai.estimation against hand values and SciPy, plus properties the lesson relies on."""
import numpy as np
from scipy.linalg import expm, solve_discrete_are

from conftest import assert_build_output
from physai.estimation import (coverage, cv_model, kalman_filter, kf_predict, kf_update, moving_average,
                               steady_state_gain)

DT, R_ENC = 0.002, 1.002e-4


def test_q_matches_van_loan_discretization():
    for dt, q_c in [(0.002, 1.0), (0.01, 3.0), (0.1, 0.2)]:
        F, Q, _, _ = cv_model(dt, q_c, 1.0)
        A, W = np.array([[0.0, 1.0], [0.0, 0.0]]), np.diag([0.0, q_c])
        C = expm(np.block([[-A, W], [np.zeros((2, 2)), A.T]]) * dt)
        assert np.allclose(F, C[2:, 2:].T, atol=1e-15)
        assert np.allclose(Q, C[2:, 2:].T @ C[:2, 2:], rtol=1e-9, atol=0)


def test_update_matches_information_form():
    rng = np.random.default_rng(3)
    F, Q, H, R = cv_model(DT, 1.0, R_ENC)
    for _ in range(20):
        L = rng.normal(size=(2, 2))
        P, x, z = L @ L.T + 1e-3 * np.eye(2), rng.normal(size=2), rng.normal()
        x1, P1 = kf_predict(x, P, F, Q)
        x2, P2, _ = kf_update(x1, P1, z, H, R)
        P_info = np.linalg.inv(np.linalg.inv(P1) + H.T @ H / R[0, 0])
        x_info = P_info @ (np.linalg.inv(P1) @ x1 + H.ravel() * z / R[0, 0])
        assert np.allclose(P2, P_info, rtol=1e-8, atol=1e-14)
        assert np.allclose(x2, x_info, rtol=1e-8, atol=1e-12)


def test_steady_state_gain_matches_dare():
    for q_c in [1e-3, 1e-1, 1, 1e3]:
        F, Q, H, R = cv_model(DT, q_c, R_ENC)
        P = solve_discrete_are(F.T, H.T, Q, R)
        K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
        assert np.allclose(steady_state_gain(F, Q, H, R)[0], K, rtol=1e-8)


def test_covariance_stays_symmetric_positive_definite_and_coverage_is_honest():
    rng = np.random.default_rng(0)
    n, q_c, r = 20000, 1.0, R_ENC
    a = rng.normal(0, np.sqrt(q_c / DT), n)               # data drawn from the filter's own model
    v = np.cumsum(a) * DT
    q = np.cumsum(v) * DT
    z = q + rng.normal(0, np.sqrt(r), n)
    xs, Ps, K = kalman_filter(z, DT, q_c, r)
    assert np.abs(Ps - Ps.transpose(0, 2, 1)).max() <= 1e-12 * np.abs(Ps).max()
    assert np.all(np.linalg.eigvalsh((Ps + Ps.transpose(0, 2, 1)) / 2) > 0)
    cov = coverage(xs[500:, 0] - q[500:], np.sqrt(Ps[500:, 0, 0]))
    assert 0.93 < cov < 0.97                               # about 95 % when the model is true


def test_only_the_ratio_q_over_r_sets_the_estimate():
    z = np.sin(np.linspace(0, 6, 2000)) + np.random.default_rng(1).normal(0, 0.01, 2000)
    xs1, Ps1, K1 = kalman_filter(z, DT, 1.0, R_ENC)
    xs2, Ps2, K2 = kalman_filter(z, DT, 50.0, 50 * R_ENC, P0=50 * np.diag([R_ENC, 1.0]))
    assert np.allclose(xs1, xs2, atol=1e-12) and np.allclose(K1, K2) and np.allclose(Ps2, 50 * Ps1)


def test_moving_average_is_causal_mean():
    x = np.random.default_rng(2).normal(size=50)
    y = moving_average(x, 5)
    assert np.allclose(y, [x[max(0, i - 4):i + 1].mean() for i in range(50)])


def test_build_script_reproduces_its_output():
    assert_build_output("18-kalman-filter")
