"""Lesson 24: physai.optimal (linearization, LQR, MPPI) against SciPy, MuJoCo and known properties."""
import mujoco
import numpy as np
from scipy.linalg import solve_discrete_are

from conftest import assert_build_output
from physai.optimal import dlqr, linearize, lqr_finite, mppi_update


def random_system(rng, n=4, m=2):
    A = rng.normal(size=(n, n)) * 0.5 + np.eye(n)
    B = rng.normal(size=(n, m))
    return A, B, np.diag(rng.uniform(0.5, 2, n)), np.diag(rng.uniform(0.1, 1, m))


def test_dlqr_matches_solve_discrete_are():
    rng = np.random.default_rng(0)
    for _ in range(5):
        A, B, Q, R = random_system(rng)
        K, P, _ = dlqr(A, B, Q, R)
        P_ref = solve_discrete_are(A, B, Q, R)
        assert np.abs(P - P_ref).max() <= 1e-8 * np.abs(P_ref).max()
        assert np.abs(np.linalg.eigvals(A - B @ K)).max() < 1.0


def test_finite_horizon_gain_approaches_infinite_horizon():
    A, B, Q, R = random_system(np.random.default_rng(1))
    K = dlqr(A, B, Q, R)[0]
    gains, _ = lqr_finite(A, B, Q, R, Q, 400)
    assert np.abs(gains[0] - K).max() < 1e-8 and len(gains) == 400


def test_linearize_matches_linear_map_and_mjd_transitionFD():
    A0, B0 = np.array([[1.0, 0.1], [-0.2, 0.9]]), np.array([[0.0], [0.1]])
    A, B = linearize(lambda x, u: A0 @ x + B0 @ u, np.ones(2), np.zeros(1))
    assert np.allclose(A, A0, atol=1e-9) and np.allclose(B, B0, atol=1e-9)
    m = mujoco.MjModel.from_xml_string("""<mujoco><option timestep="0.01" integrator="Euler"/><worldbody>
      <body><joint name="s" type="slide" axis="1 0 0"/><geom type="box" size="0.1 0.1 0.1" mass="1"/>
      <body><joint type="hinge" axis="0 1 0"/><geom type="capsule" fromto="0 0 0 0 0 0.6" size="0.02" mass="0.2"/>
      </body></body></worldbody><actuator><motor joint="s"/></actuator></mujoco>""")
    d = mujoco.MjData(m)

    def step(x, u):
        mujoco.mj_resetData(m, d)
        d.qpos[:], d.qvel[:], d.ctrl[:] = x[:2], x[2:], u
        mujoco.mj_step(m, d)
        return np.r_[d.qpos, d.qvel]

    A, B = linearize(step, np.zeros(4), np.zeros(1))
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    A_mj, B_mj = np.zeros((4, 4)), np.zeros((4, 1))
    mujoco.mjd_transitionFD(m, d, 1e-6, True, A_mj, B_mj, None, None)
    assert np.abs(A - A_mj).max() < 1e-8 and np.abs(B - B_mj).max() < 1e-8


def test_mppi_weights():
    rng = np.random.default_rng(2)
    samples, costs = rng.normal(size=(64, 10)), rng.uniform(0, 5, 64)
    U, w = mppi_update(samples, costs, 0.5)
    assert np.isclose(w.sum(), 1.0) and np.allclose(U, w @ samples)
    assert np.allclose(mppi_update(samples, costs + 100.0, 0.5)[0], U)      # a constant shift changes nothing
    assert np.allclose(mppi_update(samples, costs, 1e-6)[0], samples[np.argmin(costs)])


def test_build_script_reproduces_its_output():
    assert_build_output("24-lqr-and-mpc")
