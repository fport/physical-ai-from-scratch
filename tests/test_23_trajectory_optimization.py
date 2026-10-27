"""Lesson 23: physai.optimal (collocation, shooting) against finite differences and MuJoCo."""
import mujoco
import numpy as np
from scipy.optimize import approx_fprime

from conftest import assert_build_output
from physai.optimal import (collocate, effort_cost, pendulum_f, pendulum_jac, rk4_rollout, trapezoid_defects,
                            trapezoid_jacobian)

X0, XG = np.array([0.0, 0.0]), np.array([np.pi, 0.0])


def test_cost_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    for N in (5, 20):
        U = rng.uniform(-5, 5, N + 1)
        g_fd = approx_fprime(U, lambda u: effort_cost(u, 4.0 / N, 0.05)[0], 1e-6)
        assert np.abs(effort_cost(U, 4.0 / N, 0.05)[1] - g_fd).max() < 1e-4


def test_defect_jacobian_matches_finite_differences():
    rng = np.random.default_rng(1)
    X, U = rng.normal(size=(9, 2)), rng.uniform(-5, 5, 9)
    z = np.r_[X.ravel(), U]
    fd = approx_fprime(z, lambda z: trapezoid_defects(pendulum_f, z[:18].reshape(9, 2), z[18:], 0.5).ravel(), 1e-7)
    assert np.abs(trapezoid_jacobian(pendulum_jac, X, U, 0.5) - fd).max() < 1e-5


def test_rk4_matches_mujoco_pendulum():
    m = mujoco.MjModel.from_xml_string("""<mujoco><option timestep="0.001" integrator="RK4"/><worldbody>
      <body><joint name="h" type="hinge" axis="0 1 0" damping="0.1"/><inertial pos="0 0 -1" mass="1" diaginertia="1e-9 1e-9 1e-9"/>
      </body></worldbody><actuator><motor joint="h" ctrlrange="-5 5" ctrllimited="true"/></actuator></mujoco>""")
    d = mujoco.MjData(m)
    U = np.array([2.0, -3.0, 4.0, 1.0, 0.0])
    tk = np.linspace(0, 2.0, 5)
    for _ in range(2000):
        d.ctrl[0] = np.interp(d.time + 0.0005, tk, U)
        mujoco.mj_step(m, d)
    assert np.abs(rk4_rollout(pendulum_f, X0, U, 2.0)[0] - [d.qpos[0], d.qvel[0]]).max() < 1e-5


def test_trapezoid_defects_shrink_like_h_cubed():
    def max_defect(N):
        U = np.full(N + 1, 1.0)
        Xk = [np.array([0.5, 0.0])]
        for k in range(N):
            Xk.append(rk4_rollout(pendulum_f, Xk[-1], U[k:k + 2], 2.0 / N, sub=100)[0])
        return np.abs(trapezoid_defects(pendulum_f, np.array(Xk), U, 2.0 / N)).max()
    ratio = max_defect(20) / max_defect(40)
    assert 6.0 < ratio < 10.0


def test_collocation_converges_and_satisfies_its_constraints():
    r, X, U, res = collocate(pendulum_f, pendulum_jac, X0, XG, 20, 4.0, 5.0, 0.05, np.zeros((21, 2)), np.zeros(21))
    assert r.success and res < 1e-8 and np.abs(U).max() <= 5.0 + 1e-9
    assert np.allclose(X[0], X0) and np.allclose(X[-1], XG)


def test_build_script_reproduces_its_output():
    assert_build_output("23-trajectory-optimization")
