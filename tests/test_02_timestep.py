"""Lesson 02: physai.sim time steppers against MuJoCo's integrators, plus the stability facts the lesson uses."""
import mujoco
import numpy as np
import pytest

from conftest import assert_build_output
from physai.sim import explicit_euler, first_overlap, rk4, semi_implicit_euler, step_growth

RNG = np.random.default_rng(2)


def mj_spring(omega, h, integrator, n, x0=0.1):
    m = mujoco.MjModel.from_xml_string(f"""<mujoco><option timestep="{h}" gravity="0 0 0" integrator="{integrator}"/>
  <worldbody><body><joint type="slide" axis="1 0 0" stiffness="{omega**2}"/>
  <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/></body></worldbody></mujoco>""")
    d = mujoco.MjData(m)
    d.qpos[0] = x0
    xs = []
    for _ in range(n):
        mujoco.mj_step(m, d)
        xs.append(d.qpos[0])
    return np.array(xs)


def ours(step, omega, h, n, x0=0.1):
    x, v, xs = x0, 0.0, []
    for _ in range(n):
        x, v = step(x, v, lambda q, u: -omega**2 * q, h)
        xs.append(x)
    return np.array(xs)


@pytest.mark.parametrize("step,integrator", [(semi_implicit_euler, "Euler"), (rk4, "RK4")])
def test_steppers_match_mujoco(step, integrator):
    for _ in range(5):
        omega, hw = RNG.uniform(1, 30), RNG.uniform(0.05, 1.5)
        assert np.allclose(ours(step, omega, hw / omega, 200), mj_spring(omega, hw / omega, integrator, 200), atol=1e-12)


def test_explicit_euler_energy_grows_by_one_plus_h2w2():
    omega, h, n = 3.0, 0.02, 300
    x, v = 1.0, 0.0
    for _ in range(n):
        x, v = explicit_euler(x, v, lambda q, u: -omega**2 * q, h)
    assert np.isclose((v * v + omega**2 * x * x) / omega**2, (1 + (h * omega) ** 2) ** n, rtol=1e-10)


def test_semi_implicit_limit_is_h_omega_two():
    for hw in RNG.uniform(0.01, 1.99, 20):
        assert abs(step_growth(semi_implicit_euler, hw, 1.0) - 1) < 1e-9
    for hw in RNG.uniform(2.01, 5, 20):
        assert step_growth(semi_implicit_euler, hw, 1.0) > 1
    assert step_growth(explicit_euler, 0.01, 1.0) > 1      # explicit Euler grows at any step


def test_first_overlap_matches_mujoco():
    xml = """<mujoco><worldbody><geom name="table" type="box" size="0.12 0.12 0.005" pos="0 0 0.295"/>
      <body name="cube" pos="0 0 1.30"><freejoint/><geom type="box" size="0.025 0.025 0.025" mass="0.1"/></body>
      </worldbody></mujoco>"""
    for dt in (0.0007, 0.0035, 0.009, 0.013, 0.027):
        m = mujoco.MjModel.from_xml_string(xml)
        m.opt.timestep = dt
        d, mj = mujoco.MjData(m), None
        while d.time < 1.0 and mj is None:
            mujoco.mj_step(m, d)
            z = d.body("cube").xpos[2]
            mj = z if (z - 0.025 < 0.300 and z + 0.025 > 0.290) else None
        got = first_overlap(1.30, dt, top=0.300, bottom=0.290, half=0.025)
        assert (got is None and mj is None) or abs(got - mj) < 1e-12


def test_build_script_reproduces_its_output():
    assert_build_output("02-timestep")
