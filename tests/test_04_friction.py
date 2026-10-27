"""Lesson 04: physai.sim friction against MuJoCo's pyramidal rows and contact forces, plus Coulomb's law."""
import mujoco
import numpy as np
from scipy.optimize import linprog

from conftest import assert_build_output
from physai.sim import pyramid_rows, pyramid_to_force, ramp_slip, tangential_limit

RNG = np.random.default_rng(4)


def resting_box(mu, cone="pyramidal", tilt_deg=15.0, steps=200):
    t = np.radians(tilt_deg)
    m = mujoco.MjModel.from_xml_string(f"""<mujoco><option cone="{cone}" gravity="{9.81 * np.sin(t)} 0 {-9.81 * np.cos(t)}"/>
      <worldbody><geom type="plane" size="5 5 .1" friction="{mu} 0.005 0.0001"/>
      <body pos="0 0 .04" quat="0.95 0 0 0.31"><freejoint/>
      <geom type="box" size=".04 .03 .02" mass="0.7" friction="{mu} 0.005 0.0001"/></body></worldbody></mujoco>""")
    d = mujoco.MjData(m)
    for _ in range(steps):
        mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)
    return m, d


def test_pyramid_rows_and_forces_match_mujoco():
    for mu in (0.3, 0.8, 1.2):
        m, d = resting_box(mu)
        assert d.ncon > 0
        J = d.efc_J.reshape(d.nefc, m.nv)
        for i in range(d.ncon):
            c = d.contact[i]
            jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mujoco.mj_jac(m, d, jp, jr, c.pos, m.geom_bodyid[c.geom2])
            Jc = c.frame.reshape(3, 3) @ jp
            a = c.efc_address
            assert np.allclose(pyramid_rows(Jc[0], Jc[1:], c.friction[:2]), J[a:a + 4], atol=1e-14)
            f = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, f)
            assert np.allclose(pyramid_to_force(d.efc_force[a:a + 4], c.friction[:2]), f[:3], atol=1e-12)


def test_tangential_limit_matches_linear_program():
    for mu, phi in zip(RNG.uniform(0.1, 1.5, 10), RNG.uniform(0, 2 * np.pi, 10)):
        e = np.array([[1, mu, 0], [1, -mu, 0], [1, 0, mu], [1, 0, -mu]])
        res = linprog(-(e[:, 1] * np.cos(phi) + e[:, 2] * np.sin(phi)), bounds=[(0, None)] * 4,
                      A_eq=np.array([e[:, 0], -e[:, 1] * np.sin(phi) + e[:, 2] * np.cos(phi)]), b_eq=[1.0, 0.0])
        assert np.isclose(-res.fun, tangential_limit(phi, mu, "pyramidal"), rtol=1e-8)
        assert tangential_limit(phi, mu, "pyramidal") <= tangential_limit(phi, mu, "elliptic") + 1e-15


def test_slip_angle_is_arctan_mu_and_mass_free():
    for mu in RNG.uniform(0.1, 1.5, 10):
        crit = np.arctan(mu)
        assert not ramp_slip(crit - 1e-9, mu)[0] and ramp_slip(crit + 1e-9, mu)[0]
        assert ramp_slip(crit + 0.1, mu, g=9.81)[1] > 0


def test_holding_box_uses_exactly_the_friction_it_needs():
    m, d = resting_box(0.8, cone="elliptic", tilt_deg=20.0, steps=400)
    total = np.zeros(3)
    for i in range(d.ncon):
        f = np.zeros(6)
        mujoco.mj_contactForce(m, d, i, f)
        total += d.contact[i].frame.reshape(3, 3).T @ f[:3]
    assert np.isclose(np.hypot(total[0], total[1]) / total[2], np.tan(np.radians(20.0)), rtol=1e-4)


def test_build_script_reproduces_its_output():
    assert_build_output("04-friction")
