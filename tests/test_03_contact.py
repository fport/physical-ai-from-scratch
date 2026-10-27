"""Lesson 03: physai.sim soft contact against MuJoCo's constraint data, plus properties the lesson relies on."""
import mujoco
import numpy as np

from conftest import assert_build_output
from physai.sim import contact_accel, impedance, resting_penetration, semi_implicit_euler, solref_kb

H, R = 0.002, 0.05
RNG = np.random.default_rng(3)


def sphere(solref, solimp, z0, h=H):
    sr, si = " ".join(map(str, solref)), " ".join(map(str, solimp))
    m = mujoco.MjModel.from_xml_string(f"""<mujoco><option timestep="{h}"/><worldbody>
      <geom type="plane" size="1 1 .1" solref="{sr}" solimp="{si}" condim="1"/>
      <body pos="0 0 {z0}"><freejoint/><geom type="sphere" size="{R}" mass="1" solref="{sr}" solimp="{si}" condim="1"/>
      </body></worldbody></mujoco>""")
    return m, mujoco.MjData(m)


def test_impedance_and_spring_match_efc_kbip():
    for _ in range(20):
        solref = (RNG.uniform(0.001, 0.1), RNG.uniform(0.2, 2))
        solimp = (RNG.uniform(0.5, 0.9), RNG.uniform(0.91, 0.99), RNG.uniform(0.0005, 0.005), RNG.uniform(0.2, 0.8),
                  RNG.uniform(1, 4))
        r = RNG.uniform(0, 1.2 * solimp[2])
        m, d = sphere(solref, solimp, R - r)
        mujoco.mj_forward(m, d)
        k, b = solref_kb(solref, solimp[1], H)
        assert np.allclose(d.efc_KBIP[0, :3], [k, b, impedance(r, solimp)], rtol=1e-9)


def test_one_contact_drop_matches_mj_step():
    for solref in ((0.02, 1.0), (0.01, 0.5), (0.05, 0.2)):
        solimp = (0.9, 0.95, 0.001, 0.5, 2)
        m, d = sphere(solref, solimp, R + 0.1)
        z, v = R + 0.1, 0.0
        for _ in range(600):
            z, v = semi_implicit_euler(z, v, lambda x, u: contact_accel(x - R, u, -9.81, solref, solimp, H), H)
            mujoco.mj_step(m, d)
            assert abs(z - d.qpos[2]) < 1e-9


def test_resting_penetration_matches_resting_sphere():
    solimp = (0.9, 0.95, 0.001, 0.5, 2)
    for tc in (0.003, 0.008, 0.02, 0.1):
        m, d = sphere((tc, 1), solimp, R)
        for _ in range(1500):
            mujoco.mj_step(m, d)
        assert abs(resting_penetration(9.81, (tc, 1), solimp, H) - (R - d.qpos[2])) < 1e-8


def test_penetration_scales_with_one_minus_d_and_timeconst_squared():
    r = {dd: resting_penetration(9.81, (0.02, 1), (dd, dd, 0.001, 0.5, 2), H) for dd in (0.5, 0.9, 0.99)}
    assert np.isclose(r[0.5] / r[0.9], 0.5 / 0.1, rtol=1e-9)     # constant d: proportional to 1 - d (Sweep 2)
    deep = [resting_penetration(9.81, (tc, 1), (0.9, 0.95, 0.001, 0.5, 2), H) for tc in (0.05, 0.1)]
    assert np.isclose(deep[1] / deep[0], 4.0, rtol=1e-9)    # beyond the width d = dw: twice as soft, four times as deep


def test_build_script_reproduces_its_output():
    assert_build_output("03-contact")
