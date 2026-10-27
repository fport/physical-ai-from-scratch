"""Lesson 01: physai.sim mass properties and drag against MuJoCo, plus properties the lesson relies on."""
import mujoco
import numpy as np
import pytest

from conftest import assert_build_output
from physai.sim import combine_parts, inertia_box, quadratic_drag, semi_implicit_euler, solid_inertia

RNG = np.random.default_rng(1)


def one_body(geoms):
    return mujoco.MjModel.from_xml_string(f"<mujoco><worldbody><body><freejoint/>{geoms}</body></worldbody></mujoco>")


@pytest.mark.parametrize("kind,n", [("box", 3), ("sphere", 1), ("cylinder", 2), ("capsule", 2)])
def test_solid_inertia_matches_compiler(kind, n):
    for _ in range(10):
        size, rho = RNG.uniform(0.01, 0.3, n), RNG.uniform(100, 5000)
        m = one_body(f'<geom type="{kind}" size="{" ".join(map(str, size))}" density="{rho}"/>')
        mass, inertia = solid_inertia(kind, size, rho)
        assert np.isclose(mass, m.body_mass[1], rtol=1e-12)
        assert np.allclose(inertia, m.body_inertia[1], rtol=1e-10)


def test_combine_parts_matches_compiler():
    for _ in range(10):
        pos, half, r = RNG.uniform(-0.2, 0.2, (2, 3)), RNG.uniform(0.02, 0.1, 3), RNG.uniform(0.02, 0.1)
        m = one_body(f'<geom type="box" size="{" ".join(map(str, half))}" pos="{" ".join(map(str, pos[0]))}" '
                     f'density="1000"/><geom type="sphere" size="{r}" pos="{" ".join(map(str, pos[1]))}" density="1000"/>')
        parts = [solid_inertia("box", half, 1000), solid_inertia("sphere", [r], 1000)]
        mass, com, inertia = combine_parts([p[0] for p in parts], pos, [p[1] for p in parts])
        assert np.isclose(mass, m.body_mass[1], rtol=1e-12)
        assert np.allclose(com, m.body_ipos[1], atol=1e-12)
        assert np.allclose(np.sort(np.linalg.eigvalsh(inertia)), np.sort(m.body_inertia[1]), rtol=1e-8)


def test_inertia_box_inverts_the_box_formula():
    sides = np.array([0.3, 0.12, 0.05])
    mass, inertia = solid_inertia("box", sides / 2, 800)
    assert np.allclose(inertia_box(mass, inertia), sides, rtol=1e-12)


def test_drag_matches_qfrc_fluid():
    m = mujoco.MjModel.from_xml_string('<mujoco><option density="1.2"/><worldbody><body><freejoint/>'
                                       '<geom type="box" size=".1 .05 .02" mass="0.3"/></body></worldbody></mujoco>')
    d = mujoco.MjData(m)
    box = inertia_box(m.body_mass[1], m.body_inertia[1])
    for v in RNG.uniform(-3, 3, (10, 3)):
        d.qvel[:] = np.r_[v, 0, 0, 0]
        mujoco.mj_forward(m, d)
        assert np.allclose(quadratic_drag(v, box, 1.2), d.qfrc_fluid[:3], rtol=1e-10, atol=1e-14)


def test_drag_height_loss_scales_as_one_over_mass():
    box = np.full(3, 0.2)                              # the Experiment's 20 cm cube
    lost = []
    for mass in (0.1, 1.0):
        z = {0.0: 0.9, 1.2: 0.9}
        for rho in z:
            x, v = 0.9, 0.0
            for _ in range(140):
                x, v = semi_implicit_euler(x, v, lambda q, u: -9.81 + quadratic_drag(np.array([0, 0, u]), box, rho)[2] / mass, 0.002)
            z[rho] = x
        lost.append(z[1.2] - z[0.0])
    assert 9.5 < lost[0] / lost[1] < 10.5                 # ten times the mass, a tenth of the height lost


def test_build_script_reproduces_its_output():
    assert_build_output("01-world-as-text")
