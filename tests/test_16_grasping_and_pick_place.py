"""Lesson 16: physai.grasp against MuJoCo contact forces and SciPy, plus properties the lesson relies on."""
import mujoco
import numpy as np
from scipy.stats import binomtest

from conftest import assert_build_output
from physai.grasp import (PickMachine, cone_ratio, cone_use, max_mass, minjerk, pick_outcome, squeeze_from_cap,
                          wilson)

BOX_ON_PLANE = """
<mujoco>
  <option timestep="0.002" cone="elliptic"/>
  <worldbody>
    <geom type="plane" size="1 1 0.1" friction="{mu} 0.005 0.0001"/>
    <body pos="0 0 0.02">
      <freejoint/>
      <geom type="box" size="0.02 0.02 0.02" mass="0.2" friction="{mu} 0.005 0.0001"/>
    </body>
  </worldbody>
</mujoco>"""


def test_cone_use_matches_mujoco_sliding_threshold():
    """Push a box sideways: MuJoCo's contact forces stay inside the cone, and reach its edge only when it slides."""
    mu = 0.5
    for push, slides in [(0.5, False), (1.5, True)]:           # mu m g = 0.981 N
        m = mujoco.MjModel.from_xml_string(BOX_ON_PLANE.format(mu=mu))
        d = mujoco.MjData(m)
        f6, use = np.zeros(6), []
        for _ in range(500):
            d.xfrc_applied[1, 0] = push
            mujoco.mj_step(m, d)
            tot = np.zeros(3)
            for k in range(d.ncon):
                mujoco.mj_contactForce(m, d, k, f6)
                assert f6[0] == 0 or cone_use(f6[:3], mu) <= 1 + 1e-6      # every pushing contact inside its cone
                tot += d.contact[k].frame.reshape(3, 3).T @ f6[:3]
            if tot[2] > 0:
                use.append(np.hypot(tot[0], tot[1]) / (mu * tot[2]))      # whole box: friction / (mu N)
        assert (d.qvel[0] > 0.1) == slides
        assert (np.median(use[-100:]) > 0.99) == slides


def test_ratio_mass_and_squeeze_formulas():
    assert np.isclose(cone_ratio(0.3, [8.0, 8.0], 0.4), 0.3 * 16 / (0.4 * 9.81))
    assert np.isclose(cone_ratio(0.05, 12.74, max_mass(0.05, 12.74)), 1.0)       # max_mass is the ratio-1 mass
    assert np.isclose(squeeze_from_cap(0.5, 0.074), 2 * 0.5 / 0.074)


def test_wilson_matches_scipy():
    for n in (10, 20, 50):
        for k in range(n + 1):
            ci = binomtest(k, n).proportion_ci(method="wilson")
            assert np.allclose(wilson(k, n, z=1.959963984540054), (ci.low, ci.high), atol=1e-12)


def test_minjerk_and_machine_sequence():
    assert minjerk(0.0) == 0.0 and minjerk(1.0) == 1.0 and minjerk(-1) == 0.0 and minjerk(2) == 1.0
    h = 1e-6
    assert abs(minjerk(h) - minjerk(0)) / h < 1e-9 and abs(minjerk(1) - minjerk(1 - h)) / h < 1e-9
    dur = dict(settle=0.3, pre=1.2, down=0.8, close_ramp=0.3, lift=1.0, hold=0.8)
    for touch, close_s in [(True, 0.4), (False, float("nan"))]:
        sm, seen, t = PickMachine(dur, close_timeout=4.0), [], 0.0
        while sm.state != "done" and t < 20:
            t = round(t + 0.002, 6)
            sm.update(t, both_touch=touch, jaw_speed=0.0)
            if not seen or seen[-1] != sm.state:
                seen.append(sm.state)
        assert seen == ["settle", "pre", "down", "close", "lift", "hold", "done"]
        assert np.isclose(sm.close_s, close_s, atol=0.003) if touch else np.isnan(sm.close_s)
    assert pick_outcome(0.05, True, True) == "held"
    assert pick_outcome(0.0, False, True) == "dropped" and pick_outcome(0.0, False, False) == "not closed"


def test_build_script_reproduces_its_output():
    assert_build_output("16-grasping-and-pick-place")
