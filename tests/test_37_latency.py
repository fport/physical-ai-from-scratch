"""Lesson 37: physai.sim2real delay tools against MuJoCo and SciPy, plus properties the lesson relies on."""
import mujoco
import numpy as np
import pytest
from scipy import optimize, signal

from conftest import assert_build_output
from physai.control import simulate_joint
from physai.sim2real import DelayedPD, DelayLine, PredictivePD, is_stable, overshoot, pd_delay_margin

DT = 0.001


def joint(inertia):
    """A 1-DoF joint with total inertia `inertia`: a point mass at 0.3 m plus 1e-4 kg m^2 of its own."""
    return mujoco.MjModel.from_xml_string(f"""
<mujoco>
  <option timestep="{DT}" gravity="0 0 0"/>
  <worldbody><body><joint name="j" axis="0 1 0"/>
    <inertial pos="0.3 0 0" mass="{(inertia - 1e-4) / 0.09}" diaginertia="1e-4 1e-4 1e-4"/></body></worldbody>
  <actuator><motor joint="j"/></actuator>
</mujoco>""")


def test_delay_line_returns_value_from_n_ticks_ago():
    line = DelayLine(5, -1)
    out = [line(k) for k in range(20)]
    assert out == [-1] * 5 + list(range(15))


@pytest.mark.parametrize("seed", range(3))
def test_delay_margin_matches_scipy_frequency_response(seed):
    rng = np.random.default_rng(seed)
    inertia, kp, kd = rng.uniform(0.01, 1), rng.uniform(1, 200), rng.uniform(0.05, 10)
    w_c, pm, _ = pd_delay_margin(inertia, kp, kd)
    loop = signal.TransferFunction([kd, kp], [inertia, 0, 0])
    w = optimize.brentq(lambda w: abs(signal.freqresp(loop, [w])[1][0]) - 1, 1e-3, 1e5, xtol=1e-14)
    phase = np.angle(signal.freqresp(loop, [w])[1][0])
    assert abs(w - w_c) < 1e-8 * w_c and abs((np.pi + phase) - pm) < 1e-9


@pytest.mark.parametrize("seed", range(3))
def test_delayed_pd_matches_mj_step(seed):
    rng = np.random.default_rng(10 + seed)
    inertia, kp, nd = rng.uniform(0.05, 0.3), rng.uniform(5, 50), int(rng.integers(0, 30))
    kd = 2 * 0.7 * np.sqrt(kp * inertia)
    m = joint(inertia)
    d = mujoco.MjData(m)
    c_mj, theirs = DelayedPD(kp, kd, 0.5, DT, nd=nd), []
    for _ in range(1500):
        d.ctrl[0] = c_mj(d.qpos[0], d.qvel[0])
        mujoco.mj_step(m, d)
        theirs.append(d.qpos[0])
    ours = simulate_joint(DelayedPD(kp, kd, 0.5, DT, nd=nd), 0.0, inertia, DT, 1500)
    assert np.abs(ours - np.array(theirs)).max() < 1e-10


def test_boundary_sits_at_the_predicted_delay_margin():
    inertia, kp = 0.0901, 20.0
    kd = 2 * 0.7 * np.sqrt(kp * inertia)
    d_max = pd_delay_margin(inertia, kp, kd)[2]
    run = lambda nd: simulate_joint(DelayedPD(kp, kd, 0.5, DT, nd=nd), 0.0, inertia, DT, 10000)  # noqa: E731
    assert is_stable(run(int(np.floor(d_max / DT)) - 1), 0.5)
    assert not is_stable(run(int(np.ceil(d_max / DT))), 0.5)


def test_exact_prediction_removes_the_delay():
    inertia, kp = 0.0901, 20.0
    kd = 2 * 0.7 * np.sqrt(kp * inertia)
    free = simulate_joint(DelayedPD(kp, kd, 0.5, DT), 0.0, inertia, DT, 3000)
    pred = simulate_joint(PredictivePD(kp, kd, 0.5, DT, 30, inertia, 30), 0.0, inertia, DT, 3000)
    late = simulate_joint(DelayedPD(kp, kd, 0.5, DT, nd=30), 0.0, inertia, DT, 3000)
    assert np.abs(pred - free).max() < 1e-12 and overshoot(late, 0.5) > overshoot(free, 0.5) + 5


def test_mujoco_sensor_delay_plus_one_stale_tick():
    """MuJoCo's sensor history with delay (n - 1) ms, read after mj_step, equals our n-tick delay."""
    inertia, kp = 0.0901, 20.0
    kd = 2 * 0.7 * np.sqrt(kp * inertia)
    m = mujoco.MjModel.from_xml_string(f"""
<mujoco><option timestep="{DT}" gravity="0 0 0"/>
  <worldbody><body><joint name="j" axis="0 1 0"/>
    <inertial pos="0.3 0 0" mass="1" diaginertia="1e-4 1e-4 1e-4"/></body></worldbody>
  <actuator><motor joint="j"/></actuator>
  <sensor><jointpos joint="j" delay="0.019" nsample="19"/><jointvel joint="j" delay="0.019" nsample="19"/></sensor>
</mujoco>""")
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    qs = []
    for _ in range(2000):
        d.ctrl[0] = kp * (0.5 - d.sensordata[0]) - kd * d.sensordata[1]
        mujoco.mj_step(m, d)
        qs.append(d.qpos[0])
    ours = simulate_joint(DelayedPD(kp, kd, 0.5, DT, nd=20), 0.0, inertia, DT, 2000)
    assert np.abs(ours - np.array(qs)).max() < 1e-10


def test_build_script_reproduces_its_output():
    assert_build_output("37-latency")
