"""Lesson 17 · Sensors, noise and filtering.

The SO-101 shoulder_pan joint follows a smooth trajectory in MuJoCo. We read its
position through a simulated encoder (12-bit quantization + white noise), then
filter the reading with a moving average and a first-order low-pass filter.

Sweep: filter window / cutoff.  Measured: position RMS error, delay (ms),
velocity RMS error when velocity is obtained by differencing the filtered signal.

Also shown: MuJoCo's sensor `noise` attribute adds no noise, an accelerometer at
rest reads gravity, and a constant gyro bias integrates into angle drift.

Run:  python experiment.py
"""
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.menagerie import path  # noqa: E402

DT = 0.002                    # 500 Hz sampling = simulation timestep
T_END = 10.0
COUNTS = 4096                 # 12-bit encoder: counts per revolution (modelling choice)
SIGMA = 0.01                  # white measurement noise (rad), teaching value
GYRO_BIAS = 0.02              # rad/s, teaching value
GYRO_SIGMA = 0.05             # rad/s, teaching value
SEED = 0


def build_model():
    spec = mujoco.MjSpec.from_file(path("robotstudio_so101"))
    spec.option.timestep = DT
    grip = spec.body("gripper")
    grip.add_site(name="imu", pos=[0, 0, -0.03])
    for name, typ, obj in [("q_pan", mujoco.mjtSensor.mjSENS_JOINTPOS, "shoulder_pan"),
                           ("v_pan", mujoco.mjtSensor.mjSENS_JOINTVEL, "shoulder_pan")]:
        s = spec.add_sensor(name=name, type=typ, objname=obj, objtype=mujoco.mjtObj.mjOBJ_JOINT)
        s.noise = 0.01        # <- metadata only, see the check below
    spec.add_sensor(name="acc", type=mujoco.mjtSensor.mjSENS_ACCELEROMETER, objname="imu",
                    objtype=mujoco.mjtObj.mjOBJ_SITE)
    spec.add_sensor(name="gyro", type=mujoco.mjtSensor.mjSENS_GYRO, objname="imu",
                    objtype=mujoco.mjtObj.mjOBJ_SITE)
    return spec.compile()


def reference(t, k=1.0):
    # k scales both frequencies (used in the Break-it sweep)
    return 0.5 * np.sin(2 * np.pi * 0.2 * k * t) + 0.2 * np.sin(2 * np.pi * 0.5 * k * t)


def simulate(m, moving=True, t_end=T_END, k=1.0):
    d = mujoco.MjData(m)
    n = int(round(t_end / DT))
    adr = {k: m.sensor_adr[m.sensor(k).id] for k in ["q_pan", "v_pan", "acc", "gyro"]}
    q, v, qs = np.zeros(n), np.zeros(n), np.zeros(n)
    acc, gyro = np.zeros((n, 3)), np.zeros((n, 3))
    ctrl = np.array([0.0, -0.5, 0.8, 0.6, 0.0, 0.0])
    d.qpos[:6] = ctrl
    mujoco.mj_forward(m, d)
    for i in range(n):
        if moving:
            ctrl[0] = reference(d.time, k)
        d.ctrl[:] = ctrl
        q[i], v[i] = d.qpos[0], d.qvel[0]      # state at the start of this step
        mujoco.mj_step(m, d)                   # sensors are computed from that same state
        qs[i] = d.sensordata[adr["q_pan"]]
        acc[i] = d.sensordata[adr["acc"]:adr["acc"] + 3]
        gyro[i] = d.sensordata[adr["gyro"]:adr["gyro"] + 3]
    return q, v, qs, acc, gyro


def encoder(q, rng):
    step = 2 * np.pi / COUNTS
    return np.round(q / step) * step + rng.normal(0, SIGMA, q.shape)


def moving_average(x, n):
    # causal: mean of the last n samples (fewer at the start)
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.empty_like(x)
    for i in range(len(x)):
        lo = max(0, i + 1 - n)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def low_pass(x, fc):
    tau = 1.0 / (2 * np.pi * fc)
    a = DT / (tau + DT)
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = y[i - 1] + a * (x[i] - y[i - 1])
    return y


def delay_ms(y, ref, max_lag=400):
    """Lag (ms) that best aligns the filtered signal with the true one."""
    k0 = 500                                          # skip the start-up transient
    errs = [np.mean((y[k0 + L:] - ref[k0:len(ref) - L]) ** 2) for L in range(max_lag)]
    return 1000 * DT * int(np.argmin(errs))


def score(y, q, v):
    k0 = 500
    e_pos = 1000 * np.sqrt(np.mean((y[k0:] - q[k0:]) ** 2))           # mrad
    vel = np.gradient(y, DT)
    e_vel = np.sqrt(np.mean((vel[k0:] - v[k0:]) ** 2))                 # rad/s
    return e_pos, delay_ms(y, q), e_vel


def main():
    print(f"mujoco {mujoco.__version__} · numpy {np.__version__} · seed {SEED} · dt {DT*1000:.0f} ms")
    m = build_model()
    rng = np.random.default_rng(SEED)

    # 1. MuJoCo sensors: the noise attribute is metadata
    q, v, qs, acc, gyro = simulate(m)
    print(f"\n[1] jointpos sensor with noise=0.01: max |sensordata - qpos| = {np.max(np.abs(qs - q)):.1e} rad")
    print("    -> MuJoCo does not add noise; we add it ourselves.")

    # 2. Encoder model
    z = encoder(q, rng)
    print(f"\n[2] encoder: {COUNTS} counts/rev -> step {1000*2*np.pi/COUNTS:.3f} mrad; "
          f"added white noise sigma {SIGMA*1000:.0f} mrad")
    raw_p, _, raw_v = score(z, q, v)
    print(f"    raw reading: position RMS {raw_p:.2f} mrad, finite-difference velocity RMS {raw_v:.2f} rad/s "
          f"(true velocity RMS {np.sqrt(np.mean(v**2)):.2f} rad/s)")

    # 3. Sweep: moving average window
    print("\n[3] moving average (causal), window N")
    print(f"{'N':>5} {'theory delay ms':>16} {'measured delay ms':>18} {'pos RMS mrad':>13} {'vel RMS rad/s':>14}")
    best = (1e9, None)
    for n in [1, 3, 5, 9, 17, 33, 65, 129]:
        y = moving_average(z, n)
        p, dl, ve = score(y, q, v)
        best = min(best, (p, f"MA N={n}"))
        print(f"{n:>5} {1000*DT*(n-1)/2:>16.1f} {dl:>18.1f} {p:>13.2f} {ve:>14.2f}")

    # 4. Sweep: first-order low-pass cutoff
    print("\n[4] first-order low-pass, cutoff fc")
    print(f"{'fc Hz':>6} {'tau ms':>8} {'measured delay ms':>18} {'pos RMS mrad':>13} {'vel RMS rad/s':>14}")
    for fc in [100, 50, 20, 10, 5, 2, 1, 0.5]:
        y = low_pass(z, fc)
        p, dl, ve = score(y, q, v)
        best = min(best, (p, f"LPF fc={fc} Hz"))
        print(f"{fc:>6} {1000/(2*np.pi*fc):>8.1f} {dl:>18.1f} {p:>13.2f} {ve:>14.2f}")
    print(f"\n    lowest position RMS: {best[1]} ({best[0]:.2f} mrad)")

    # 5. IMU at rest: gravity and gyro bias
    _, _, _, acc_r, gyro_r = simulate(m, moving=False, t_end=10.0)
    a_norm = np.linalg.norm(acc_r[-1])
    g_true = gyro_r[:, 2]
    g_meas = g_true + GYRO_BIAS + rng.normal(0, GYRO_SIGMA, g_true.shape)
    drift = np.cumsum(g_meas) * DT
    print(f"\n[5] arm at rest: |accelerometer| = {a_norm:.3f} m/s^2 (gravity is included)")
    print(f"    gyro z: true max |w| = {np.max(np.abs(g_true)):.1e} rad/s; with bias {GYRO_BIAS} rad/s and "
          f"noise {GYRO_SIGMA} rad/s, integrated angle after 10 s = {drift[-1]:.3f} rad "
          f"({np.degrees(drift[-1]):.1f} deg)")


    # 6. Break it: keep MA N=17 (16 ms delay), make the motion faster
    print("\n[6] break it: fixed moving average N=17, motion frequencies scaled by k")
    print(f"{'k':>4} {'true vel RMS rad/s':>19} {'raw pos RMS mrad':>17} {'MA17 pos RMS mrad':>18}")
    for k in [0.5, 1, 2, 4, 8]:
        qk, vk, _, _, _ = simulate(m, k=k)
        zk = encoder(qk, np.random.default_rng(SEED))
        p_raw, _, _ = score(zk, qk, vk)
        p17, _, _ = score(moving_average(zk, 17), qk, vk)
        print(f"{k:>4} {np.sqrt(np.mean(vk[500:]**2)):>19.2f} {p_raw:>17.2f} {p17:>18.2f}")


if __name__ == "__main__":
    main()
