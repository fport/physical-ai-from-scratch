"""Lesson 04 · Friction: does a heavier cube slip later?

An 8 cm cube is placed at rest on a tilted ramp. The ramp angle is swept in
0.05° steps; 3 masses (0.1 · 1 · 10 kg) × 3 friction coefficients (μ = 0.3 · 0.5 · 0.8).

Extra runs: creep below the threshold, a speed-only criterion, the elliptic cone,
a cube rotated 45° about the ramp normal, and μ = 1.

Run:  python experiment.py          (full run, under a minute on Apple Silicon)
      python experiment.py --quick  (main table, μ = 0.5 only)
"""
import sys

import mujoco
import numpy as np

HALF = 0.04                    # cube half-size -> 8 cm cube
RAMP = (0.40, 0.15, 0.01)      # ramp half-sizes -> 80 × 30 × 2 cm
V_TH, A_TH = 2.0, 0.5          # cm/s: slip criterion (below)
GRID = 0.05                    # degrees


def xml(theta_deg, mu, mass, cone="pyramidal"):
    return f"""
<mujoco>
  <option cone="{cone}"/>
  <worldbody>
    <body name="ramp" euler="0 {-theta_deg} 0">
      <geom type="box" size="{RAMP[0]} {RAMP[1]} {RAMP[2]}" friction="{mu} 0.005 0.0001"/>
    </body>
    <body name="cube">
      <freejoint/>
      <geom type="box" size="{HALF} {HALF} {HALF}" mass="{mass}" friction="{mu} 0.005 0.0001"/>
    </body>
  </worldbody>
</mujoco>"""


def speeds(theta_deg, mu, mass, cone="pyramidal", yaw_deg=0.0):
    m = mujoco.MjModel.from_xml_string(xml(theta_deg, mu, mass, cone))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    R = d.body("ramp").xmat.reshape(3, 3)
    n, s = R[:, 2], R[:, 0]                 # surface normal, ramp axis (uphill)
    d.qpos[:3] = n * (RAMP[2] + HALF)       # cube touches the surface
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    yaw = np.array([np.cos(np.radians(yaw_deg) / 2), 0, 0, np.sin(np.radians(yaw_deg) / 2)])   # about ramp normal
    mujoco.mju_mulQuat(d.qpos[3:7], q, yaw)
    v = []
    while d.time < 2.0 - 1e-9:
        mujoco.mj_step(m, d)
        v.append(-d.qvel[:3] @ s * 100)     # downhill speed, cm/s
    v, dt = np.array(v), m.opt.timestep
    return v[int(0.5 / dt):int(1.0 / dt)].mean(), v[int(1.5 / dt):int(2.0 / dt)].mean()


def slips(v1, v2, speed_only=False):
    """Slipping <=> both fast (v2 > 2 cm/s) and accelerating (v2 - v1 > 0.5 cm/s)."""
    return v2 > V_TH and (speed_only or (v2 - v1) > A_TH)


def sweep(mu, mass, cone="pyramidal", yaw_deg=0.0):
    ref = np.degrees(np.arctan(mu))
    angles = np.round(np.arange(round((ref - 3) / GRID) * GRID, ref + 1 + 1e-9, GRID), 2)
    return ref, angles, np.array([speeds(float(a), mu, mass, cone, yaw_deg) for a in angles])


def slip_angle(angles, v, speed_only=False):
    flags = [slips(a, b, speed_only) for a, b in v]
    k = next(i for i in range(len(flags)) if all(flags[i:]))   # this angle and every angle above it slip
    return float(angles[k])


def main():
    print("mujoco", mujoco.__version__)
    quick = "--quick" in sys.argv
    mus = (0.5,) if quick else (0.3, 0.5, 0.8)
    print("slip angle, pyramidal cone (criterion: v2 > 2 cm/s and accelerating)")
    runs = {}
    for mu in mus:
        row = []
        for mass in (0.1, 1.0, 10.0):
            ref, angles, v = sweep(mu, mass)
            runs[(mu, mass)] = (angles, v)
            row.append(slip_angle(angles, v))
        print(f"  μ {mu} · theory arctan μ = {ref:5.2f}° · measured (0.1 / 1 / 10 kg): {row} · "
              f"max deviation {max(row) - ref:+.2f}°")
        a, v_light = runs[(mu, 0.1)]
        print(f"    max |speed difference| 0.1 kg vs 10 kg over the sweep: "
              f"{np.abs(v_light - runs[(mu, 10.0)][1]).max():.1e} cm/s")
    if quick:
        return

    print("\ncreep: largest steady downhill speed v2 at least 1° below arctan μ (1 kg)")
    for mu in mus:
        angles, v = runs[(mu, 1.0)]
        below = angles <= np.degrees(np.arctan(mu)) - 1
        print(f"  μ {mu} · max v2 {v[below, 1].max():.2f} cm/s · max (v2 - v1) {(v[below, 1] - v[below, 0]).max():.3f} cm/s")

    print("\nspeed-only criterion (v2 > 2 cm/s, no acceleration check), 1 kg")
    for mu in mus:
        angles, v = runs[(mu, 1.0)]
        sa = slip_angle(angles, v, speed_only=True)
        print(f"  μ {mu} · {sa:.2f}° ({sa - np.degrees(np.arctan(mu)):+.2f}° vs theory)")

    print("\nelliptic cone, 1 kg")
    for mu in mus:
        ref, angles, v = sweep(mu, 1.0, cone="elliptic")
        print(f"  μ {mu} · {slip_angle(angles, v):.2f}° (theory {ref:.2f}°)")

    print("\ncube rotated 45° about the ramp normal, 1 kg")
    for mu in (0.5, 0.8):
        for cone in ("pyramidal", "elliptic"):
            ref, angles, v = sweep(mu, 1.0, cone=cone, yaw_deg=45)
            below = angles <= ref - 0.5
            sa = slip_angle(angles, v)
            print(f"  μ {mu} · {cone:9s} · slip angle {sa:.2f}° ({sa - ref:+.2f}°) · "
                  f"max v2 at least 0.5° below theory {v[below, 1].max():5.2f} cm/s")

    ref, angles, v = sweep(1.0, 1.0)
    print(f"\nμ 1.0, 1 kg, pyramidal: {slip_angle(angles, v):.2f}° (theory {ref:.2f}°)")


if __name__ == "__main__":
    main()
