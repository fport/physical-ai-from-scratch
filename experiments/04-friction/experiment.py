"""Lesson 04 · Friction: does a heavier cube slip later?

An 8 cm cube is placed at rest on a tilted ramp. The ramp angle is swept in
0.05° steps; 3 masses (0.1 · 1 · 10 kg) × 3 friction coefficients (μ = 0.3 · 0.5 · 0.8).

Run:  python experiment.py          (full sweep, ~10 s on Apple Silicon)
      python experiment.py --quick  (μ = 0.5 only)
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


def speeds(theta_deg, mu, mass, cone="pyramidal"):
    m = mujoco.MjModel.from_xml_string(xml(theta_deg, mu, mass, cone))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    R = d.body("ramp").xmat.reshape(3, 3)
    n, s = R[:, 2], R[:, 0]                 # surface normal, ramp axis (uphill)
    d.qpos[:3] = n * (RAMP[2] + HALF)       # cube touches the surface
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    d.qpos[3:7] = q
    v = []
    while d.time < 2.0 - 1e-9:
        mujoco.mj_step(m, d)
        v.append(-d.qvel[:3] @ s * 100)     # downhill speed, cm/s
    v, dt = np.array(v), m.opt.timestep
    return v[int(0.5 / dt):int(1.0 / dt)].mean(), v[int(1.5 / dt):int(2.0 / dt)].mean()


def slips(theta_deg, mu, mass, cone="pyramidal"):
    """Slipping <=> both fast (v2 > 2 cm/s) and accelerating (v2 - v1 > 0.5 cm/s)."""
    v1, v2 = speeds(theta_deg, mu, mass, cone)
    return v2 > V_TH and (v2 - v1) > A_TH


def slip_angle(mu, mass, cone="pyramidal"):
    ref = np.degrees(np.arctan(mu))
    angles = np.round(np.arange(round((ref - 3) / GRID) * GRID, ref + 1 + 1e-9, GRID), 2)
    flags = [slips(float(a), mu, mass, cone) for a in angles]
    k = next(i for i in range(len(flags)) if all(flags[i:]))   # this angle and every angle above it slip
    return float(angles[k])


def main():
    print("mujoco", mujoco.__version__)
    mus = (0.5,) if "--quick" in sys.argv else (0.3, 0.5, 0.8)
    for mu in mus:
        row = [slip_angle(mu, m) for m in (0.1, 1.0, 10.0)]
        print(f"μ {mu} · theory arctan μ = {np.degrees(np.arctan(mu)):5.2f}° · "
              f"measured (0.1 / 1 / 10 kg): {row}")


if __name__ == "__main__":
    main()
