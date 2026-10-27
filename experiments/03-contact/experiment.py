"""Lesson 03 · Contact: how deep does a resting cube sink, and how high does it bounce?

A 10 cm, 1 kg cube on a plane. MuJoCo contact is a soft constraint set by
solref = (timeconst, dampratio) and solimp. Sweeps:
  1. timeconst -> steady penetration of a resting cube (and a sphere, to check the doc formula)
  2. solimp impedance d -> steady penetration
  3. dampratio -> rebound height after a 10 cm drop
Break it: timeconst below 2*timestep, with the refsafe flag on and off; and dampratio too small.

Run:  python experiment.py      (~1 s on Apple Silicon)
"""
import mujoco
import numpy as np

H = 0.05          # cube half-size / sphere radius (m)
DT = 0.002        # MuJoCo default timestep (s)
G = 9.81


def xml(tc, dr, z0=H, shape="box", solimp=None, refsafe=True, dt=DT):
    flag = "" if refsafe else '<flag refsafe="disable"/>'
    geom = f'type="box" size="{H} {H} {H}"' if shape == "box" else f'type="sphere" size="{H}"'
    imp = f'solimp="{solimp}"' if solimp else ""
    return f"""
<mujoco>
  <option timestep="{dt}">{flag}</option>
  <worldbody>
    <geom type="plane" size="1 1 .1" solref="{tc} {dr}" {imp}/>
    <body pos="0 0 {z0}">
      <freejoint/>
      <geom {geom} mass="1" solref="{tc} {dr}" {imp}/>
    </body>
  </worldbody>
</mujoco>"""


def rest(tc, dr=1.0, shape="box", solimp=None, T=2.0):
    """Place the body touching the plane, let it settle. Returns penetration (m), impedance d, #contacts."""
    m = mujoco.MjModel.from_xml_string(xml(tc, dr, shape=shape, solimp=solimp))
    d = mujoco.MjData(m)
    while d.time < T - 1e-9:
        mujoco.mj_step(m, d)
    pen = -np.mean([c.dist for c in d.contact[:d.ncon]])
    return pen, d.efc_KBIP[0, 2], d.ncon


def drop(tc, dr, height=0.10, refsafe=True, T=2.0, dt=DT):
    """Drop from `height` (bottom face above plane). Returns peak penetration, rebound height,
    final penetration, std of height over the last 0.5 s, max height reached (all in m)."""
    m = mujoco.MjModel.from_xml_string(xml(tc, dr, z0=H + height, refsafe=refsafe, dt=dt))
    d = mujoco.MjData(m)
    z = []
    while d.time < T - 1e-9:
        mujoco.mj_step(m, d)
        z.append(d.qpos[2] - H)            # height of the bottom face
    z = np.array(z)
    i = int(np.argmax(z <= 0))             # first step in contact
    k = i + int(np.argmin(z[i:]))          # deepest point of the first impact
    tail = z[-int(0.5 / dt):]
    return dict(peak=-z[k], rebound=max(0.0, z[k:].max()), final=-tail.mean(), jitter=tail.std(), zmax=z.max())


def main():
    print("mujoco", mujoco.__version__, "· numpy", np.__version__)
    print(f"cube 10 cm, 1 kg · timestep {DT} s · default solimp 0.9 0.95 0.001 0.5 2\n")

    print("sweep 1: timeconst (dampratio 1) -> steady penetration at rest")
    print(" timeconst(s)  cube(mm)  #contacts  sphere(mm)  d(sphere)  g*(1-d)*tc^2 (mm)  g*(1-d)*(dw/d)^2*tc^2 (mm)")
    for tc in [0.001, 0.002, 0.004, 0.005, 0.01, 0.02, 0.05, 0.1]:
        pc, _, nc = rest(tc)
        ps, ds, _ = rest(tc, shape="sphere")
        tc_eff = max(tc, 2 * DT)          # refsafe: solver uses max(timeconst, 2*timestep)
        # steady state of  a1 + d*(b*v + k*r) = (1-d)*a_u  with a1 = v = 0 and k = d/(dw^2*tc^2*dr^2):
        general = G * (1 - ds) * (0.95 / ds) ** 2 * tc_eff ** 2
        print(f"  {tc:6.3f}      {pc * 1000:7.4f}     {nc}      {ps * 1000:8.4f}    {ds:6.4f}     "
              f"{G * (1 - ds) * tc_eff ** 2 * 1000:8.4f}          {general * 1000:8.4f}")

    print("\nsweep 2: constant impedance solimp='d d 0.001' (timeconst 0.02) -> steady penetration of the cube")
    for dval in [0.5, 0.9, 0.95, 0.99]:
        pc, dd, _ = rest(0.02, solimp=f"{dval} {dval} 0.001")
        print(f"  d {dval:4.2f} · penetration {pc * 1000:7.4f} mm · (1-d) {1 - dd:5.3f}")

    print("\nsweep 3: dampratio (timeconst 0.02) -> 10 cm drop")
    print(" dampratio  peak penetration(mm)  rebound(mm)  final penetration(mm)")
    for dr in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 2.0]:
        r = drop(0.02, dr)
        print(f"   {dr:4.1f}         {r['peak'] * 1000:7.3f}          {r['rebound'] * 1000:7.3f}        "
              f"{r['final'] * 1000:7.3f}")

    print("\nsweep 4: timeconst (dampratio 1) -> 10 cm drop")
    for tc in [0.004, 0.01, 0.02, 0.05, 0.1]:
        r = drop(tc, 1.0)
        print(f"  timeconst {tc:5.3f} · peak penetration {r['peak'] * 1000:7.3f} mm · rebound {r['rebound'] * 1000:6.3f} mm")

    print(f"\nbreak it 1: timeconst below 2*timestep = {2 * DT} s (dampratio 1, 10 cm drop)")
    print(" timeconst  refsafe  peak pen(mm)  rebound(mm)  max height(m)  jitter last 0.5 s(mm)")
    for tc in [0.0005, 0.001, 0.002, 0.003, 0.004]:
        for rs in (True, False):
            r = drop(tc, 1.0, refsafe=rs)
            print(f"  {tc:6.4f}    {'on ' if rs else 'off'}    {r['peak'] * 1000:9.3f}   {r['rebound'] * 1000:10.3f}"
                  f"   {r['zmax']:10.3f}     {r['jitter'] * 1000:10.3f}")

    print("\nbreak it 2: refsafe on, timeconst 0.02, small dampratio (natural frequency 1/(timeconst*dampratio))")
    for dr in [0.1, 0.07, 0.05]:
        r = drop(0.02, dr)
        w = 1 / (0.02 * dr)
        print(f"  dampratio {dr:4.2f} · timestep*w {DT * w:4.2f} · rebound {r['rebound'] * 1000:9.3f} mm · "
              f"max height {r['zmax']:7.3f} m · jitter {r['jitter'] * 1000:9.3f} mm")


if __name__ == "__main__":
    main()
