"""Lesson 24 · LQR and model predictive control on a MuJoCo cart-pole.

Part A  Linearize the MJCF cart-pole at the upright with mujoco.mjd_transitionFD, solve the
        discrete-time Riccati equation (scipy.linalg.solve_discrete_are) -> LQR gain K.
Part B  Region of attraction: sweep the initial pole angle (and the force limit) -> does LQR
        bring the pole back upright?
Part C  MPPI (sampling-based MPC, rollouts with mujoco.rollout) swings the pole up from hanging
        down, which no linear controller around the upright can do. Sweep the horizon ->
        swing-up success, balance quality, cost. Then the same MPC with an LQR hand-off.

Run:  python experiment.py      (deterministic, fixed seeds)
"""
import platform

import mujoco
import numpy as np
import scipy
from mujoco import rollout
from scipy.linalg import solve_discrete_are

print(f"MuJoCo {mujoco.__version__} | NumPy {np.__version__} | SciPy {scipy.__version__} "
      f"| Python {platform.python_version()} | {platform.machine()}")

# theta = 0 is upright; the pole rotates about the y axis. Cart on a 3 m track (+-1.5 m).
XML = """
<mujoco>
  <option timestep="0.01" integrator="Euler"/>
  <worldbody>
    <body name="cart">
      <joint name="slider" type="slide" axis="1 0 0" range="-1.5 1.5" limited="true"/>
      <geom type="box" size="0.15 0.1 0.05" mass="1"/>
      <body name="pole">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.6" size="0.02" mass="0.2"/>
      </body>
    </body>
  </worldbody>
  <actuator><motor joint="slider" gear="1" ctrllimited="true" ctrlrange="-100 100"/></actuator>
</mujoco>"""
m = mujoco.MjModel.from_xml_string(XML)
DT = m.opt.timestep
SPEC = mujoco.mjtState.mjSTATE_FULLPHYSICS       # state = [time, x, theta, xdot, thetadot]
NSTATE = mujoco.mj_stateSize(m, SPEC)


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


# ---------------------------------------------------------------- Part A: linearize + LQR
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)                          # upright, at rest, ctrl = 0 is an equilibrium
A = np.zeros((2 * m.nv, 2 * m.nv))
Bm = np.zeros((2 * m.nv, m.nu))
mujoco.mjd_transitionFD(m, d, 1e-6, True, A, Bm, None, None)   # x_{t+1} = A x_t + B u_t
Q = np.diag([1.0, 10.0, 0.1, 0.1])               # weights on x, theta, xdot, thetadot
R = np.array([[0.01]])
P = solve_discrete_are(A, Bm, Q, R)
K = np.linalg.solve(R + Bm.T @ P @ Bm, Bm.T @ P @ A)
np.set_printoptions(precision=4, suppress=True)
print(f"\nPart A · cart 1 kg, pole 0.2 kg / 0.6 m, dt = {DT} s, state (x, theta, xdot, thetadot)")
print("A =\n", A)
print("B =", Bm.ravel())
print("open-loop |eig(A)| max =", f"{np.abs(np.linalg.eigvals(A)).max():.4f}  (> 1: unstable)")
print("K =", K.ravel())
print("closed-loop |eig(A - B K)| =", np.sort(np.abs(np.linalg.eigvals(A - Bm @ K))))


def lqr_u(x, F):
    xs = x.copy()
    xs[1] = wrap(xs[1])
    return float(np.clip(-(K @ xs)[0], -F, F))


def lqr_trial(theta0, F, T=5.0):
    """'fell' = pole passed 90 deg from upright. A trial that starts below 90 deg (hanging down)
    cannot 'fall', so it runs the full T and ends 'ok' or 'not up'."""
    d = mujoco.MjData(m)
    d.qpos[1] = theta0
    from_below = abs(wrap(theta0)) > np.pi / 2
    for _ in range(int(round(T / DT))):
        d.ctrl[0] = lqr_u(np.r_[d.qpos, d.qvel], F)
        mujoco.mj_step(m, d)
        if not from_below and abs(wrap(d.qpos[1])) > np.pi / 2:
            return "fell"
        if abs(d.qpos[0]) > 1.49:
            return "track end"
    if abs(wrap(d.qpos[1])) < np.radians(2):
        return "ok"
    return "not up" if from_below else "not settled"


# ---------------------------------------------------------------- Part B: region of attraction
print("\nPart B · LQR region of attraction: initial angle (cart at rest at x = 0), 5 s, "
      "success = |theta| < 2 deg at the end")
forces = [5, 10, 20, 50]
angles = list(range(0, 61, 5))
print(f"{'theta0 (deg)':>12} | " + " | ".join(f"{'F=' + str(F) + ' N':>11}" for F in forces))
for a in angles:
    print(f"{a:12d} | " + " | ".join(f"{lqr_trial(np.radians(a), F):>11}" for F in forces))
print(f"{'180 (down)':>12} | " + " | ".join(f"{lqr_trial(np.pi - 1e-3, F):>11}" for F in forces))

# ---------------------------------------------------------------- Part C: MPPI swing-up
F_MPC = 20.0                 # force limit for the swing-up (N)
SUBC = 2                     # MPC acts every 2 physics steps (50 Hz)
K_SAMPLES, SIGMA, LAMBDA = 64, 8.0, 0.5
datas = [mujoco.MjData(m) for _ in range(8)]   # rollout threads


def stage_cost(S, U):
    x, th, xd, thd = S[..., 1], S[..., 2], S[..., 3], S[..., 4]
    return (5 * (1 - np.cos(th)) + 0.5 * x ** 2 + 0.01 * thd ** 2 + 0.01 * xd ** 2
            + 1e-4 * U ** 2 + 100 * (np.abs(x) > 1.4))


def mpc_trial(horizon_s, seed, handoff_deg=None, T=6.0):
    """MPPI from hanging down. Optional: switch to LQR when |theta| < handoff_deg."""
    rng = np.random.default_rng(seed)
    H = int(round(horizon_s / (DT * SUBC)))
    d = mujoco.MjData(m)
    d.qpos[1] = np.pi
    st = np.zeros(NSTATE)
    Unom = np.zeros(H)
    cost, log = 0.0, []
    for _ in range(int(round(T / (DT * SUBC)))):
        th = wrap(d.qpos[1])
        if handoff_deg is not None and abs(th) < np.radians(handoff_deg):
            u = lqr_u(np.r_[d.qpos, d.qvel], F_MPC)
        else:
            mujoco.mj_getState(m, d, st, SPEC)
            Uk = np.clip(Unom + rng.normal(0.0, SIGMA, (K_SAMPLES, H)), -F_MPC, F_MPC)
            ctrl = np.repeat(Uk, SUBC, axis=1)[:, :, None]
            S, _ = rollout.rollout(m, datas, st, ctrl)          # K_SAMPLES x (H*SUBC) states
            c = stage_cost(S, ctrl[..., 0]).sum(axis=1)
            w = np.exp(-(c - c.min()) / LAMBDA)
            Unom = np.clip((w / w.sum()) @ Uk, -F_MPC, F_MPC)   # weighted average of samples
            u = Unom[0]
            Unom = np.r_[Unom[1:], Unom[-1]]                    # warm start: shift
        for _ in range(SUBC):
            d.ctrl[0] = u
            mujoco.mj_step(m, d)
            cost += stage_cost(np.r_[0.0, d.qpos, d.qvel], u) * DT
        log.append((d.time, np.degrees(wrap(d.qpos[1]))))
    L = np.array(log)
    up = np.flatnonzero(np.abs(L[:, 1]) < 10)
    t_up = L[up[0], 0] if len(up) else np.nan
    wobble = np.abs(L[-50:, 1]).max()                 # worst |theta| in the last second
    return t_up, wobble, cost


def mpc_table(label, horizons, seeds, handoff=None):
    print(f"\n{label}")
    print(f"{'horizon (s)':>11} | {'H steps':>7} | {'passed <10 deg':>14} | {'held (<10 deg)':>14} | "
          f"{'med t_up (s)':>12} | {'med worst |theta| last 1 s (deg)':>32} | {'med cost':>8}")
    for hs in horizons:
        res = [mpc_trial(hs, s, handoff) for s in seeds]
        t_up = np.array([r[0] for r in res])
        wob = np.array([r[1] for r in res])
        cst = np.array([r[2] for r in res])
        reached = int(np.sum(~np.isnan(t_up)))
        held = int(np.sum(wob < 10))
        mt = f"{np.nanmedian(t_up):12.2f}" if reached else f"{'-':>12}"
        print(f"{hs:11.1f} | {int(round(hs / (DT * SUBC))):7d} | {reached:6d}/{len(seeds):<7d} | "
              f"{held:6d}/{len(seeds):<7d} | {mt} | {np.median(wob):32.1f} | {np.median(cst):8.2f}")


SEEDS = range(8)
print(f"\nPart C · MPPI swing-up from hanging down, |F| <= {F_MPC} N, {K_SAMPLES} samples, "
      f"sigma {SIGMA} N, lambda {LAMBDA}, 50 Hz, 6 s, {len(SEEDS)} seeds")
mpc_table("C1 · MPPI only", [0.2, 0.3, 0.5, 0.7, 1.0, 1.5], SEEDS)
mpc_table("C2 · MPPI for the swing-up, LQR once |theta| < 20 deg", [0.5, 1.0, 1.5], SEEDS, handoff=20)
