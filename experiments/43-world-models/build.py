"""Lesson 43 · Build it: a world model of the pendulum by least squares, its compounding error, and MPC inside it.

The pendulum is the Experiment's (1 kg at 0.5 m, damping 0.05, |u| <= 2 N m, control step 0.05 s = 5 x 10 ms).
Instead of the Experiment's MLP (PyTorch), the model here is linear in polynomial features, fitted in one
least-squares solve, so the whole chain runs in NumPy in a few seconds.

Step 1  The true dynamics by hand vs MuJoCo.
Step 2  Data: 400 random-action trajectories x 50 steps, the Experiment's recipe.
Step 3  Fit the change of state for feature degrees 1-4: one-step error on held-out data.
Step 4  Roll the models out open loop: error vs horizon, and the growth factor.
Step 5  Random-shooting MPC inside the true dynamics and inside the degree-2 model.

The functions live in physai/foundation.py. Run:  uv run python experiments/43-world-models/build.py  (~4 s)
"""
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "..")]
from physai.foundation import (fit_world_model, pendulum_step, poly_features, predict, rollout_errors,  # noqa: E402
                               shooting_mpc)

XML = """
<mujoco>
  <option timestep="0.01" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="pole">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="0.05"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.01" mass="0"/>
      <geom type="sphere" pos="0 0 -0.5" size="0.03" mass="1"/>
    </body>
  </worldbody>
  <actuator><motor joint="hinge" ctrlrange="-2 2" ctrllimited="true"/></actuator>
</mujoco>
"""
HORIZONS = (1, 5, 10, 20, 40, 60, 100)


def collect(n_traj, T, rng):
    """Random torques held for two control steps, from random starts; all trajectories stepped as one batch."""
    s = np.stack([rng.uniform(-np.pi, np.pi, n_traj), rng.uniform(-8, 8, n_traj)], axis=-1)
    S, A, S2 = [], [], []
    for t in range(T):
        if t % 2 == 0:
            u = rng.uniform(-2.0, 2.0, n_traj)
        s2 = pendulum_step(s, u)
        S.append(s), A.append(u), S2.append(s2)
        s = s2
    return np.concatenate(S), np.concatenate(A), np.concatenate(S2)


print(f"MuJoCo {mujoco.__version__} | NumPy {np.__version__}")

print("\n== Step 1. The pendulum step by hand vs MuJoCo (5 x mj_step of 10 ms) ==")
m = mujoco.MjModel.from_xml_string(XML)
d = mujoco.MjData(m)
M = np.zeros((1, 1))
mujoco.mj_forward(m, d)
mujoco.mj_fullM(m, d, M)
rng = np.random.default_rng(0)
worst = 0.0
for _ in range(200):
    s, u = rng.uniform([-np.pi, -8], [np.pi, 8]), rng.uniform(-3, 3)      # torques beyond the 2 N m limit too
    d.qpos[0], d.qvel[0], d.ctrl[0] = s[0], s[1], u
    mujoco.mj_step(m, d, nstep=5)
    worst = max(worst, np.abs(pendulum_step(s, u) - [d.qpos[0], d.qvel[0]]).max())
print(f"inertia from MuJoCo's mass matrix: {M[0, 0]:.5f} kg m^2 (0.25 + 2/5 m r^2 = {0.25 + 0.4 * 0.03 ** 2:.5f})")
print(f"max |ours - MuJoCo| over 200 random states and torques: {worst:.1e}")

print("\n== Step 2. Data: 400 random-action trajectories x 50 steps ==")
S, A, S2 = collect(400, 50, np.random.default_rng(0))
St, At, S2t = collect(100, 50, np.random.default_rng(5))                  # held out
print(f"training transitions {len(S)}, held-out transitions {len(St)}")
print(f"|delta theta| up to {np.abs(S2[:, 0] - S[:, 0]).max():.3f} rad per step, "
      f"|delta theta_dot| up to {np.abs(S2[:, 1] - S[:, 1]).max():.3f} rad/s per step")

print("\n== Step 3. Fit the change of state, feature degree 1-4 ==")
print(f"{'degree':>6s} {'features':>8s} {'one-step |angle err| rad':>25s} {'one-step |vel err| rad/s':>25s}")
models = {}
for deg in (1, 2, 3, 4):
    models[deg] = fit_world_model(S, A, S2, deg)
    err = np.abs(predict(models[deg], St, At, deg) - S2t).mean(axis=0)
    print(f"{deg:6d} {poly_features(St[:1], At[:1], deg).shape[1]:8d} {err[0]:25.1e} {err[1]:25.1e}")

print("\n== Step 4. Open-loop rollouts: 200 held-out starts, the same random torques to model and truth ==")
rng = np.random.default_rng(1)
s0 = np.stack([rng.uniform(-np.pi, np.pi, 200), rng.uniform(-8, 8, 200)], axis=-1)
U = np.repeat(rng.uniform(-2.0, 2.0, (200, 50)), 2, axis=1)
print(f"{'degree':>6s} " + " ".join(f"{'H=' + str(h):>9s}" for h in HORIZONS) + f" {'H100/H1':>8s}   (mean |angle err| rad; med 2: median, degree 2)")
res = {}
for deg in (1, 2, 3, 4):
    res[deg] = rollout_errors(pendulum_step, lambda s, u, deg=deg: predict(models[deg], s, u, deg), s0, U, HORIZONS)
    print(f"{deg:6d} " + " ".join(f"{res[deg][h][0]:9.1e}" for h in HORIZONS) + f" {res[deg][100][0] / res[deg][1][0]:8.0f}")
print(f"{"med 2":>6s} " + " ".join(f"{res[2][h][1]:9.1e}" for h in HORIZONS))

print("\n== Step 5. Random-shooting MPC swing-up, K = 100, 3 starts, 5 s ==")
starts = [np.array([x, 0.0]) for x in np.random.default_rng(2).uniform(-0.1, 0.1, 3)]
print(f"{'planner':<16s} {'H':>3s} {'success':>8s} {'mean tail err rad':>18s} {'first upright step':>19s}")
for name, plan in (("true dynamics", pendulum_step), ("degree-2 model", lambda s, u: predict(models[2], s, u, 2))):
    for H in (10, 30):
        outs = [shooting_mpc(plan, pendulum_step, s, H, 100, np.random.default_rng(10 + i)) for i, s in enumerate(starts)]
        first = [o[2] for o in outs if o[2] >= 0]
        print(f"{name:<16s} {H:3d} {sum(o[0] for o in outs):6d}/3 {np.mean([o[1] for o in outs]):18.3f} "
              f"{(f'{np.mean(first):.0f}' if first else '-'):>19s}")
