"""Lesson 43 · World models: learn s' = f(s, a) from data, then plan inside the learned model.

System: a torque-limited pendulum in MuJoCo (1 kg point mass at 0.5 m, hinge damping 0.05 N m s/rad).
The motor can give at most 2 N m, less than the 4.9 N m gravity torque at horizontal, so swinging up needs
pumping: a planner has to look ahead.
Control step 0.05 s (5 MuJoCo steps of 10 ms, semi-implicit Euler). State s = (theta, theta_dot); theta = 0 hangs down.

  1. Data: random-action trajectories from random start states (the true simulator).
  2. Model: MLP (cos theta, sin theta, theta_dot, u) -> delta s, trained with MSE on normalised targets.
  3. Sweep A: open-loop rollout horizon H -> prediction error on held-out trajectories (compounding error).
  4. Sweep B: planning horizon H for random-shooting MPC (K random action sequences, pick the cheapest, execute the first action,
     replan) planning inside the learned model vs inside the true simulator; swing-up success.
  5. Break it: train the model only on low-energy data (starts near the bottom, weak random torques) and plan.

Run:  python experiment.py      (~1-3 min on CPU; depends on machine load)
"""
import math
import platform
import time

import mujoco
import numpy as np
from mujoco import rollout
import torch
from torch import nn

torch.set_num_threads(1)                      # bit-for-bit reproducible training on CPU

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
U_MAX, DT, SUBSTEPS = 2.0, 0.05, 5
N_START, K, HOLD = 3, 100, 5
M = mujoco.MjModel.from_xml_string(XML)
D = mujoco.MjData(M)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def sim_step(s, u):
    """True dynamics: one control step from state s = (theta, theta_dot) with torque u."""
    D.qpos[0], D.qvel[0], D.ctrl[0] = s[0], s[1], u
    mujoco.mj_step(M, D, nstep=SUBSTEPS)
    return np.array([D.qpos[0], D.qvel[0]])


def collect(n_traj, T, rng, low_energy=False):
    """Random actions (held for 2 control steps) from random starts. Returns S, A, S_next."""
    S, A, S2 = [], [], []
    for _ in range(n_traj):
        if low_energy:
            s = np.array([rng.uniform(-0.3, 0.3), rng.uniform(-0.5, 0.5)])
        else:
            s = np.array([rng.uniform(-np.pi, np.pi), rng.uniform(-8, 8)])
        for t in range(T):
            if t % 2 == 0:
                u = rng.uniform(-U_MAX, U_MAX) * (0.25 if low_energy else 1.0)
            s2 = sim_step(s, u)
            S.append(s), A.append(u), S2.append(s2)
            s = s2
    return np.array(S), np.array(A)[:, None], np.array(S2)


def features(s, u):
    """s: (..., 2) theta, theta_dot; u: (..., 1). Angle enters as cos/sin so the model is periodic."""
    return torch.cat([torch.cos(s[..., :1]), torch.sin(s[..., :1]), s[..., 1:2], u], dim=-1)


class WorldModel(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 2))
        self.register_buffer("x_mu", torch.zeros(4))
        self.register_buffer("x_sd", torch.ones(4))
        self.register_buffer("d_mu", torch.zeros(2))
        self.register_buffer("d_sd", torch.ones(2))

    def forward(self, s, u):
        """Predicted next state: s + delta, delta de-normalised (predicting the change, not the state)."""
        x = (features(s, u) - self.x_mu) / self.x_sd
        return s + self.net(x) * self.d_sd + self.d_mu


def train(S, A, S2, grad_steps=1200, seed=0):
    """Fixed number of gradient steps (1200 = 30 epochs of the full dataset), whatever the dataset size."""
    torch.manual_seed(seed)
    s, u, s2 = (torch.tensor(v, dtype=torch.float32) for v in (S, A, S2))
    model = WorldModel()
    x = features(s, u)
    delta = s2 - s
    model.x_mu, model.x_sd = x.mean(0), x.std(0) + 1e-6
    model.d_mu, model.d_sd = delta.mean(0), delta.std(0) + 1e-6
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    n = len(s)
    epochs = math.ceil(grad_steps / math.ceil(n / 512))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for idx in torch.randperm(n, generator=g).split(512):
            pred = model.net((x[idx] - model.x_mu) / model.x_sd)
            loss = ((pred - (delta[idx] - model.d_mu) / model.d_sd) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    return model


def rollout_errors(model, horizons, rng, n_test=200, T=100):
    """Open-loop: start from a true state, feed the same random actions to both, compare after H steps."""
    errs = {h: [] for h in horizons}
    vel = {h: [] for h in horizons}
    for _ in range(n_test):
        s0 = np.array([rng.uniform(-np.pi, np.pi), rng.uniform(-8, 8)])
        us = np.repeat(rng.uniform(-U_MAX, U_MAX, T // 2), 2)
        true = [s0]
        for u in us:
            true.append(sim_step(true[-1], u))
        with torch.no_grad():
            sp = torch.tensor(s0, dtype=torch.float32)[None]
            pred = [sp[0].numpy()]
            for u in us:
                sp = model(sp, torch.tensor([[u]], dtype=torch.float32))
                pred.append(sp[0].numpy())
        for h in horizons:
            errs[h].append(abs(wrap(pred[h][0] - true[h][0])))
            vel[h].append(abs(pred[h][1] - true[h][1]))
    return {h: (np.mean(errs[h]), np.median(errs[h]), np.mean(vel[h])) for h in horizons}


def cost(theta, theta_dot, u):
    return wrap(theta - np.pi) ** 2 + 0.1 * theta_dot ** 2 + 0.001 * u ** 2


SPEC = mujoco.mjtState.mjSTATE_FULLPHYSICS
D_ROLL = mujoco.MjData(M)


def plan_true(s, U):
    """Total cost of each candidate sequence (K, H) rolled out in the true simulator (mujoco.rollout, in C)."""
    K, H = U.shape
    D_ROLL.qpos[0], D_ROLL.qvel[0], D_ROLL.time = s[0], s[1], 0.0
    x0 = np.zeros(mujoco.mj_stateSize(M, SPEC))
    mujoco.mj_getState(M, D_ROLL, x0, SPEC)
    ctrl = np.repeat(U, SUBSTEPS, axis=1)[:, :, None]          # hold each action for SUBSTEPS sim steps
    st, _ = rollout.rollout(M, D_ROLL, x0[None], ctrl)
    st = st[:, SUBSTEPS - 1::SUBSTEPS]                          # state after each control step
    th, thd = st[..., 1], st[..., 2]                            # layout: time, qpos, qvel
    return (cost(th, thd, U)).sum(axis=1)


def plan_learned(model, s, U):
    K, H = U.shape
    with torch.no_grad():
        x = torch.tensor(s, dtype=torch.float32).repeat(K, 1)
        Ut = torch.tensor(U, dtype=torch.float32)
        c = torch.zeros(K)
        for t in range(H):
            x = model(x, Ut[:, t:t + 1])
            th, thd = x[:, 0], x[:, 1]
            c += (torch.remainder(th, 2 * math.pi) - math.pi) ** 2 + 0.1 * thd ** 2 + 0.001 * Ut[:, t] ** 2
    return c.numpy()


def mpc_episode(planner, s0, H, K, rng, steps=100, execute=1):
    """Random shooting MPC: plan, apply the first `execute` actions of the best sequence open loop, replan.
    Success = within 0.2 rad of upright for the whole last second (20 steps)."""
    s, angles, plan = s0.copy(), [], []
    for _ in range(steps):
        if not plan:
            # candidate sequences: piecewise constant, a new random torque every HOLD steps (0.25 s)
            U = np.repeat(rng.uniform(-U_MAX, U_MAX, (K, -(-H // HOLD))), HOLD, axis=1)[:, :H]
            plan = list(U[np.argmin(planner(s, U)), :execute])
        s = sim_step(s, plan.pop(0))          # the plan is always executed in the TRUE simulator
        angles.append(abs(wrap(s[0] - np.pi)))
    a = np.array(angles)
    first = int(np.argmax(a < 0.2)) if np.any(a < 0.2) else -1
    return bool(np.all(a[-20:] < 0.2)), float(np.mean(a[-20:])), first


def main():
    t_start = time.time()
    print(f"mujoco {mujoco.__version__} · numpy {np.__version__} · torch {torch.__version__} (CPU, 1 thread) · "
          f"python {platform.python_version()} · {platform.machine()}")
    print(f"pendulum: 1 kg at 0.5 m, |u| <= {U_MAX} N m (gravity torque at horizontal 4.91 N m), "
          f"control step {DT} s = {SUBSTEPS} x 10 ms, semi-implicit Euler")
    rng = np.random.default_rng(0)
    S, A, S2 = collect(400, 50, rng)
    t0 = time.time()
    model = train(S, A, S2)
    n_par = sum(p.numel() for p in model.parameters())
    with torch.no_grad():
        s, u, s2 = (torch.tensor(v, dtype=torch.float32) for v in (S, A, S2))
        train_mse = float(((model(s, u) - s2) ** 2).mean())
    print(f"data: 400 random-action trajectories x 50 steps = {len(S)} transitions; "
          f"MLP 4-128-128-2 ({n_par} parameters), 1200 gradient steps (batch 512, 30 epochs), train MSE {train_mse:.2e} "
          f"(training ~{time.time() - t0:.0f} s, loaded machine)")

    print("\n[A] Open-loop rollout horizon -> prediction error (200 held-out trajectories, random actions)")
    horizons = (1, 2, 5, 10, 20, 40, 60, 100)
    res = rollout_errors(model, horizons, np.random.default_rng(1))
    print(f"{'H steps':>8}{'H s':>7}{'mean |angle err| rad':>22}{'median rad':>12}{'mean |vel err| rad/s':>22}")
    for h in horizons:
        print(f"{h:>8}{h * DT:>7.2f}{res[h][0]:>22.4f}{res[h][1]:>12.4f}{res[h][2]:>22.4f}")

    print(f"\n[B] Random-shooting MPC swing-up from hanging ({N_START} starts, theta ~ U(-0.1, 0.1), K = {K} sequences,")
    print(f"    each piecewise constant with a new random torque every {HOLD} steps)")
    print("    success = within 0.2 rad of upright for the whole last 1 s of a 5 s episode; the chosen action")
    print("    is always applied to the true simulator, only the planner's predictions differ.")
    starts = [np.array([x, 0.0]) for x in np.random.default_rng(2).uniform(-0.1, 0.1, N_START)]
    print(f"{'planner':>9}{'H':>4}{'success':>9}{'mean tail err rad':>19}{'first upright step':>20}{'plan time s':>13}")
    for H in (10, 20, 30):
        for name, planner in (("true", plan_true), ("learned", lambda s_, U_: plan_learned(model, s_, U_))):
            t0 = time.time()
            outs = [mpc_episode(planner, s0, H, K, np.random.default_rng(10 + i)) for i, s0 in enumerate(starts)]
            ok = sum(o[0] for o in outs)
            first = [o[2] for o in outs if o[2] >= 0]
            print(f"{name:>9}{H:>4}{f'{ok}/{N_START}':>9}{np.mean([o[1] for o in outs]):>19.3f}"
                  f"{(f'{np.mean(first):.0f}' if first else '-'):>20}{time.time() - t0:>13.1f}")

    print("\n[C] Break it: a world model trained only on low-energy data (starts |theta| <= 0.3 rad,")
    print("    |theta_dot| <= 0.5 rad/s, torques <= 0.5 N m) -> it never saw the upright half of the circle")
    S_lo, A_lo, S2_lo = collect(400, 50, np.random.default_rng(3), low_energy=True)
    model_lo = train(S_lo, A_lo, S2_lo)
    print(f"  max |theta| in this dataset: {np.max(np.abs(wrap(S_lo[:, 0]))):.2f} rad "
          f"(full-range dataset: {np.max(np.abs(wrap(S[:, 0]))):.2f} rad)")
    res_lo = rollout_errors(model_lo, (1, 10), np.random.default_rng(1))
    print(f"  one-step mean |angle err| on the full-range test set: {res_lo[1][0]:.4f} rad "
          f"(full-range model {res[1][0]:.4f}); 10-step: {res_lo[10][0]:.4f} rad (full-range model {res[10][0]:.4f})")
    outs = [mpc_episode(lambda s_, U_: plan_learned(model_lo, s_, U_), s0, 20, K, np.random.default_rng(10 + i))
            for i, s0 in enumerate(starts)]
    print(f"  MPC with the low-energy model, H = 20: success {sum(o[0] for o in outs)}/{N_START}, "
          f"mean tail err {np.mean([o[1] for o in outs]):.3f} rad")
    print("\n[D] Break it harder: shrink the (full-range) dataset; same 1200 gradient steps, MPC with H = 30")
    print(f"{'trajectories':>13}{'transitions':>12}{'1-step err rad':>16}{'20-step err rad':>17}{'MPC success':>13}"
          f"{'mean tail err rad':>19}")
    for n_traj in (400, 40, 8, 2):
        S_n, A_n, S2_n = S[:n_traj * 50], A[:n_traj * 50], S2[:n_traj * 50]
        mdl = model if n_traj == 400 else train(S_n, A_n, S2_n)
        r = rollout_errors(mdl, (1, 20), np.random.default_rng(1))
        outs = [mpc_episode(lambda s_, U_: plan_learned(mdl, s_, U_), s0, 30, K, np.random.default_rng(10 + i))
                for i, s0 in enumerate(starts)]
        print(f"{n_traj:>13}{len(S_n):>12}{r[1][0]:>16.4f}{r[20][0]:>17.4f}{f'{sum(o[0] for o in outs)}/{N_START}':>13}"
              f"{np.mean([o[1] for o in outs]):>19.3f}")
    print("\n[E] Break it: replan less often. Execute the first k actions of each plan open loop (H = 30)")
    print(f"{'k':>4}{'replan every s':>16}{'true: success':>15}{'tail err rad':>14}{'learned: success':>18}"
          f"{'tail err rad':>14}")
    for k in (1, 5, 10, 20, 30):
        row = []
        for planner in (plan_true, lambda s_, U_: plan_learned(model, s_, U_)):
            outs = [mpc_episode(planner, s0, 30, K, np.random.default_rng(10 + i), execute=k)
                    for i, s0 in enumerate(starts)]
            row += [f"{sum(o[0] for o in outs)}/{N_START}", np.mean([o[1] for o in outs])]
        print(f"{k:>4}{k * DT:>16.2f}{row[0]:>15}{row[1]:>14.3f}{row[2]:>18}{row[3]:>14.3f}")
    print(f"\ntotal time ~{(time.time() - t_start) / 60:.1f} min (approximate; heavily loaded machine, not a benchmark)")


if __name__ == "__main__":
    main()
