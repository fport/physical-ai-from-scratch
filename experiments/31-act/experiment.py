"""Lesson 31 · Action chunking and temporal ensembling on a MuJoCo planar arm.

Task: a 2-link planar arm (position servos, 50 Hz control, random torque disturbances) starts at a
home pose, moves to a random target, holds still there for 0.5 s (think: a gripper closing), then
moves to a fixed drop pose. The scripted demonstrator is a state-feedback controller with an
internal pause timer. The policy observes (q, qdot, goal) only: no timer, so during the pause the
observation does not tell how long the arm has already waited.

Policy: an MLP maps the observation to a chunk of the next k absolute joint-position targets and
is trained with an L1 loss, as in ACT. No CVAE, no transformer, no images: only the chunking and
the execution scheme are kept.

Sweep: chunk size k in {1, 4, 8, 16, 32}, executed either
  open-loop  : query every k steps, execute the whole chunk
  ensembled  : query every step, average all predictions for the current step with
               weights w_i = exp(-m i), i = 0 for the oldest prediction, m = 0.01
-> success (3 training seeds x 200 episodes, 95% Wilson CI), failure types, time spent at the
   target, RMS jerk of the commanded joint targets.

Run:  python experiment.py      (deterministic, fixed seeds; timings depend on machine load)
"""
import math
import platform
import time

import mujoco
import numpy as np
import torch

torch.set_num_threads(2)
print(f"MuJoCo {mujoco.__version__} | NumPy {np.__version__} | PyTorch {torch.__version__} "
      f"| Python {platform.python_version()} | {platform.machine()} | torch threads {torch.get_num_threads()}")

XML = """
<mujoco>
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="l1">
      <joint name="j1" type="hinge" axis="0 0 1" damping="0.05"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="0.6"/>
      <body name="l2" pos="0.3 0 0">
        <joint name="j2" type="hinge" axis="0 0 1" damping="0.05"/>
        <geom type="capsule" fromto="0 0 0 0.25 0 0" size="0.02" mass="0.4"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position joint="j1" kp="20" kv="1.5"/>
    <position joint="j2" kp="20" kv="1.5"/>
  </actuator>
</mujoco>"""
M = mujoco.MjModel.from_xml_string(XML)
L1, L2 = 0.3, 0.25
CTRL_HZ = 50
NSUB = int(round(1 / (CTRL_HZ * M.opt.timestep)))   # 10 physics steps per control step
DT = 1.0 / CTRL_HZ
Q_HOME = np.array([-1.0, 1.8])
Q_DROP = np.array([1.6, 1.0])
STEP = 0.12            # rad, max lead of the commanded target over q (sets the speed)
ARRIVE = 0.01          # m, the demonstrator starts its pause timer inside this radius
T_PAUSE = 25           # control steps = 0.5 s
T_DEMO = 125 + T_PAUSE # 3.0 s per demonstration
T_EVAL = T_DEMO + 50   # 4.0 s budget at test time
TORQUE_NOISE = 0.2     # N m, std per joint, resampled every control step
TOL = 0.03             # m
DWELL = 10             # control steps (0.2 s) inside TOL of the target count as "paused"
N_DEMO, N_EVAL = 100, 200
TRAIN_SEEDS = (0, 1, 2)
SEED = 0


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * max(0.0, c - h), 100 * min(1.0, c + h)


def fk(q):
    q = np.atleast_2d(q)
    return np.stack([L1 * np.cos(q[:, 0]) + L2 * np.cos(q[:, 0] + q[:, 1]),
                     L1 * np.sin(q[:, 0]) + L2 * np.sin(q[:, 0] + q[:, 1])], 1)


def ik(p):
    """Elbow-positive analytic IK for a target p = (x, y)."""
    c2 = (p[0] ** 2 + p[1] ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    q2 = math.acos(np.clip(c2, -1, 1))
    return np.array([math.atan2(p[1], p[0]) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2)), q2])


def sample_target(rng):
    r, a = rng.uniform(0.25, 0.45), rng.uniform(-0.6, 0.9)
    return np.array([r * math.cos(a), r * math.sin(a)])


class Demonstrator:
    """Reach -> pause T_PAUSE steps -> go to the drop pose. The pause timer is internal state."""

    def __init__(self, target, pause=T_PAUSE):
        self.target, self.goal, self.phase, self.count, self.pause = target, ik(target), 0, 0, pause

    def act(self, q):
        if self.phase == 0 and np.linalg.norm(fk(q)[0] - self.target) < ARRIVE:
            self.phase = 1
        if self.phase == 1:
            self.count += 1
            if self.count > self.pause:
                self.phase = 2
            else:
                return self.goal.copy()
        g = self.goal if self.phase == 0 else Q_DROP
        return q + np.clip(g - q, -STEP, STEP)


def obs_of(q, qd, target):
    """(q, qdot / 5, goal in joint space): the goal is the IK of the target position, as if a
    perception module handed it over. No timer, no phase variable."""
    return np.concatenate([q, qd / 5.0, ik(target)])


def new_episode(rng):
    d = mujoco.MjData(M)
    d.qpos[:] = Q_HOME + rng.normal(0, 0.02, 2)
    d.ctrl[:] = d.qpos
    mujoco.mj_forward(M, d)
    return d, sample_target(rng)


def step(d, a, rng):
    d.ctrl[:] = a
    d.qfrc_applied[:] = rng.normal(0, TORQUE_NOISE, 2)
    mujoco.mj_step(M, d, nstep=NSUB)


# ── demonstrations ───────────────────────────────────────────────────────────────────
def collect(n, rng, pause=T_PAUSE):
    O, A = [], []
    for _ in range(n):
        d, target = new_episode(rng)
        demo = Demonstrator(target, pause)
        obs, act = [], []
        for _ in range(125 + pause):
            obs.append(obs_of(d.qpos.copy(), d.qvel.copy(), target))
            act.append(demo.act(d.qpos.copy()))
            step(d, act[-1], rng)
        O.append(np.array(obs)); A.append(np.array(act))
    return O, A


def chunk_dataset(O, A, k):
    """Input o_t, target a_{t..t+k-1}; past the end of a demo the last action is repeated."""
    X, Y = [], []
    for o, a in zip(O, A):
        a_pad = np.vstack([a, np.repeat(a[-1:], k, 0)])
        for t in range(len(o)):
            X.append(o[t])
            Y.append(a_pad[t:t + k].reshape(-1))
    return np.array(X, np.float32), np.array(Y, np.float32)


class ChunkPolicy(torch.nn.Module):
    def __init__(self, k, hidden=256):
        super().__init__()
        self.k = k
        self.net = torch.nn.Sequential(torch.nn.Linear(6, hidden), torch.nn.ReLU(),
                                       torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
                                       torch.nn.Linear(hidden, 2 * k))

    def forward(self, o):
        return self.net(o).view(-1, self.k, 2)


TRAIN_STEPS, BATCH, LR = 3000, 512, 1e-3


def train(O, A, k, seed):
    torch.manual_seed(seed)
    X, Y = map(torch.tensor, chunk_dataset(O, A, k))
    pol = ChunkPolicy(k)
    opt = torch.optim.Adam(pol.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, TRAIN_STEPS)
    g = torch.Generator().manual_seed(seed)
    for _ in range(TRAIN_STEPS):
        idx = torch.randint(0, len(X), (BATCH,), generator=g)
        loss = (pol.net(X[idx]) - Y[idx]).abs().mean()        # L1, as in ACT
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return pol, loss.item()


# ── closed-loop evaluation, all episodes batched ─────────────────────────────────────
@torch.no_grad()
def evaluate(pol, mode, seed, m=0.01, pause=T_PAUSE):
    T_EVAL = 175 + pause
    rng = np.random.default_rng(seed)
    eps = [new_episode(rng) for _ in range(N_EVAL)]
    ds, targets = [e[0] for e in eps], np.array([e[1] for e in eps])
    demos = [Demonstrator(t, pause) for t in targets]
    k = pol.k if pol is not None else 1
    cmd = np.zeros((N_EVAL, T_EVAL, 2))
    ee = np.zeros((N_EVAL, T_EVAL + 1, 2))
    ee[:, 0] = fk(np.array([d.qpos for d in ds]))
    plan = np.zeros((N_EVAL, k, 2))
    num = np.zeros((N_EVAL, T_EVAL + k, 2))
    den = np.zeros((N_EVAL, T_EVAL + k, 1))
    w = np.exp(-m * (k - 1 - np.arange(k)))[None, :, None]   # chunk offset j: older => larger weight
    for t in range(T_EVAL):
        if pol is None:                                       # the demonstrator itself
            a = np.array([dm.act(d.qpos.copy()) for dm, d in zip(demos, ds)])
        else:
            if mode == "ensembled" or t % k == 0:
                o = torch.tensor(np.array([obs_of(d.qpos, d.qvel, tg) for d, tg in zip(ds, targets)]),
                                 dtype=torch.float32)
                chunk = pol(o).numpy()
                if mode == "ensembled":
                    num[:, t:t + k] += w * chunk
                    den[:, t:t + k] += w
                else:
                    plan = chunk
            a = num[:, t] / den[:, t] if mode == "ensembled" else plan[:, t % k]
        cmd[:, t] = a
        for i, d in enumerate(ds):
            step(d, a[i], rng)
        ee[:, t + 1] = fk(np.array([d.qpos for d in ds]))
    drop = fk(Q_DROP)[0]
    near = np.linalg.norm(ee - targets[:, None], axis=2) < TOL
    out = dict(succ=0, missed=0, short=0, stuck=0, other=0, at_target=[])
    for i in range(N_EVAL):
        runs, run = [], 0
        for flag in near[i]:
            run = run + 1 if flag else 0
            runs.append(run)
        if not near[i].any():
            out["missed"] += 1                    # never within TOL of the target
        elif max(runs) < DWELL:
            out["short"] += 1                     # reached it, left before 0.2 s
        elif np.linalg.norm(ee[i, -1] - drop) < TOL:
            out["succ"] += 1
            out["at_target"].append(max(runs) * DT)
        elif near[i, -1]:
            out["stuck"] += 1                     # still at the target at the end
        else:
            out["other"] += 1
    jerk = np.diff(cmd, n=3, axis=1) / DT ** 3
    out["jerk"] = float(np.sqrt((jerk ** 2).sum(2).mean()))
    return out


@torch.no_grad()
def predicted_hold(pol, seed, thresh=0.02):
    """Query the policy at rest exactly on the goal (q = goal, qdot = 0) for N_EVAL targets.
    Return the median chunk index at which the predicted target first leaves the goal by more than
    `thresh` rad, and how many chunks contain such a departure at all."""
    rng = np.random.default_rng(seed)
    goals = np.array([ik(sample_target(rng)) for _ in range(N_EVAL)])
    o = torch.tensor(np.concatenate([goals, np.zeros_like(goals), goals], 1), dtype=torch.float32)
    dev = np.linalg.norm(pol(o).numpy() - goals[:, None], axis=2) > thresh
    idx = [int(np.argmax(r)) for r in dev if r.any()]
    return (float(np.median(idx)) if idx else float("nan")), len(idx)


def line(name, rs):
    """Pool the evaluations of several training seeds (N_EVAL episodes each)."""
    n, s = N_EVAL * len(rs), sum(r["succ"] for r in rs)
    lo, hi = wilson(s, n)
    per = "/".join(str(r["succ"]) for r in rs)
    fails = "  ".join(f"{sum(r[c] for r in rs):>5}" for c in ("missed", "short", "stuck", "other"))
    at = np.concatenate([r["at_target"] for r in rs]) if s else np.array([np.nan])
    return (f"{name:<20} {s:>4}/{n}  {100 * s / n:5.1f}%  [{lo:5.1f}, {hi:5.1f}]  {per:>11}  {fails}  "
            f"{np.median(at):>7.2f}  {np.mean([r['jerk'] for r in rs]):>7.0f}")


HEAD = (f"{'policy':<20} {'success':>8}  {'rate':>6}  {'95% CI':>14}  {'per seed':>11}  {'miss':>5}  "
        f"{'short':>5}  {'stuck':>5}  {'other':>5}  {'pause s':>7}  {'jerk':>7}")

if __name__ == "__main__":
    t0 = time.time()
    O, A = collect(N_DEMO, np.random.default_rng(SEED))
    print(f"\n{N_DEMO} demos x {T_DEMO} steps at {CTRL_HZ} Hz; demonstrator pauses {T_PAUSE} steps "
          f"({T_PAUSE * DT:.1f} s) at the target; torque noise {TORQUE_NOISE} N m")
    print(f"policy: MLP 6-256-256-2k, L1 loss, {TRAIN_STEPS} Adam steps (cosine lr from {LR}), batch {BATCH}, "
          f"training seeds {TRAIN_SEEDS}; eval per seed: {N_EVAL} new targets, {T_EVAL} steps (4.0 s), "
          f"same eval seed for every row")
    print(f"success = within {100 * TOL:.0f} cm of the target for >= {DWELL} steps ({DWELL * DT:.1f} s), "
          f"then within {100 * TOL:.0f} cm of the drop point at 4.0 s")
    print("miss = never reached the target; short = reached, left before 0.2 s; stuck = still at the target "
          "at 4.0 s; pause s = median longest stay at the target (successes); jerk = RMS third difference "
          "of the commanded joint targets, rad/s^3")
    print("\n" + HEAD)
    print(line("demonstrator", [evaluate(None, "-", SEED + 100)]))
    t_train = 0.0
    holds = []
    for k in [1, 4, 8, 16, 32]:
        res = {"open": [], "ensembled": []}
        for s in TRAIN_SEEDS:
            t1 = time.time()
            pol, _ = train(O, A, k, 1000 * s + k)
            t_train += time.time() - t1
            if s == 0:
                holds.append((k, *predicted_hold(pol, SEED + 200)))
            res["open"].append(evaluate(pol, "open", SEED + 100))
            if k > 1:
                res["ensembled"].append(evaluate(pol, "ensembled", SEED + 100))
        print(line(f"k={k:<2} open-loop", res["open"]))
        if k > 1:
            print(line(f"k={k:<2} ensembled", res["ensembled"]))
    print(f"\nChunk predicted at rest on the goal (training seed 0, {N_EVAL} targets): first index whose target "
          f"leaves the goal by > 0.02 rad")
    print(f"{'k':>3}  {'chunks with a departure':>23}  {'median first departure index':>28}")
    for k, med, n_dep in holds:
        print(f"{k:>3}  {n_dep:>17}/{N_EVAL}  {med:>28.1f}")

    print(f"\nBreak it · longer pause, k=32 ensembled, training seed 0; budget = pause + 3.5 s")
    print(HEAD)
    for pause in [25, 50, 75, 100]:
        Op, Ap = (O, A) if pause == T_PAUSE else collect(N_DEMO, np.random.default_rng(SEED), pause)
        t1 = time.time()
        pol, _ = train(Op, Ap, 32, 32)
        t_train += time.time() - t1
        print(line(f"pause {pause * DT:.1f} s", [evaluate(pol, "ensembled", SEED + 100, pause=pause)]))
    print(f"\n(wall time: {t_train:.0f} s training, {time.time() - t0:.0f} s total, on a heavily loaded "
          f"machine; timings approximate)")
