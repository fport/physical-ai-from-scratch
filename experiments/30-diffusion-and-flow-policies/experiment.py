"""Lesson 30 · Multimodal actions: MSE regression vs a mixture model vs a flow-matching policy.

Task (2D, bird's-eye view): a point goes from (0, -1) to (0, +1); a round obstacle of radius 0.30
sits at the origin. 200 demonstrations: half pass on the left, half on the right; the side is
independent of the start. A policy sees the start (x0, y0) and outputs the whole path
(40 waypoints = an 80-dimensional "action chunk").

Part A  MSE MLP (NumPy) vs 2-component mixture density network (NumPy), ported from
        notebooks/30-diffusion-and-flow-policies.ipynb (same seeds, same code path).
Part B  Sweeps for the MSE model: number of demonstrations, fraction of left demonstrations.
Part C  Flow-matching policy (PyTorch, CPU) trained on the same 200 demonstrations.
        Sweep the number of Euler integration steps -> collision rate, left/right split,
        single-query latency.

Run:  python experiment.py      (deterministic, fixed seeds; timings depend on machine load)
"""
import math
import platform
import time

import numpy as np
import torch

torch.set_num_threads(2)
print(f"NumPy {np.__version__} | PyTorch {torch.__version__} | Python {platform.python_version()} "
      f"| {platform.machine()} | torch threads {torch.get_num_threads()}")

START, GOAL = np.array([0.0, -1.0]), np.array([0.0, 1.0])
OBS_C, OBS_R = np.array([0.0, 0.0]), 0.30
T_DEMO = 40          # waypoints per path
N_ROLLOUT = 200
EPOCHS = 1500
HIDDEN = 64
LR = 3e-3
SEED = 0
N_MAIN = 200
SIGMA = 0.2          # fixed std of the mixture components (scaled units)
SCALE = 2.0          # path offsets are ~0.5; the networks see ~1


def wilson(k, n, z=1.96):
    """95% Wilson score interval for k successes in n trials, in percent."""
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * max(0.0, c - h), 100 * min(1.0, c + h)


# ── demonstrations ───────────────────────────────────────────────────────────────
def demo(rng, side, x0):
    """Smooth path passing the obstacle on `side` (-1 left, +1 right); lateral offset jitters."""
    a = 0.55 + rng.normal(0, 0.04)
    s = np.linspace(0, 1, T_DEMO + 1)
    y = -1 + 2 * s
    x = x0 * (1 - s) + side * a * np.sin(np.pi * s) ** 1.2
    return np.stack([x, y], 1)


def make_data(rng, n_demo, left_frac):
    n_left = int(round(n_demo * left_frac))
    sides = np.array([-1] * n_left + [1] * (n_demo - n_left))
    rng.shuffle(sides)
    x0s = rng.uniform(-0.05, 0.05, n_demo)
    paths = np.stack([demo(rng, s, x0) for s, x0 in zip(sides, x0s)])
    X = paths[:, 0, :]                                            # start point (observation)
    Y = (paths[:, 1:, :] - paths[:, :1, :]).reshape(n_demo, -1)   # path relative to start (action)
    return paths, sides, X, Y


# ── small NumPy MLP (2 hidden tanh layers), Adam, full batch ─────────────────────────
class MLP:
    def __init__(self, rng, n_in, n_out):
        sz = [n_in, HIDDEN, HIDDEN, n_out]
        self.W = [rng.normal(0, 1 / np.sqrt(a), (a, b)) for a, b in zip(sz[:-1], sz[1:])]
        self.b = [np.zeros(b) for b in sz[1:]]
        self.m = [np.zeros_like(p) for p in self.W + self.b]
        self.v = [np.zeros_like(p) for p in self.W + self.b]
        self.k = 0

    def forward(self, X):
        self.h = [X]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = self.h[-1] @ W + b
            self.h.append(np.tanh(z) if i < len(self.W) - 1 else z)
        return self.h[-1]

    def backward(self, g):
        gW, gb = [], []
        for i in reversed(range(len(self.W))):
            gW.insert(0, self.h[i].T @ g)
            gb.insert(0, g.sum(0))
            if i:
                g = (g @ self.W[i].T) * (1 - self.h[i] ** 2)
        self.k += 1
        for j, (p, d) in enumerate(zip(self.W + self.b, gW + gb)):
            self.m[j] = 0.9 * self.m[j] + 0.1 * d
            self.v[j] = 0.999 * self.v[j] + 0.001 * d * d
            p -= LR * (self.m[j] / (1 - 0.9 ** self.k)) / (np.sqrt(self.v[j] / (1 - 0.999 ** self.k)) + 1e-8)


def inp(X):
    return np.stack([X[:, 0] * 20, X[:, 1]], 1)  # x0 in +-0.05 -> +-1


def train_mse(rng, X, Y, epochs=EPOCHS):
    net = MLP(rng, 2, Y.shape[1])
    I, T = inp(X), Y * SCALE
    for _ in range(epochs):
        P = net.forward(I)
        net.backward(2 * (P - T) / T.size)
    return net


K = 2


def mdn_split(out, D):
    logit = out[:, :K]
    mu = out[:, K:K + K * D].reshape(-1, K, D)
    return logit, mu, np.full((len(out), K), np.log(SIGMA))  # fixed isotropic sigma


def train_mdn(rng, X, Y, epochs=EPOCHS):
    D = Y.shape[1]
    net = MLP(rng, 2, K + K * D)
    I, T = inp(X), Y * SCALE
    n = len(X)
    # k-means++-like init: a random demo and the demo farthest from it (no side labels used)
    i = int(rng.integers(n))
    j = int(np.argmax(((T - T[i]) ** 2).sum(1)))
    net.b[-1][K:] = np.concatenate([T[i], T[j]])
    net.W[-1][:, K:] *= 0.1
    for _ in range(epochs):
        logit, mu, logs = mdn_split(net.forward(I), D)
        lw = logit - logit.max(1, keepdims=True)
        lw = lw - np.log(np.exp(lw).sum(1, keepdims=True))
        d2 = ((T[:, None, :] - mu) ** 2).sum(2)
        s2 = np.exp(2 * logs)
        lp = lw - d2 / (2 * s2) - D * logs
        r = np.exp(lp - lp.max(1, keepdims=True))
        r /= r.sum(1, keepdims=True)                       # responsibilities
        g_logit = (np.exp(lw) - r) / n
        g_mu = (-(r / s2)[:, :, None] * (T[:, None, :] - mu)) / n
        net.backward(np.concatenate([g_logit, g_mu.reshape(n, -1)], 1))
    return net


# ── rollout and collision check ───────────────────────────────────────────────────────
_S = np.linspace(0, 1, 51)


def hits_obstacle(path):
    """True if any of 51 points on any segment is inside the obstacle."""
    p, d = path[:-1], path[1:] - path[:-1]
    pts = p[:, None, :] + _S[None, :, None] * d[:, None, :]
    return bool((np.linalg.norm(pts - OBS_C, axis=2) < OBS_R).any())


def side_of(path):
    mid = path[np.argmin(np.abs(path[:, 1]))]
    return -1 if mid[0] < 0 else 1


def rollout_path(x0, rel):
    p0 = np.array([x0, START[1]])
    return np.vstack([p0, p0 + rel.reshape(-1, 2)])


def evaluate(predict, rng, n=N_ROLLOUT):
    """predict: array of n starts x0 -> (n, 80) relative paths (unscaled)."""
    xs = rng.uniform(-0.05, 0.05, n)
    rel = predict(xs)
    fulls = [rollout_path(x0, r) for x0, r in zip(xs, rel)]
    hit = np.array([hits_obstacle(f) for f in fulls])
    side = np.array([side_of(f) for f in fulls])
    return dict(n=n, hits=int(hit.sum()), left=int(((side < 0) & ~hit).sum()),
                right=int(((side > 0) & ~hit).sum()), fulls=fulls)


def starts(xs):
    return np.stack([xs, np.full_like(xs, START[1])], 1)


def mse_predict(net):
    return lambda xs: net.forward(inp(starts(xs))) / SCALE


def mdn_predict(net, rng, D):
    def f(xs):
        out = []
        for x0 in xs:                                           # per trial: pick a mode, follow it
            logit, mu, _ = mdn_split(net.forward(inp(starts(np.array([x0])))), D)
            w = np.exp(logit[0] - logit[0].max())
            k = int(rng.choice(K, p=w / w.sum()))
            out.append(mu[0, k] / SCALE)
        return np.array(out)
    return f


def row(name, ev):
    lo, hi = wilson(ev["hits"], ev["n"])
    return (f"{name:<34} {ev['hits']:>4}/{ev['n']}  {100 * ev['hits'] / ev['n']:6.1f}%  "
            f"[{lo:5.1f}, {hi:5.1f}]   {ev['left']:>4}  {ev['right']:>5}")


HDR = f"{'model':<34} {'collided':>8}  {'rate':>7}  {'95% CI (Wilson)':>15}  {'left':>4}  {'right':>5}"

# ══ Part A ════════════════════════════════════════════════════════════════════════
paths, sides, X, Y = make_data(np.random.default_rng(SEED), N_MAIN, 0.5)
D = Y.shape[1]
print(f"\nPart A · {N_MAIN} demonstrations ({(sides < 0).sum()} left, {(sides > 0).sum()} right), "
      f"{N_ROLLOUT} trials, NumPy MLP {HIDDEN}-{HIDDEN} tanh, {EPOCHS} epochs, seed {SEED}")
t0 = time.time()
mse = train_mse(np.random.default_rng(SEED + 1), X, Y)
ev_mse = evaluate(mse_predict(mse), np.random.default_rng(SEED + 2))
mdn = train_mdn(np.random.default_rng(SEED + 1), X, Y)
ev_mdn = evaluate(mdn_predict(mdn, np.random.default_rng(SEED + 3), D), np.random.default_rng(SEED + 2))
_, _, XL, YL = make_data(np.random.default_rng(SEED), N_MAIN, 1.0)
ev_L = evaluate(mse_predict(train_mse(np.random.default_rng(SEED + 1), XL, YL)), np.random.default_rng(SEED + 2))
print(HDR)
print(row("MSE, half left / half right", ev_mse))
print(row("MSE, all demonstrations left", ev_L))
print(row("mixture (MDN, K=2), half / half", ev_mdn))
mse_c = rollout_path(0.0, mse_predict(mse)(np.array([0.0]))[0])
print(f"at the obstacle (y = 0): demos pass at x = {paths[sides < 0, 20, 0].mean():+.2f} / "
      f"{paths[sides > 0, 20, 0].mean():+.2f}; MSE path at x = {mse_c[20, 0]:+.3f} (obstacle radius {OBS_R})")

print("\nrobustness: three other seeds (data, init, trials)")
print(f"{'seed':>4}  {'MSE collided':>12}  {'MDN collided':>12}")
for s in (1, 2, 3):
    _, _, Xs, Ys = make_data(np.random.default_rng(10 * s), N_MAIN, 0.5)
    e1 = evaluate(mse_predict(train_mse(np.random.default_rng(10 * s + 1), Xs, Ys)), np.random.default_rng(10 * s + 2))
    e2 = evaluate(mdn_predict(train_mdn(np.random.default_rng(10 * s + 1), Xs, Ys), np.random.default_rng(10 * s + 3), D),
                  np.random.default_rng(10 * s + 2))
    print(f"{s:>4}  {e1['hits']:>8}/{e1['n']}  {e2['hits']:>8}/{e2['n']}")

# ══ Part B ════════════════════════════════════════════════════════════════════════
print("\nPart B · MSE model, sweep the number of demonstrations (half left / half right)")
print(f"{'demos':>5}  {'collided':>8}  {'rate':>6}")
for n in [10, 20, 50, 100, 200, 500]:
    _, _, Xn, Yn = make_data(np.random.default_rng(SEED), n, 0.5)
    ev = evaluate(mse_predict(train_mse(np.random.default_rng(SEED + 1), Xn, Yn)), np.random.default_rng(SEED + 2))
    print(f"{n:>5}  {ev['hits']:>4}/{ev['n']}  {100 * ev['hits'] / ev['n']:5.1f}%")

print("\nPart B · MSE model, sweep the fraction of left demonstrations (200 demos)")
print(f"{'left %':>6}  {'collided':>8}  {'rate':>6}")
for lf in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    _, _, Xf, Yf = make_data(np.random.default_rng(SEED), N_MAIN, lf)
    ev = evaluate(mse_predict(train_mse(np.random.default_rng(SEED + 1), Xf, Yf)), np.random.default_rng(SEED + 2))
    print(f"{100 * lf:>6.0f}  {ev['hits']:>4}/{ev['n']}  {100 * ev['hits'] / ev['n']:5.1f}%")
print(f"(Parts A-B: {time.time() - t0:.1f} s wall time)")

# ══ Part C · flow matching ════════════════════════════════════════════════════════
FM_STEPS, FM_BATCH, FM_LR, FM_HIDDEN = 4000, 256, 1e-3, 256


class VelocityNet(torch.nn.Module):
    """v_theta(a_t, t | o): action chunk (80) + time embedding (16) + observation (2) -> velocity (80)."""

    def __init__(self, d_act, d_obs):
        super().__init__()
        self.register_buffer("freq", torch.exp(torch.linspace(0, math.log(100), 8)))
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_act + 16 + d_obs, FM_HIDDEN), torch.nn.SiLU(),
            torch.nn.Linear(FM_HIDDEN, FM_HIDDEN), torch.nn.SiLU(),
            torch.nn.Linear(FM_HIDDEN, FM_HIDDEN), torch.nn.SiLU(),
            torch.nn.Linear(FM_HIDDEN, d_act))

    def forward(self, a, t, o):
        e = t[:, None] * self.freq[None]
        return self.net(torch.cat([a, torch.sin(e), torch.cos(e), o], 1))


def train_flow(X, Y, seed):
    torch.manual_seed(seed)
    O = torch.tensor(inp(X), dtype=torch.float32)
    A1 = torch.tensor(Y * SCALE, dtype=torch.float32)
    net = VelocityNet(A1.shape[1], O.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=FM_LR)
    g = torch.Generator().manual_seed(seed)
    for it in range(FM_STEPS):
        idx = torch.randint(0, len(A1), (FM_BATCH,), generator=g)
        a1, o = A1[idx], O[idx]
        a0 = torch.randn(a1.shape, generator=g)                  # noise sample
        t = torch.rand(FM_BATCH, generator=g)
        at = (1 - t[:, None]) * a0 + t[:, None] * a1             # straight-line path (sigma_min = 0)
        loss = ((net(at, t, o) - (a1 - a0)) ** 2).mean()         # regress the conditional velocity
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net, loss.item()


@torch.no_grad()
def flow_sample(net, xs, n_steps, gen):
    o = torch.tensor(inp(starts(np.asarray(xs, dtype=float))), dtype=torch.float32)
    a = torch.randn((len(o), net.net[-1].out_features), generator=gen)
    dt = 1.0 / n_steps
    for i in range(n_steps):                                     # Euler integration t: 0 -> 1
        t = torch.full((len(o),), i * dt)
        a = a + dt * net(a, t, o)
    return a.numpy() / SCALE


print(f"\nPart C · flow-matching policy, same {N_MAIN} demos; MLP 3x{FM_HIDDEN} SiLU, "
      f"{FM_STEPS} Adam steps, batch {FM_BATCH}, lr {FM_LR}, seed {SEED}")
t0 = time.time()
flow, last_loss = train_flow(X, Y, SEED)
print(f"training: {time.time() - t0:.1f} s wall time (machine under load), final batch loss {last_loss:.4f}")
print(f"{'Euler steps':>11}  {'collided':>8}  {'rate':>6}  {'95% CI (Wilson)':>15}  {'left':>4}  {'right':>5}  "
      f"{'net evals':>9}  {'1-query ms (min)':>16}  {'mean |x| at y=0':>15}")
for n_steps in [1, 2, 4, 8, 16, 32]:
    gen = torch.Generator().manual_seed(SEED + 3)
    ev = evaluate(lambda xs: flow_sample(flow, xs, n_steps, gen), np.random.default_rng(SEED + 2))
    lo, hi = wilson(ev["hits"], ev["n"])
    lat = []
    for _ in range(30):
        t1 = time.perf_counter()
        flow_sample(flow, [0.0], n_steps, gen)
        lat.append(1000 * (time.perf_counter() - t1))
    absx = np.mean([abs(f[20, 0]) for f in ev["fulls"]])
    print(f"{n_steps:>11}  {ev['hits']:>4}/{ev['n']}  {100 * ev['hits'] / ev['n']:5.1f}%  [{lo:5.1f}, {hi:5.1f}]   "
          f"{ev['left']:>4}  {ev['right']:>5}  {n_steps:>9}  {np.min(lat):>16.2f}  {absx:>15.3f}")
print("(latency: fastest of 30 single-trajectory queries on CPU, 2 threads; the machine was under heavy load,\n so treat absolute ms as approximate; network evaluations per query = Euler steps)")
