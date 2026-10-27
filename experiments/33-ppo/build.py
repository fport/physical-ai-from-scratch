"""Lesson 33 · Build it: the pieces of PPO from scratch, checked at every step.

Step 1  GAE on three steps by hand; lambda = 0 and lambda = 1; a 64 x 16 rollout vs the direct double sum.
Step 2  Episode ends inside a rollout: done masks vs computing each episode separately.
Step 3  The Gaussian policy's log-probability vs SciPy and torch.distributions.
Step 4  The clipped surrogate: values and gradients by autograd.
Step 5  One PPO iteration on the reach task from our pieces vs ppo.py; ppo.py's first-iteration KL for 3 seeds.
Step 6  The discount as a time horizon.

Step 5 runs one iteration (1024 samples) per seed, not the Experiment's 300,000-step trainings.
The functions live in physai/rl.py (the lesson shows them step by step).
Run:  uv run python experiments/33-ppo/build.py      (~6 s, deterministic)
"""
import os
import sys

import mujoco
import numpy as np
import scipy
import torch
import torch.nn as nn
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "32-rl-environment"), os.path.join(HERE, ".."), os.path.join(HERE, "..", "..")]
from physai.rl import approx_kl, clipped_surrogate, gae, gaussian_log_prob, horizon_seconds  # noqa: E402
from ppo import ActorCritic, PPOConfig, train  # noqa: E402
from reach_env import ReachEnv  # noqa: E402

torch.set_num_threads(1)
print(f"MuJoCo {mujoco.__version__} | NumPy {np.__version__} | SciPy {scipy.__version__} | PyTorch {torch.__version__}")
BASE = dict(lr=1e-3, gamma=0.95, lam=0.95, log_std_init=-1.0)       # the lesson's reference config

print("\n== Step 1. GAE on three steps: r = (1, 0, 2), V = (0.5, 0.4, 0.3), V(s_3) = 0.2, gamma 0.9 ==")
r, v, last, nodone = np.array([1.0, 0.0, 2.0]), np.array([0.5, 0.4, 0.3]), np.array(0.2), np.zeros(3)
for lam in (0.8, 0.0, 1.0):
    adv, ret = gae(r, v, last, nodone, 0.9, lam)
    print(f"lambda {lam:.1f}: A = {np.round(adv, 6)}, returns = {np.round(ret, 6)}")
delta = r + 0.9 * np.r_[v[1:], last] - v
mc = np.array([sum(0.9 ** (l - t) * r[l] for l in range(t, 3)) + 0.9 ** (3 - t) * 0.2 for t in range(3)])
print(f"TD errors delta = {np.round(delta, 6)} (= lambda 0); discounted return - V = {np.round(mc - v, 6)} (= lambda 1)")
rng = np.random.default_rng(0)
R, Vv, L = rng.normal(size=(64, 16)), rng.normal(size=(64, 16)), rng.normal(size=16)
adv, _ = gae(R, Vv, L, np.zeros((64, 16)), 0.95, 0.95)
Vn = np.vstack([Vv[1:], L])
direct = np.array([sum((0.95 * 0.95) ** (l - t) * (R[l] + 0.95 * Vn[l] - Vv[l]) for l in range(t, 64)) for t in range(64)])
print(f"64 x 16 rollout, no episode ends: max |gae - direct double sum| = {np.abs(adv - direct).max():.1e}")

print("\n== Step 2. Episode ends inside the rollout ==")
dones = (rng.random((64, 16)) < 0.05).astype(float)
adv, _ = gae(R, Vv, L, dones, 0.95, 0.95)
ref = np.zeros_like(adv)
for n in range(16):                                              # each episode piece on its own, backwards
    end = 64
    for t in range(63, -1, -1):
        if dones[t, n]:
            end = t + 1
        nv = lambda l: 0.0 if dones[l, n] else (L[n] if l == 63 else Vv[l + 1, n])  # noqa: E731,B023
        ref[t, n] = sum((0.95 * 0.95) ** (l - t) * (R[l, n] + 0.95 * nv(l) - Vv[l, n]) for l in range(t, end))
print(f"{int(dones.sum())} episode ends: max |gae - per-episode sums| = {np.abs(adv - ref).max():.1e}")
print("truncated step: r + gamma * V(s_final) is stored and done = 1, so the future is not lost: "
      f"{-0.01 + 0.95 * -0.2:.3f} for r = -0.01, V(s_final) = -0.200")

print("\n== Step 3. Gaussian log-probability, 1000 actions in 4-D ==")
mu, log_std = rng.normal(size=(1000, 4)), np.full(4, -1.0)
a = mu + np.exp(log_std) * rng.normal(size=(1000, 4))
lp = gaussian_log_prob(a, mu, log_std)
lp_scipy = norm.logpdf(a, mu, np.exp(log_std)).sum(-1)
lp_torch = torch.distributions.Normal(torch.tensor(mu), torch.tensor(log_std).exp()).log_prob(torch.tensor(a)).sum(-1).numpy()
print(f"max |ours - scipy| = {np.abs(lp - lp_scipy).max():.1e}, max |ours - torch| = {np.abs(lp - lp_torch).max():.1e}")
print(f"at the mean: {gaussian_log_prob(mu[:1], mu[:1], log_std)[0]:.4f} = 4 * (1 - 0.5 ln 2 pi) "
      f"= {4 * (1 - 0.5 * np.log(2 * np.pi)):.4f}")

print("\n== Step 4. The clipped surrogate, eps = 0.2 ==")
print(" ratio   A   objective min(rA, clip(r)A)   d objective / d ratio")
for A in (1.0, -1.0):
    for rv in (0.7, 0.9, 1.1, 1.3):
        ratio = torch.tensor([rv], requires_grad=True)
        loss = clipped_surrogate(ratio, torch.tensor([A]))
        loss.backward()
        print(f"{rv:6.1f} {A:+4.0f} {-loss.item():29.2f} {-ratio.grad.item() + 0.0:23.1f}")

print("\n== Step 5. One PPO iteration (16 envs x 64 steps, 10 epochs x 4 minibatches) ==")


def one_iteration(seed, cfg):
    """ppo.py's first iteration, written with physai's gae, clipped_surrogate and approx_kl."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    envs = [ReachEnv() for _ in range(cfg.n_envs)]
    obs = np.stack([e.reset(seed=int(cfg.seed * 1000 + i))[0] for i, e in enumerate(envs)])
    pi = ActorCritic(15, 4, cfg.hidden, cfg.log_std_init)
    opt = torch.optim.Adam(pi.parameters(), lr=cfg.lr, eps=1e-5)
    T, N = cfg.n_steps, cfg.n_envs
    buf = {k: np.zeros((T, N) + s, np.float32) for k, s in (("o", (15,)), ("a", (4,)), ("lp", ()), ("v", ()), ("r", ()), ("d", ()))}
    for t in range(T):
        a, lp, v = pi.act(obs)
        buf["o"][t], buf["a"][t], buf["lp"][t], buf["v"][t] = obs, a, lp, v
        for i, e in enumerate(envs):
            o2, r, term, trunc, _ = e.step(a[i])
            if trunc and not term:                                # time limit: keep the future
                with torch.no_grad():
                    r += cfg.gamma * pi.v(torch.as_tensor(o2)).item()
            buf["r"][t, i], buf["d"][t, i] = r, float(term or trunc)
            if term or trunc:
                o2, _ = e.reset()
            obs[i] = o2
    with torch.no_grad():
        last_v = pi.v(torch.as_tensor(obs)).squeeze(-1).numpy()
    adv, ret = gae(buf["r"], buf["v"], last_v, buf["d"], cfg.gamma, cfg.lam)
    b_o, b_a = torch.as_tensor(buf["o"].reshape(T * N, -1)), torch.as_tensor(buf["a"].reshape(T * N, -1))
    b_lp, b_adv, b_ret = (torch.as_tensor(x.reshape(-1)) for x in (buf["lp"], adv, ret))
    kls = []
    for _ in range(cfg.epochs):
        idx = rng.permutation(T * N)
        for s in range(0, T * N, cfg.minibatch):
            mb = idx[s:s + cfg.minibatch]
            d = pi.dist(b_o[mb])
            lp = d.log_prob(b_a[mb]).sum(-1)
            A = b_adv[mb]
            A = (A - A.mean()) / (A.std() + 1e-8)
            pg = clipped_surrogate((lp - b_lp[mb]).exp(), A, cfg.clip)
            vl = ((pi.v(b_o[mb]).squeeze(-1) - b_ret[mb]) ** 2).mean()
            loss = pg + cfg.vf_coef * vl - cfg.ent_coef * d.entropy().sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(pi.parameters(), cfg.max_grad_norm)
            opt.step()
            with torch.no_grad():
                kls.append(approx_kl(b_lp[mb], lp).item())
    return pi, float(np.mean(kls))


cfg = PPOConfig(seed=0, total_steps=1024, **BASE)
mine, kl_mine = one_iteration(0, cfg)
theirs, log = train(lambda i: ReachEnv(), cfg)
dp = max((p1 - p2).abs().max().item() for p1, p2 in zip(mine.parameters(), theirs.parameters()))
print(f"seed 0: approx KL ours {kl_mine:.6f}, ppo.py {log['iters'][0]['kl']:.6f}; max |weights ours - ppo.py| = {dp:.1e}")
for clip, epochs, label in ((0.2, 10, "reference (eps 0.2, 10 epochs)"), (100.0, 30, "Break it (eps 100, 30 epochs)")):
    kls = [train(lambda i: ReachEnv(), PPOConfig(seed=s, total_steps=1024, clip=clip, epochs=epochs, **BASE))[1]["iters"][0]["kl"]
           for s in (0, 1, 2)]
    print(f"{label}: ppo.py iteration-1 KL, seeds 0/1/2 = " + " / ".join(f"{k:.4g}" for k in kls)
          + f", mean {np.mean(kls):.3g}")

print("\n== Step 6. The discount as a time horizon, policy period 20 ms ==")
print("  gamma  horizon")
for g in (0.9, 0.95, 0.99, 0.995):
    print(f"{g:7.3f}  {horizon_seconds(0.02, g):5.2f} s")
