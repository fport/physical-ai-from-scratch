"""Lesson 27 · Robot datasets.

Input: the 120 scripted SO-101 demonstrations recorded in lesson 26 (experiments/26-teleop-replay-data/demos.npz).

  A. Anatomy: per-frame columns (observation, action, timestamp, frame_index, episode_index, index, task_index),
     episode boundaries, and normalization statistics (mean, std, min, max, 1st/99th percentile) per feature,
     in the spirit of a LeRobotDataset's meta/stats.json.
  B. Normalization: the same MLP trained on raw features (cube in m, and in mm), MEAN_STD- and MIN_MAX-normalized
     features -> validation error on held-out episodes.
  C. Splitting: frame-level random split vs episode-level split, for 10 / 30 / 90 training episodes.
     The validation error each split reports is compared with the error on 20 never-seen test episodes.

`train_mlp` and `MLPPolicy` are imported by lessons 28-29.

Run:  python experiment.py      (deterministic; ~0.5 min CPU on Apple M5)
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
DEMOS = os.path.join(EXP, "26-teleop-replay-data", "demos.npz")
torch.set_num_threads(1)


def load_demos(path=DEMOS):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def stats(x):
    """Per-feature statistics, the same kinds a LeRobot stats.json stores (mean, std, min, max, quantiles)."""
    return dict(mean=x.mean(0), std=x.std(0), min=x.min(0), max=x.max(0),
                q01=np.percentile(x, 1, axis=0), q99=np.percentile(x, 99, axis=0), count=len(x))


class MLPPolicy:
    """Numpy forward pass of a trained MLP with its normalization baked in (fast inside a MuJoCo loop)."""

    def __init__(self, layers, x_shift, x_scale, y_shift, y_scale):
        self.layers, self.xs, self.xk, self.ys, self.yk = layers, x_shift, x_scale, y_shift, y_scale

    def __call__(self, x):
        h = (np.asarray(x, np.float64) - self.xs) / self.xk
        for i, (w, b) in enumerate(self.layers):
            h = h @ w.T + b
            if i < len(self.layers) - 1:
                h = np.maximum(h, 0.0)
        return h * self.yk + self.ys


def train_mlp(X, Y, steps=4000, seed=0, hidden=256, batch=256, lr=1e-3, norm="mean_std", obs_noise=None):
    """MSE regression X -> Y with a 2-hidden-layer ReLU MLP (Adam). norm: 'mean_std', 'min_max' or 'none'
    (applied to inputs and targets). obs_noise: per-input-feature std of Gaussian noise added to each training
    batch (in raw units), a regularizer used in lesson 28."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    X, Y = np.asarray(X, np.float32), np.asarray(Y, np.float32)
    if norm == "mean_std":
        xs, xk, ys, yk = X.mean(0), X.std(0) + 1e-6, Y.mean(0), Y.std(0) + 1e-6
    elif norm == "min_max":            # map [min, max] to [-1, 1]
        xs, xk = (X.max(0) + X.min(0)) / 2, (X.max(0) - X.min(0)) / 2 + 1e-6
        ys, yk = (Y.max(0) + Y.min(0)) / 2, (Y.max(0) - Y.min(0)) / 2 + 1e-6
    else:
        xs, xk, ys, yk = np.zeros(X.shape[1]), np.ones(X.shape[1]), np.zeros(Y.shape[1]), np.ones(Y.shape[1])
    xt, yt = torch.tensor(X), torch.tensor((Y - ys) / yk, dtype=torch.float32)
    XS, XK = torch.tensor(xs, dtype=torch.float32), torch.tensor(xk, dtype=torch.float32)
    NZ = None if obs_noise is None else torch.tensor(obs_noise, dtype=torch.float32)
    net = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                        nn.Linear(hidden, Y.shape[1]))
    opt = torch.optim.Adam(net.parameters(), lr)
    for _ in range(steps):
        idx = torch.randint(0, len(xt), (batch,), generator=g)
        x = xt[idx]
        if NZ is not None:
            x = x + torch.randn(x.shape, generator=g) * NZ
        loss = ((net((x - XS) / XK) - yt[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    layers = [(m.weight.detach().double().numpy(), m.bias.detach().double().numpy()) for m in net if isinstance(m, nn.Linear)]
    return MLPPolicy(layers, xs.astype(np.float64), xk.astype(np.float64), ys.astype(np.float64), yk.astype(np.float64))


def rmse_deg(pol, X, Y):
    return float(np.degrees(np.sqrt(np.mean((pol(X) - Y) ** 2))))


def main():
    t0 = time.process_time()
    print(f"numpy {np.__version__} · torch {torch.__version__}")
    D = load_demos()
    O, A, EP, FR, TS = D["obs"], D["action"], D["episode_index"], D["frame_index"], D["timestamp"]
    n_ep = int(EP.max()) + 1
    starts = np.flatnonzero(np.r_[True, EP[1:] != EP[:-1]])
    print(f"\n[A] Anatomy of the lesson-26 dataset")
    print(f"episodes {n_ep}, frames {len(O)}, fps {int(D['fps'])}, task '{D['task']}', "
          f"successful episodes {int(D['episode_success'].sum())}/{n_ep}")
    print(f"episode boundaries (first index of each episode): {starts[:4].tolist()} ... {starts[-1]}")
    print("first frames of episode 1 as rows (index, episode_index, frame_index, timestamp, task_index, "
          "observation.state[:3], action[:3]):")
    for i in range(starts[1], starts[1] + 3):
        print(f"  {i:>5} {EP[i]:>3} {FR[i]:>4} {TS[i]:>6.2f} {0:>2}   {np.round(O[i, :3], 3)}   {np.round(A[i, :3], 3)}")
    S_o, S_a = stats(O), stats(A)
    print("\nstatistics (observation.state | action), like meta/stats.json; joints in rad, cube in m")
    print(f"{'feature':>14}{'mean':>9}{'std':>9}{'min':>9}{'max':>9}{'q01':>9}{'q99':>9}")
    for name, S, j in [(n, S_o, j) for j, n in enumerate(D["obs_names"])] + \
                      [("act." + n, S_a, j) for j, n in enumerate(D["action_names"])]:
        print(f"{name:>14}" + "".join(f"{S[k][j]:>9.3f}" for k in ("mean", "std", "min", "max", "q01", "q99")))

    # fixed held-out test episodes (never used for training or validation)
    test_eps = np.arange(100, 120)
    te = np.isin(EP, test_eps)
    Xte, Yte = O[te], A[te]
    rng = np.random.default_rng(27)

    print("\n[B] Normalization: 80 training episodes (0-79), validation on episodes 80-99, same MLP, 4000 steps")
    tr, va = EP < 80, (EP >= 80) & (EP < 100)
    print(f"{'normalization':>14}{'val RMSE deg':>14}{'worst joint':>13}{'its RMSE deg':>14}")
    mm = O.copy()
    mm[:, 6:9] *= 1000.0                       # the same data with the cube position logged in mm
    for label, X, norm in (("none", O, "none"), ("none, cube mm", mm, "none"),
                           ("mean_std", O, "mean_std"), ("mean_std, mm", mm, "mean_std"), ("min_max", O, "min_max")):
        pol = train_mlp(X[tr], A[tr], norm=norm)
        per = np.degrees(np.sqrt(np.mean((pol(X[va]) - A[va]) ** 2, axis=0)))
        print(f"{label:>14}{rmse_deg(pol, X[va], A[va]):>14.3f}{D['action_names'][per.argmax()]:>13}{per.max():>14.3f}")

    print("\n[C] Frame-level vs episode-level split (MEAN_STD, 4000 steps); test = 20 unseen episodes (100-119)")
    print(f"{'train eps':>10}{'split':>9}{'val RMSE deg':>14}{'test RMSE deg':>15}{'test / val':>12}")
    for n in (10, 30, 90):
        eps = rng.permutation(100)[:n]
        sel = np.isin(EP, eps)
        idx = np.flatnonzero(sel)
        # frame-level: shuffle all frames of these episodes, 20% to validation
        perm = rng.permutation(idx)
        cut = int(0.8 * len(perm))
        f_tr, f_va = perm[:cut], perm[cut:]
        # episode-level: 20% of the episodes (all their frames) to validation
        n_va = max(2, n // 5)
        e_va = eps[:n_va]
        e_tr_m, e_va_m = sel & ~np.isin(EP, e_va), np.isin(EP, e_va)
        for split, (a, b) in (("frame", (f_tr, f_va)), ("episode", (e_tr_m, e_va_m))):
            pol = train_mlp(O[a], A[a])
            v, t = rmse_deg(pol, O[b], A[b]), rmse_deg(pol, Xte, Yte)
            print(f"{n:>10}{split:>9}{v:>14.3f}{t:>15.3f}{t / v:>12.2f}")
    print(f"CPU time {time.process_time() - t0:.0f} s", file=sys.stderr)


if __name__ == "__main__":
    main()
