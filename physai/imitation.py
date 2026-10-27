"""Learning from demonstrations (lessons 28-31 · Build it).

Behavior cloning (normalization, an MLP policy, its training loop), DAgger with a nearest-neighbour learner,
flow matching (the training pair, the exact velocity field of a finite dataset, Euler sampling) and action
chunking with temporal ensembling. NumPy, except `train_bc`, which imports PyTorch when it is called.
Checked in experiments/{28-behavior-cloning,29-dagger,30-diffusion-and-flow-policies,31-act}/build.py
and in tests/.
"""
import numpy as np
from scipy.spatial import cKDTree


# ---------------------------------------------------------------- lesson 28 · behavior cloning
def fit_mean_std(x, eps=1e-6):
    """MEAN_STD normalization statistics per feature (lesson 27): shift and scale."""
    x = np.asarray(x, np.float32)
    return x.mean(0), x.std(0) + eps


def mlp_forward(layers, x):
    """ReLU MLP in NumPy: layers = [(W, b), ...] with W of shape (out, in); no activation on the last layer."""
    h = np.asarray(x, np.float64)
    for i, (W, b) in enumerate(layers):
        h = h @ W.T + b
        if i < len(layers) - 1:
            h = np.maximum(h, 0.0)
    return h


def train_bc(X, Y, hidden=256, steps=4000, batch=256, lr=1e-3, seed=0, obs_noise=None):
    """Behavior cloning: MSE regression from observations X to expert actions Y with a 2 x hidden ReLU MLP (Adam).

    Inputs and targets are MEAN_STD-normalized; obs_noise (raw units, one std per input feature) is added to the
    inputs of every batch, the targets stay clean. Returns (layers, x_stats, y_stats) for `bc_policy`.
    """
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    X, Y = np.asarray(X, np.float32), np.asarray(Y, np.float32)
    (xs, xk), (ys, yk) = fit_mean_std(X), fit_mean_std(Y)
    xt, yt = torch.tensor(X), torch.tensor((Y - ys) / yk, dtype=torch.float32)
    XS, XK = torch.tensor(xs, dtype=torch.float32), torch.tensor(xk, dtype=torch.float32)
    NZ = None if obs_noise is None else torch.tensor(obs_noise, dtype=torch.float32)
    net = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                        nn.Linear(hidden, Y.shape[1]))
    opt = torch.optim.Adam(net.parameters(), lr)
    for _ in range(steps):
        idx = torch.randint(0, len(xt), (batch,), generator=g)        # a random batch of frames
        x = xt[idx]
        if NZ is not None:
            x = x + torch.randn(x.shape, generator=g) * NZ           # perturb the input, keep the label
        loss = ((net((x - XS) / XK) - yt[idx]) ** 2).mean()         # MSE in normalized action units
        opt.zero_grad()
        loss.backward()
        opt.step()
    layers = [(m.weight.detach().double().numpy(), m.bias.detach().double().numpy()) for m in net if isinstance(m, nn.Linear)]
    return layers, (xs.astype(np.float64), xk.astype(np.float64)), (ys.astype(np.float64), yk.astype(np.float64))


def bc_policy(layers, x_stats, y_stats):
    """The trained network as a function raw observation(s) -> raw action(s): normalize, forward, unnormalize."""
    (xs, xk), (ys, yk) = x_stats, y_stats
    return lambda x: mlp_forward(layers, (np.atleast_2d(x) - xs) / xk) * yk + ys


# ---------------------------------------------------------------- lesson 29 · DAgger
def nearest_distance(states, frames):
    """Distance of each state to its nearest frame, max-abs over the coordinates: min_j max_i |s_i - f_ji|."""
    d, _ = cKDTree(np.atleast_2d(frames)).query(np.atleast_2d(states), p=np.inf)
    return d


def knn_policy(X, Y):
    """A learner that is exact on its data and blind elsewhere: the action of the nearest stored state."""
    tree = cKDTree(np.atleast_2d(X))
    return lambda S: Y[tree.query(np.atleast_2d(S))[1]]


def dagger(fit, expert, rollout, X, Y, starts, aggregate=True):
    """DAgger (Ross, Gordon, Bagnell 2011) with beta = 1 only for the initial demonstrations.

    fit(X, Y) -> policy; expert(states) -> action labels; rollout(policy, start) -> states the policy visited.
    starts: one list of start states per iteration. Returns the policies of iterations 0, 1, ..., len(starts).
    """
    policies = [fit(X, Y)]                                   # iteration 0 is plain behavior cloning
    for batch in starts:
        S = np.concatenate([rollout(policies[-1], s0) for s0 in batch])   # the learner drives ...
        A = expert(S)                                                   # ... the expert labels
        X, Y = (np.concatenate([X, S]), np.concatenate([Y, A])) if aggregate else (S, A)
        policies.append(fit(X, Y))
    return policies


# ---------------------------------------------------------------- lesson 30 · flow matching
def flow_matching_pair(a0, a1, t):
    """Training input and target of flow matching with straight paths: a_t on the line noise -> data, and its velocity."""
    t = np.reshape(t, (-1,) + (1,) * (np.ndim(a1) - 1))
    return (1 - t) * a0 + t * a1, a1 - a0


def empirical_velocity(a, t, data):
    """The velocity field that minimizes the flow-matching loss on a finite dataset: E[a1 - a0 | a_t = a].

    Given a1 = data[j], a_t is Gaussian around t * data[j] with std (1 - t), so the posterior over j is a softmax.
    a: (n, D) points at time t (0 <= t < 1); data: (m, D) demonstrations.
    """
    a = np.atleast_2d(a)
    d2 = ((a[:, None, :] - t * data[None, :, :]) ** 2).sum(-1)      # (n, m) squared distances
    logw = -d2 / (2 * (1 - t) ** 2)
    w = np.exp(logw - logw.max(1, keepdims=True))
    w /= w.sum(1, keepdims=True)
    return (w @ data - a) / (1 - t)                                 # sum_j w_j (data_j - a) / (1 - t)


def euler_sample(velocity, a0, n_steps):
    """Integrate da/dt = velocity(a, t) from t = 0 (noise a0) to t = 1 in n_steps equal Euler steps."""
    a = np.array(a0, dtype=float)
    for i in range(n_steps):
        a = a + velocity(a, i / n_steps) / n_steps
    return a


# ---------------------------------------------------------------- lesson 31 · action chunking
def chunk_targets(actions, k):
    """Training targets of a chunked policy: for every step t the next k actions, padded with the last action."""
    actions = np.asarray(actions)
    padded = np.concatenate([actions, np.repeat(actions[-1:], k - 1, axis=0)])
    return np.stack([padded[t:t + k] for t in range(len(actions))])


def ensembled_action(chunks, t, k, m=0.01):
    """Temporal ensembling (ACT): weighted mean of every prediction made for step t; chunks[s][j] predicts s + j.

    Weights w_i = exp(-m i) with i = 0 the oldest prediction, so older predictions weigh more for m > 0.
    """
    preds = [chunks[s][t - s] for s in range(max(0, t - k + 1), t + 1)]   # oldest first
    w = np.exp(-m * np.arange(len(preds)))
    return np.tensordot(w, np.array(preds), axes=1) / w.sum()


class TemporalEnsembler:
    """The same ensembling as a running average (one new chunk per step, O(k) memory)."""

    def __init__(self, k, m=0.01):
        self.w = np.exp(-m * np.arange(k))       # w[i]: weight of the i-th prediction for a step, oldest first
        self.cum = np.cumsum(self.w)
        self.avg, self.count = None, None

    def update(self, chunk):
        """Add the chunk predicted now (shape (k, n)); return the action to execute now."""
        chunk = np.asarray(chunk, dtype=float)
        if self.avg is None:
            self.avg, self.count = chunk.copy(), np.ones(len(chunk), dtype=int)
        else:                                    # entries 0..k-2 of the stored average are steps 1..k-1 of the new chunk
            c = self.count
            self.avg = (self.avg * self.cum[c - 1, None] + chunk[:-1] * self.w[c, None]) / self.cum[c, None]
            self.avg = np.concatenate([self.avg, chunk[-1:]])
            self.count = np.concatenate([np.minimum(c + 1, len(self.w) - 1), [1]])
        action, self.avg, self.count = self.avg[0], self.avg[1:], self.count[1:]
        return action
