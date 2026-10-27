"""Demonstration data: recording, replay and episode datasets (lessons 26-27 · Build it).

Lesson 26: cut a per-physics-step recording into (observation, action) frames, hold them back up for replay,
and re-target a recorded joint trajectory when the object moves. Lesson 27: episode bookkeeping, normalization
statistics, splits by episode, and a nearest-neighbour probe for split leaks. Plain NumPy.
"""
import numpy as np


# ---------------------------------------------------------------- lesson 26: recording and replay
def to_frames(o0, obs_steps, act_steps, k):
    """Frames every k physics steps: frame j holds the observation at its start (o0 for j = 0, else the state
    after step j*k) and the command sent at the end of its k steps."""
    n = len(act_steps) // k
    return np.vstack([o0, obs_steps[k - 1:(n - 1) * k:k]]), act_steps[k - 1:n * k:k]


def hold(frames, k):
    """Zero-order hold: repeat each frame's command for k physics steps."""
    return np.repeat(frames, k, axis=0)


def retarget(q_frames, jac_pos, shift):
    """Move a recorded joint trajectory so the tool point moves by `shift` (m) at every frame, to first order:
    q' = q + pinv(J_p(q)) shift, with J_p the 3 x n position Jacobian."""
    return np.array([q + np.linalg.pinv(jac_pos(q)) @ shift for q in q_frames])


# ---------------------------------------------------------------- lesson 27: episode datasets
def episode_starts(episode_index):
    """Row of the first frame of every episode, for rows sorted by episode."""
    return np.flatnonzero(np.r_[True, episode_index[1:] != episode_index[:-1]])


def global_index(episode_index, frame_index):
    """The dataset-wide row number `index` from (episode_index, frame_index)."""
    starts = episode_starts(episode_index)
    return starts[np.searchsorted(episode_index[starts], episode_index)] + frame_index


def feature_stats(x):
    """Per-feature statistics as a dataset's stats file stores them."""
    return dict(mean=x.mean(0), std=x.std(0), min=x.min(0), max=x.max(0),
                q01=np.percentile(x, 1, axis=0), q99=np.percentile(x, 99, axis=0), count=len(x))


def merge_stats(a, b):
    """Combine the mean/std/min/max/count of two disjoint sets of frames (pooled variance). Quantiles do not merge."""
    n = a["count"] + b["count"]
    mean = (a["count"] * a["mean"] + b["count"] * b["mean"]) / n
    m2 = (a["count"] * (a["std"] ** 2 + (a["mean"] - mean) ** 2) + b["count"] * (b["std"] ** 2 + (b["mean"] - mean) ** 2))
    return dict(mean=mean, std=np.sqrt(m2 / n), min=np.minimum(a["min"], b["min"]),
                max=np.maximum(a["max"], b["max"]), count=n)


def split_frames(rows, val_frac, rng):
    """Frame split (the wrong one for time series): shuffle the rows, the last val_frac go to validation."""
    perm = rng.permutation(rows)
    cut = int((1 - val_frac) * len(perm))
    return perm[:cut], perm[cut:]


def split_episodes(episode_index, val_episodes):
    """Episode split: every frame of the held-out episodes goes to validation. Returns (train, val) masks."""
    val = np.isin(episode_index, val_episodes)
    return ~val, val


def normalize(x, s, mode="mean_std"):
    """MEAN_STD: (x - mean) / std. MIN_MAX: map [min, max] to [-1, 1]. s comes from the training frames."""
    if mode == "mean_std":
        return (x - s["mean"]) / (s["std"] + 1e-8)
    return 2 * (x - s["min"]) / (s["max"] - s["min"] + 1e-8) - 1


def unnormalize(y, s, mode="mean_std"):
    """Inverse of normalize (for actions a policy outputs in normalized units)."""
    if mode == "mean_std":
        return y * (s["std"] + 1e-8) + s["mean"]
    return (y + 1) / 2 * (s["max"] - s["min"] + 1e-8) + s["min"]


def nearest_neighbour(X_train, Y_train, X_query, chunk=1024):
    """1-nearest-neighbour regression: the target of the closest training row (L2), chunked to bound memory."""
    out, sq = [], (X_train ** 2).sum(1)
    for i in range(0, len(X_query), chunk):
        q = X_query[i:i + chunk]
        out.append(Y_train[np.argmin(sq[None] - 2 * q @ X_train.T, axis=1)])
    return np.vstack(out)
