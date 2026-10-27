"""Filters and state estimators for a noisy joint sensor (lessons 17-18 · Build it).

Plain NumPy. The encoder model, the two smoothing filters, the velocity estimate and the delay measure
(lesson 17) are checked against SciPy and the lesson's table in experiments/17-sensors-noise-filtering/build.py;
the Kalman filter (lesson 18) against hand-computed values, the information form and SciPy's discrete
algebraic Riccati solver in experiments/18-kalman-filter/build.py (and in tests/).
"""
import numpy as np


def moving_average(x, n):
    """Causal moving average: the mean of the last n samples (fewer at the start), as in lesson 17."""
    c = np.cumsum(np.insert(x, 0, 0.0))
    i = np.arange(len(x))
    lo = np.maximum(0, i + 1 - n)
    return (c[i + 1] - c[lo]) / (i + 1 - lo)



def encoder_reading(q, counts, sigma, rng):
    """What an encoder reports for true angles q: rounded to steps of 2 pi / counts, plus white noise."""
    step = 2 * np.pi / counts                       # one count, in rad
    return np.round(q / step) * step + rng.normal(0, sigma, np.shape(q))


def low_pass(x, fc, dt):
    """First-order low-pass with cutoff fc (Hz): move a fraction alpha toward each new sample."""
    tau = 1.0 / (2 * np.pi * fc)                    # time constant (s)
    alpha = dt / (tau + dt)                         # small alpha = heavy smoothing
    y = np.empty(len(x))
    y[0] = x[0]                                     # start at the first reading
    for k in range(1, len(x)):
        y[k] = y[k - 1] + alpha * (x[k] - y[k - 1])
    return y


def central_difference(y, dt):
    """Velocity from samples: (y[k+1] - y[k-1]) / 2 dt inside, one-sided at the two ends."""
    v = np.empty(len(y))
    v[1:-1] = (y[2:] - y[:-2]) / (2 * dt)
    v[0], v[-1] = (y[1] - y[0]) / dt, (y[-1] - y[-2]) / dt
    return v


def delay_samples(y, ref, max_lag=400, skip=500):
    """The shift (in samples) that best aligns a filtered signal y with the true signal ref."""
    errs = [np.mean((y[skip + lag:] - ref[skip:len(ref) - lag]) ** 2) for lag in range(max_lag)]
    return int(np.argmin(errs))

def cv_model(dt, q_c, r):
    """Constant-velocity model of one joint: state [q, qdot], the sensor reads q.

    q_c: intensity of the white acceleration noise (rad^2/s^3); r: variance of one reading (rad^2).
    """
    F = np.array([[1.0, dt], [0.0, 1.0]])                            # q += qdot * dt, qdot stays
    Q = q_c * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])   # white acceleration over one step
    H = np.array([[1.0, 0.0]])                                       # the encoder sees q only
    R = np.array([[r]])
    return F, Q, H, R


def kf_predict(x, P, F, Q):
    """Forecast one step: the state moves with the model, the uncertainty grows by Q."""
    return F @ x, F @ P @ F.T + Q


def kf_update(x, P, z, H, R):
    """Blend the forecast with a reading z; returns the new state, covariance and the gain."""
    S = H @ P @ H.T + R                            # expected spread of the innovation
    K = P @ H.T @ np.linalg.inv(S)                 # share of the surprise to believe
    x = x + K @ (np.atleast_1d(z) - H @ x)         # z - H x is the innovation
    P = (np.eye(len(x)) - K @ H) @ P               # uncertainty shrinks
    return x, P, K


def kalman_filter(z, dt, q_c, r, x0=None, P0=None):
    """Run the constant-velocity filter over readings z; returns states, covariances and the last gain."""
    F, Q, H, R = cv_model(dt, q_c, r)
    x = np.array([z[0], 0.0]) if x0 is None else x0     # start at the first reading, at rest
    P = np.diag([r, 1.0]) if P0 is None else P0         # sure of q to one reading, unsure of qdot
    xs, Ps = np.zeros((len(z), 2)), np.zeros((len(z), 2, 2))
    for k in range(len(z)):
        x, P = kf_predict(x, P, F, Q)
        x, P, K = kf_update(x, P, z[k], H, R)
        xs[k], Ps[k] = x, P
    return xs, Ps, K


def steady_state_gain(F, Q, H, R, rtol=1e-12, max_iter=100_000):
    """Iterate the covariance recursion until it stops changing; returns the constant gain K and forecast P."""
    P = Q.copy()
    for _ in range(max_iter):
        K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
        P_next = F @ (P - K @ H @ P) @ F.T + Q     # update, then predict
        done = np.allclose(P_next, P, rtol=rtol, atol=0.0)
        P = P_next
        if done:
            break
    return P @ H.T @ np.linalg.inv(H @ P @ H.T + R), P


def coverage(err, sd, k=2.0):
    """Fraction of samples whose error lies within +-k of the filter's own standard deviation."""
    return np.mean(np.abs(err) <= k * sd)
