# 18 · The Kalman filter

<!-- meta -->
| | |
|---|---|
| **Part** | [5 · Perception and estimation](README.md) |
| **Run type** | CPU · Mac or Colab CPU |
| **Prerequisites** | [17](17-sensors-noise-filtering.md) |
| **Learn first** | [Probability and Distributions](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/06-probability-and-distributions) · [Bayes' Theorem](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/07-bayes-theorem) |
| **Time** | ~60 min |
| **Code** | [`experiments/18-kalman-filter/experiment.py`](../experiments/18-kalman-filter/experiment.py) |
| **Status** | ✅ done |
<!-- /meta -->

> A Kalman filter is a moving average that knows how the joint moves, so it can smooth without lagging, and it tells you how wrong it thinks it is.

## The problem

Lesson 17 ended with a contradiction measured on the same noisy SO-101 joint signal: the best position filter had 4 ms of delay (moving average of 5 samples, 5.17 mrad), the best velocity filter needed 64 ms (65 samples, 0.12 rad/s). A controller wants both from one estimator, without delay, plus a number saying how much to trust them. Kalman's 1960 recursive filter ([Kalman 1960](https://doi.org/10.1115/1.3662552)) does this when you can write a linear model of the motion.

## Core idea

State $x_k\in\mathbb R^n$, measurement $z_k$ ([Welch & Bishop, TR 95-041](https://techreports.cs.unc.edu/papers/95-041.pdf), eqs. 1.1–1.2):

$$x_k = F x_{k-1} + w_{k-1},\quad w\sim\mathcal N(0,Q) \qquad z_k = H x_k + v_k,\quad v\sim\mathcal N(0,R)$$

**Predict:** $\hat x^-_k = F\hat x_{k-1}$, $\;P^-_k = F P_{k-1} F^\top + Q$.
**Update:** $K_k = P^-_k H^\top (H P^-_k H^\top + R)^{-1}$, $\;\hat x_k = \hat x^-_k + K_k(z_k - H\hat x^-_k)$, $\;P_k = (I - K_k H)P^-_k$.

The Bayesian reading is in AIEFS (see AIEFS: [Bayes' Theorem](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/07-bayes-theorem)). Here we need the engineering version.

**Constant-velocity model for one joint.** $x = [q,\ \dot q]^\top$, the encoder sees only $q$:

$$F = \begin{bmatrix}1 & \Delta t\\ 0 & 1\end{bmatrix},\quad H = [1\ \ 0],\quad Q = q_c\begin{bmatrix}\Delta t^3/3 & \Delta t^2/2\\ \Delta t^2/2 & \Delta t\end{bmatrix}$$

$Q$ comes from modelling acceleration as white noise of intensity $q_c$ (rad²/s³). $R$ is the encoder variance: here $\sigma^2 + \Delta^2/12$.

| Knob | Physical meaning | How to set it |
|---|---|---|
| $R$ | measurement noise variance | measure it: log the sensor at rest or along a known motion ([Welch & Bishop](https://techreports.cs.unc.edu/papers/95-041.pdf), §1 "Filter Parameters and Tuning") |
| $q_c$ | how hard the joint can accelerate unannounced | tune it: the only real design choice |

Two facts worth knowing before the experiment: with constant $Q$ and $R$, $P_k$ and $K_k$ converge to constants, so the steady-state filter is a fixed linear filter (Welch & Bishop say the same). And once the start-up transient has passed, the estimate depends only on the ratio $q_c/R$; the absolute scale only changes the reported $P$.

## Experiment

**Setup.** The same signal as lesson 17: SO-101 `shoulder_pan` in MuJoCo, 2 ms timestep, the same trajectory, a 12-bit encoder model plus 10 mrad white noise, same seed, so the noise sequence is identical. $R = 1.002\times10^{-4}$ rad² (σ 10.01 mrad). We sweep $q_c$ over six decades and measure position and velocity RMS error, delay, and **2σ coverage**: the fraction of samples where the true position lies within $\pm2\sqrt{P_{00}}$. For a Gaussian error that matches its claimed $P$, coverage is about 95 %.

```python
F = np.array([[1.0, DT], [0.0, 1.0]]); H = np.array([[1.0, 0.0]])
Q = q_psd * np.array([[DT**3 / 3, DT**2 / 2], [DT**2 / 2, DT]])
for k in range(len(z)):
    x = F @ x;  P = F @ P @ F.T + Q                       # predict
    S = H @ P @ H.T + r;  K = (P @ H.T) / S               # gain
    x = x + (K * (z[k] - H @ x)).ravel();  P = (np.eye(2) - K @ H) @ P   # update
```

**Predict.** Too small $q_c$: the filter trusts the constant-velocity model, smooths a lot, lags, and is overconfident. Too large: it follows the noise.

**Run** `python experiments/18-kalman-filter/experiment.py` (MuJoCo 3.17.0, NumPy 2.5.4, Apple M5, deterministic).

| $q_c$ | Position RMS (mrad) | Delay (ms) | Velocity RMS (rad/s) | $K_q$ | $K_{\dot q}$ | 2σ coverage (%) |
|---|---|---|---|---|---|---|
| 0.001 | 20.71 | 16.0 | 0.250 | 0.0235 | 0.140 | 8.0 |
| 0.01 | 6.91 | 4.0 | 0.138 | 0.0414 | 0.437 | 32.4 |
| 0.1 | 3.23 | 0.0 | **0.084** | 0.0724 | 1.361 | 91.0 |
| **1** | **3.15** | 0.0 | 0.093 | 0.1251 | 4.179 | 97.8 |
| 10 | 4.02 | 0.0 | 0.203 | 0.2116 | 12.545 | 98.1 |
| 100 | 5.22 | 0.0 | 0.482 | 0.3448 | 36.165 | 97.8 |
| 1000 | 6.61 | 0.0 | 1.136 | 0.5285 | 97.018 | 97.3 |

Lesson 17 baselines on the same data: MA N=5 gives 5.17 mrad at 4.0 ms delay; MA N=65 gives 0.120 rad/s velocity error.

**Observe.**

1. At $q_c = 1$ the filter beats the best lesson-17 position filter (3.15 vs 5.17 mrad) with zero measured delay, and at $q_c = 0.1$ it beats the best velocity filter (0.084 vs 0.120 rad/s), from the same single filter. The baselines used `np.gradient`, which peeks one sample ahead; the Kalman filter is strictly causal and still wins.
2. Zero delay is not magic: the prediction step extrapolates with the estimated velocity, so a smooth motion is not "averaged into the past".
3. The gains are constants after the start-up transient (the table shows the final $K$). A steady-state Kalman filter is a fixed two-state linear filter whose gains you could hard-code.
4. Coverage is the honesty check. At $q_c = 0.001$ the filter claims a small $P$ while lagging 16 ms behind: only 8.0 % of the samples fall inside its own 2σ band. Never use $P$ from a mistuned filter for decisions such as gating outliers or fusing sensors.

## Break it

**Wrong R.** True noise fixed, the filter is told $R_{\text{used}} = f\cdot R$, with $q_c = 1$:

| $R_{\text{used}}/R$ | Position RMS (mrad) | Velocity RMS (rad/s) | 2σ coverage (%) |
|---|---|---|---|
| 0.01 | 5.22 | 0.482 | 17.5 |
| 0.1 | 4.02 | 0.203 | 53.9 |
| 1 | 3.15 | 0.093 | 97.8 |
| 10 | 3.23 | 0.084 | 100.0 |
| 100 | 6.91 | 0.138 | 100.0 |

The RMS columns repeat the $q_c$ sweep exactly: $R\times0.01$ at $q_c=1$ gives the same 5.22 mrad / 0.482 rad/s as $q_c = 100$ with the true $R$. The estimate only sees $q_c/R$. Coverage does not repeat: underestimating $R$ makes the filter overconfident (17.5 %), overestimating it makes it pessimistic (100 %). So RMS error can look fine while the covariance is wrong.

**Unmodeled acceleration.** Add a 0.3 rad step to the joint command at $t = 5$ s. The position actuator answers with a true peak velocity of 4.05 rad/s, an acceleration the constant-velocity model calls "noise".

| $q_c$ | Max position error in 0.5 s after step (mrad) | Max velocity error (rad/s) | Position RMS before step (mrad) |
|---|---|---|---|
| 1 | 44.1 | 2.92 | 3.22 |
| 100 | 19.6 | 2.57 | 5.26 |
| 10000 | 22.3 | 7.74 | 7.98 |

The tuned filter lags the step by up to 44.1 mrad. Raising $q_c$ a hundredfold halves the transient but costs smooth-motion accuracy (5.26 vs 3.22 mrad), and at $10^4$ the filter follows noise and velocity error explodes. The mechanism: the model has no way to represent a known command. The fix is not a bigger $Q$ but a better model: feed the commanded acceleration or motor torque as the control input $Bu$ (Welch & Bishop eq. 1.1), or switch $Q$ during known transients.

## On the real robot

- $R$ is measured, not guessed: log each SO-101 joint in a held pose and along a slow ramp (procedure in [lesson 17](17-sensors-noise-filtering.md)). Not yet measured on the real arm.
- Serial-bus reads arrive with jitter. Build $F$ and $Q$ from the actual time since the last sample, not a nominal $\Delta t$; both are explicit functions of $\Delta t$ above.
- The SO-101 servos close their own position loop internally (LeRobot puts them in position mode and writes their PID gains, [`so_follower.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/so_follower.py)); a host-side Kalman filter improves the velocity and state estimates you log, feed to a policy, or use for collision detection, not the servo's internal loop.
- If you know the command sent to each joint (you do), put it into the model before increasing $Q$. The step test above is the failure you will otherwise see at every fast start and stop.

## Check yourself

1. **Easy.** In the table, which $q_c$ would you choose if you only cared about velocity? About position?
2. **Medium.** Explain why $R_{\text{used}} = 0.1R$ at $q_c = 1$ gives exactly the same RMS as the true $R$ at $q_c = 10$.
3. **Hard.** Extend the state to $[q,\dot q,\ddot q]$ (constant-acceleration model) and rerun the step test. Does the transient shrink without hurting the smooth part?

### Answers

1. Velocity: $q_c = 0.1$ (0.084 rad/s). Position: $q_c = 1$ (3.15 mrad).
2. Scaling $Q$ and $R$ (and the initial $P$, whose influence is gone after the transient) by the same factor scales $P$ by that factor and leaves $K = P^-H^\top(HP^-H^\top+R)^{-1}$ unchanged. $q_c=1, R/10$ is the same ratio as $q_c=10, R$.
3. Not computed here; run it. Expect a smaller lag at the step and a noisier velocity on the smooth part, because the extra state also has to be estimated from position alone.

## Takeaways and sources

- A constant-velocity Kalman filter gives position and velocity from one noisy encoder, causally and without the delay of averaging filters (3.15 mrad and 0.084–0.093 rad/s here).
- Only $q_c/R$ sets the estimate; the absolute values set the covariance. Check coverage before trusting $P$.
- Unmodeled acceleration shows up as transient lag; fix the model (control input) before inflating $Q$.

Next: [lesson 19](19-camera-calibration.md), the camera model and calibration.

Sources:
- R. E. Kalman, "A New Approach to Linear Filtering and Prediction Problems", *Journal of Basic Engineering* 82(1):35–45, 1960. [doi:10.1115/1.3662552](https://doi.org/10.1115/1.3662552)
- G. Welch, G. Bishop, "An Introduction to the Kalman Filter", TR 95-041, UNC Chapel Hill (updated 2004). [PDF](https://techreports.cs.unc.edu/papers/95-041.pdf)
- Signal and noise model: [lesson 17](17-sensors-noise-filtering.md).
- All numbers above: [`experiment.py`](../experiments/18-kalman-filter/experiment.py), output in [`output.txt`](../experiments/18-kalman-filter/output.txt).
