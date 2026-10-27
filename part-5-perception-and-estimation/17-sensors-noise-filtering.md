# 17 · Sensors, noise and filtering

<!-- meta -->
| | |
|---|---|
| **Part** | [5 · Perception and estimation](README.md) |
| **Run type** | CPU · Mac or Colab CPU |
| **Prerequisites** | [12](../part-4-dynamics-and-control/12-joint-control.md) |
| **Learn first** | [Probability and Distributions](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/06-probability-and-distributions) · [Stochastic Processes](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/22-stochastic-processes) |
| **Time** | ~45 min |
| **Code** | [`experiments/17-sensors-noise-filtering/experiment.py`](../experiments/17-sensors-noise-filtering/experiment.py) |
| **Status** | ✅ done |
<!-- /meta -->

> Every filter buys lower noise with delay; the right window is the one where the two errors are equal for the motion you actually have.

## The problem

A joint controller ([lesson 12](../part-4-dynamics-and-control/12-joint-control.md)) needs position and velocity. On a real arm you get neither directly: you get encoder counts with quantization and electrical noise, and you get velocity only by differentiating them. Differentiation amplifies noise by $1/\Delta t$, so a raw finite difference at 500 Hz is useless, and the obvious fix (smooth first) adds delay that a feedback loop sees as phase lag ([lesson 37](../part-9-sim-to-real/37-latency.md) treats latency in general). This lesson measures that trade-off on a simulated SO-101 joint and gives you a rule for choosing a filter window.

## Core idea

**Sensor models.** An incremental or absolute encoder with $N$ counts per revolution reports $\tilde q = \Delta\,\mathrm{round}(q/\Delta)$, $\Delta = 2\pi/N$. Quantization alone behaves like uniform noise of variance $\Delta^2/12$ when the joint moves. Add white noise $v\sim\mathcal N(0,\sigma^2)$ for electronics. An IMU gyro reports $\tilde\omega = \omega + b + n$: a slowly varying bias $b$ plus white noise. Integrating it gives an angle error that grows like $b\,t$, which is why a gyro alone cannot hold an orientation. An accelerometer measures specific force, so at rest it reads gravity, not zero.

**MuJoCo sensors.** `jointpos`, `jointvel`, `accelerometer`, `gyro` (and many others) write into `mjData.sensordata`. The `noise` attribute is only metadata: the docs state it "does not affect the simulation" ([MuJoCo Modeling · sensors](https://mujoco.readthedocs.io/en/stable/modeling.html#sensor-noise)). You add noise yourself.

**Two filters** (causal, sample time $\Delta t$):

| Filter | Update | Delay at low frequency |
|---|---|---|
| Moving average, window $N$ | $y_k = \frac1N\sum_{i=0}^{N-1} z_{k-i}$ | $\frac{N-1}{2}\Delta t$ (exact, linear phase) |
| First-order low-pass, cutoff $f_c$ | $y_k = y_{k-1} + \alpha(z_k - y_{k-1})$, $\alpha = \frac{\Delta t}{\tau+\Delta t}$, $\tau = \frac{1}{2\pi f_c}$ | $\approx\tau$ for $f \ll f_c$ |

Both cut white-noise variance roughly in inverse proportion to the delay they add (moving average: variance $\sigma^2/N$ for a delay of $\frac{N-1}{2}\Delta t$). The cost of delay is a tracking error of about $|\dot q|\cdot t_d$. So the total position error behaves like

$$e_{\text{rms}}^2 \approx \underbrace{\sigma^2 \cdot g(t_d)}_{\text{noise, falls with delay}} + \underbrace{\overline{\dot q^2}\, t_d^2}_{\text{lag, grows with delay}}$$

and has a minimum. Where it sits depends on how fast the joint moves. That is the whole lesson.

## Experiment

**Setup.** SO-101 from MuJoCo Menagerie, timestep set to 2 ms (500 Hz sampling). `shoulder_pan` follows $0.5\sin(2\pi\,0.2t) + 0.2\sin(2\pi\,0.5t)$ rad through its position actuator for 10 s. We model a 12-bit encoder (4096 counts/rev, step 1.534 mrad) plus white noise $\sigma = 10$ mrad. These are teaching values, not datasheet values. Velocity is computed with `np.gradient` (central difference) of the filtered signal. Delay is the lag that best aligns the filtered and true signals.

```python
s = spec.add_sensor(name="q_pan", type=mujoco.mjtSensor.mjSENS_JOINTPOS,
                    objname="shoulder_pan", objtype=mujoco.mjtObj.mjOBJ_JOINT)
s.noise = 0.01                          # metadata only
...
z = np.round(q / step) * step + rng.normal(0, SIGMA, q.shape)   # our encoder
y = moving_average(z, n)                # or low_pass(z, fc)
```

**Predict.** The raw central-difference velocity noise should be about $\sqrt2\,\sigma/(2\Delta t) \approx 3.5$ rad/s, larger than the true velocity. The best position window should be short, the best velocity window much longer.

**Run** `python experiments/17-sensors-noise-filtering/experiment.py` (MuJoCo 3.17.0, NumPy 2.5.4, Apple M5, deterministic).

First checks: the `jointpos` sensor declared with `noise=0.01` differs from `qpos` by exactly 0.0 rad. (Sensors are computed at the start of `mj_step`, from the state before integration; compare against that state or you see a one-step offset.) The raw reading has position RMS error 9.95 mrad and velocity RMS error 3.50 rad/s, against a true velocity RMS of 0.63 rad/s.

| Filter | Delay (ms) | Position RMS (mrad) | Velocity RMS (rad/s) |
|---|---|---|---|
| raw | 0.0 | 9.95 | 3.50 |
| MA N=3 | 2.0 | 5.91 | 1.65 |
| **MA N=5** | 4.0 | **5.17** | 1.01 |
| MA N=9 | 8.0 | 6.03 | 0.55 |
| MA N=17 | 16.0 | 10.22 | 0.29 |
| MA N=33 | 32.0 | 19.76 | 0.16 |
| **MA N=65** | 64.0 | 39.01 | **0.12** |
| MA N=129 | 128.0 | 76.52 | 0.19 |
| LPF 50 Hz | 4.0 | 5.30 | 1.35 |
| LPF 20 Hz | 8.0 | 6.02 | 0.71 |
| LPF 10 Hz | 16.0 | 10.16 | 0.39 |
| LPF 2 Hz | 78.0 | 47.33 | 0.14 |
| LPF 0.5 Hz | 274.0 | 152.59 | 0.35 |

**Observe.**

1. The measured moving-average delay equals $(N-1)\Delta t/2$ in every row. The low-pass delay matches $\tau$ (15.9 → 16.0 ms at 10 Hz) until $f_c$ approaches the signal frequencies: at 0.5 Hz it is 274 ms, not 318.3 ms, because $\tau$ is only the low-frequency limit of the phase delay.
2. Position error is minimized at about 4 ms of delay (MA N=5, 5.17 mrad) and grows almost linearly beyond it: from 17 samples on the filter makes things worse than no filter at all.
3. Velocity needs 16 times the delay (MA N=65, 64 ms, 0.12 rad/s). One filter cannot serve both. This is the gap the Kalman filter of lesson 18 closes.
4. At equal delay, the two filter types perform almost the same (MA N=9: 6.03 mrad; LPF 20 Hz: 6.02 mrad at 8 ms). The delay you accept matters more than the filter shape.

The IMU check (same script): the arm at rest reads $|a| = 9.810$ m/s² on the accelerometer. A gyro with a 0.02 rad/s bias and 0.05 rad/s noise, integrated for 10 s, reports a rotation of 0.209 rad (12.0°) for a joint that did not move.

## Break it

Keep MA N=17 (16 ms delay) and make the motion faster by scaling both frequencies by $k$:

| k | True velocity RMS (rad/s) | Raw position RMS (mrad) | MA17 position RMS (mrad) |
|---|---|---|---|
| 0.5 | 0.28 | 9.94 | 5.01 |
| 1 | 0.61 | 9.95 | 10.22 |
| 2 | 1.27 | 9.95 | 20.46 |
| 4 | 2.51 | 9.95 | 40.00 |
| 8 | 3.73 | 9.94 | 59.49 |

The raw error does not care about speed. The filtered error tracks $|\dot q|\cdot t_d$: 0.61 rad/s × 16 ms ≈ 9.8 mrad versus 10.22 measured. A window tuned on a slow test move becomes worse than no filter on a fast move. Tune filters on the fastest motion the robot will actually perform, or use a model-based filter that predicts the motion instead of averaging it.

## On the real robot

- The SO-101 follower uses six Feetech STS3215 servos with 1/345 gearing ([LeRobot SO-101 docs](https://github.com/huggingface/lerobot/blob/main/docs/source/so101.mdx)). Position comes from the servo's internal encoder over the serial bus; there is no velocity sensor and no IMU in the [Menagerie model](https://github.com/google-deepmind/mujoco_menagerie/tree/f3b6013/robotstudio_so101). We did not verify the STS3215 encoder resolution; the 4096 counts above is our modelling choice.
- On real hardware the read rate is limited by the bus, not by a 2 ms timestep, so $\Delta t$ and the delays above change. Recompute them for your loop rate.
- Procedure to measure your own $\sigma$: torque the arm in a fixed pose, log every joint for 10 s, and take the standard deviation per joint. With a pure quantizer at rest you may see zero variance; command a slow ramp and look at the residual from a fitted line instead. Not yet measured on the real arm.

## Check yourself

1. **Easy.** A 64-sample moving average runs at 200 Hz. What is its delay?
2. **Medium.** In the table, why is the best velocity window (65) much longer than the best position window (5)? Use the $1/\Delta t$ noise amplification argument.
3. **Hard.** Replace `np.gradient` with a causal backward difference $(y_k - y_{k-1})/\Delta t$. Predict the raw velocity noise, then run it. Does the best velocity window change?

### Answers

1. $(64-1)/2 \times 5$ ms $= 157.5$ ms.
2. Differentiation multiplies the remaining noise by about $1/\Delta t = 500$ s⁻¹, while the lag term only changes from $|\dot q|\,t_d$ to $|\ddot q|\,t_d$. Relative to its lag term, the noise term is far larger for velocity than for position, so the minimum of noise + lag moves to a much longer delay.
3. Predicted raw noise $\sqrt2\,\sigma/\Delta t \approx 7.1$ rad/s (twice the central-difference value, because the central difference uses a 2Δt baseline). Run it to see the new optimum; we did not.

## Takeaways and sources

- MuJoCo sensors are exact; `noise` is metadata. Quantization, white noise and gyro bias are yours to add.
- Filter delay is predictable ($\frac{N-1}{2}\Delta t$ or $\approx\tau$), and the lag error is about $|\dot q|\,t_d$. The optimum window depends on the motion speed and differs for position and velocity.
- Integrating a biased gyro drifts linearly; you need a second sensor or a model to correct it.

Next: [lesson 18](18-kalman-filter.md), the Kalman filter.

Sources:
- [MuJoCo Modeling · Sensors](https://mujoco.readthedocs.io/en/stable/modeling.html#sensor-noise): sensors do not affect the simulation; `noise` is stored standard deviation only.
- [MuJoCo XML reference · accelerometer](https://mujoco.readthedocs.io/en/stable/XMLreference.html#sensor-accelerometer): `accelerometer` outputs linear acceleration including gravity in the site frame.
- [LeRobot SO-101 docs](https://github.com/huggingface/lerobot/blob/main/docs/source/so101.mdx): follower arm with six STS3215 servos.
- [MuJoCo Menagerie · robotstudio_so101](https://github.com/google-deepmind/mujoco_menagerie/tree/f3b6013/robotstudio_so101) (Apache-2.0).
- All numbers above: [`experiment.py`](../experiments/17-sensors-noise-filtering/experiment.py), output in [`output.txt`](../experiments/17-sensors-noise-filtering/output.txt).
