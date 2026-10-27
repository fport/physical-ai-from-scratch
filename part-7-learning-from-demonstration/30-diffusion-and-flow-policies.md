# 30 · Multimodal actions: diffusion and flow policies

<!-- meta -->
| | |
|---|---|
| **Part** | [7 · Learning from demonstration](README.md) |
| **Run type** | CPU · Mac or Colab CPU |
| **Prerequisites** | [28](28-behavior-cloning.md) |
| **Learn first** | [Diffusion Models — DDPM from Scratch](https://aiengineeringfromscratch.com/lesson?path=phases/08-generative-ai/06-diffusion-ddpm-from-scratch) · [Flow Matching & Rectified Flows](https://aiengineeringfromscratch.com/lesson?path=phases/08-generative-ai/13-flow-matching-rectified-flows) · [Loss Functions](https://aiengineeringfromscratch.com/lesson?path=phases/03-deep-learning-core/05-loss-functions) |
| **Time** | ~75 min |
| **Code** | [`experiments/30-diffusion-and-flow-policies/experiment.py`](../experiments/30-diffusion-and-flow-policies/experiment.py) · [notebook](../notebooks/30-diffusion-and-flow-policies.ipynb) |
| **Status** | ✅ done |
<!-- /meta -->

> A regression policy returns the average of the demonstrations. When the demonstrations disagree, the average can be the one motion nobody showed.

## The problem

Diffusion Policy ([Chi et al. 2023](https://arxiv.org/abs/2303.04137)) starts from multimodal action distributions: in the same state there can be several correct motions, each one a separate "mode". In their Push-T task the policy "learns to approach the contact point equally likely from left or right", while LSTM-GMM and IBC show a bias toward one side and BET "cannot commit to one mode" (Sec. 5.3).

The smallest version of that failure: a point must go from $(0,-1)$ to $(0,+1)$ past a round obstacle of radius 0.30 at the origin. Of 200 demonstrations, 100 pass on the left and 100 on the right, and the side does not depend on the start. A behavior-cloning MLP (lesson 28) trained with MSE on these demonstrations collides in **200 of 200** trials in simulation. This lesson shows why and what replaces it.

## Core idea

### In plain words

Two equally good roads go around a field, one on the left and one on the right. Half of the drivers you learn from take the left road, half the right. Now train a navigator to give *one* answer that is as close as possible, on average, to what the drivers did. The answer that minimizes the average squared distance is the midpoint of the two roads: straight through the field. That is exactly what regression with an MSE loss does. It returns the average of the demonstrations for the situation, which is fine when they agree and wrong when they split.

The fix is to stop asking for one answer and learn to *sample* one: each query returns one plausible road, left or right, about as often as the drivers chose each. A **mixture model** does this by predicting a fixed number of candidates plus their probabilities and picking one. **Generative models** (diffusion, flow matching) need no fixed number: they start from random noise and push it, step by step, until it looks like a demonstration. Flow matching learns the push directly, as a velocity field that carries noise to data. Where the random start lies decides which road you end on.

The number of pushes is a knob. Many small pushes follow the curving flow to one road. One big push from pure noise lands on the average, and you are back in the field.

```mermaid
flowchart TD
    O["observation o<br/>(start point)"] --> M{"how is the path produced?"}
    M -->|"MSE regression"| A["one answer:<br/>the average path"]
    A --> X["through the obstacle"]
    M -->|"flow matching"| Z["random noise a0"]
    Z --> V["N small pushes along the<br/>learned velocity v(a, t, o)"]
    V --> S["one sampled path:<br/>left or right"]
```

### Flow matching as code

```python
# MSE regression, for comparison: one answer per observation -> learns E[a1 | o]
loss_mse = ((net_mse(o) - a1) ** 2).mean()

# flow matching, one training step
a1, o = sample_demos(batch)               # demonstrated path (80 numbers) and its observation
a0 = torch.randn_like(a1)                 # pure noise with the same shape
t = torch.rand(batch, 1)                  # random time in [0, 1]
at = (1 - t) * a0 + t * a1                # a point on the straight line from noise to demo
target = a1 - a0                          # the velocity of that straight line
loss = ((net(at, t, o) - target) ** 2).mean()   # still MSE, but on velocities, not on paths

# sampling: Euler integration from noise (t = 0) to a path (t = 1)
def sample(net, o, n_steps):
    a = torch.randn(80)                   # a different random start can end on the other side
    for i in range(n_steps):
        a = a + net(a, i / n_steps, o) / n_steps   # one small push along the learned velocity
    return a
```

| What | Formula |
|---|---|
| MSE optimum | $f^*(o) = \mathbb{E}[a \mid o]$; two 50/50 modes give $\tfrac12(a_L + a_R)$ |
| Mixture model | $p(a \mid o) = \sum_k \pi_k(o)\,\mathcal N(a;\mu_k(o), \sigma^2 I)$ |
| Flow-matching loss | $a_t = (1-t)\,a_0 + t\,a_1, \quad \mathcal L = \mathbb{E}\,\bigl\lVert v_{\theta}(a_t, t, o) - (a_1 - a_0)\bigr\rVert^2$ |
| Sampling (Euler) | $a \leftarrow a_0, \quad a \leftarrow a + \tfrac1N\, v_{\theta}\!\left(a, \tfrac iN, o\right), \quad i = 0,\dots,N-1$ |

$o$ is the observation (here the start point), $a$ the action (here the whole path), $a_L$ and $a_R$ the left and right demonstrated paths, $\pi_k$, $\mu_k$ and $\sigma$ the weight, mean and fixed spread of mixture component $k$ out of $K$, $a_0 \sim \mathcal N(0, I)$ the noise, $a_1$ a demonstration, $t \sim U(0,1)$ the time along the path from noise to data, $v_\theta$ the learned velocity network and $N$ the number of Euler steps (each one network call).

<details>
<summary>Math details: why MSE averages, and why one Euler step is regression again</summary>

**MSE learns the conditional mean.** For any $f$, $\mathbb{E}\,\lVert a - f(o)\rVert^2 = \mathbb{E}\,\lVert a - \mathbb{E}[a\mid o]\rVert^2 + \mathbb{E}\,\lVert \mathbb{E}[a\mid o] - f(o)\rVert^2$; the first term does not depend on $f$, so the minimizer is $f^*(o) = \mathbb{E}[a \mid o]$ (see AIEFS: [Loss Functions](https://aiengineeringfromscratch.com/lesson?path=phases/03-deep-learning-core/05-loss-functions)). With two equally likely modes this is the middle of the obstacle. An L1 loss gives the per-dimension median instead; with exactly 50/50 modes any point between them is a median, so it is no better (L1 returns in [lesson 31](31-act.md)).

**Mixture model.** Train by negative log-likelihood; at run time first sample $k$ from $\pi_k(o)$, then output $\mu_k$. It works if you know $K$ and the components do not collapse onto each other.

**Generative action models** learn to *sample* from $p(a \mid o)$ without fixing $K$. The action $a$ is usually a whole chunk of future actions, and the observation $o$ is a condition, not something the model generates. The version we implement is flow matching with straight paths ($\sigma_{\min} = 0$).

**One step is regression.** At $t = 0$, $a_t = a_0$ carries no information about $a_1$, so the best velocity is $v^*(a_0, 0, o) = \mathbb{E}[a_1 \mid o] - a_0$. A single Euler step lands on $a_0 + v^* = \mathbb{E}[a_1 \mid o]$, the MSE answer. Later steps see an $a_t$ that already leans to one side, and the learned velocity bends toward that side's mode. The integration steps are what turn regression into sampling.

Go deeper (AIEFS): [Diffusion Models — DDPM from Scratch](https://aiengineeringfromscratch.com/lesson?path=phases/08-generative-ai/06-diffusion-ddpm-from-scratch), [Flow Matching & Rectified Flows](https://aiengineeringfromscratch.com/lesson?path=phases/08-generative-ai/13-flow-matching-rectified-flows), [Loss Functions](https://aiengineeringfromscratch.com/lesson?path=phases/03-deep-learning-core/05-loss-functions). Primary sources: [Lipman et al. 2023](https://arxiv.org/abs/2210.02747), [Ho et al. 2020](https://arxiv.org/abs/2006.11239), [Chi et al. 2023](https://arxiv.org/abs/2303.04137).

</details>

<!-- figure: 30-mean-vs-flow -->

### Four ways to produce an action

| | trained to predict | sampling | cost per action |
|---|---|---|---|
| MSE regression | $\mathbb{E}[a\mid o]$ | none | 1 network call |
| Mixture (MDN) | $\pi_k, \mu_k$ | pick $k$, take $\mu_k$ | 1 call |
| DDPM ([Ho et al. 2020](https://arxiv.org/abs/2006.11239)) | the noise $\epsilon$ added to $a$, given $(a_t, t, o)$ | start from Gaussian noise, denoise $N$ times | $N$ calls |
| Flow matching ([Lipman et al. 2023](https://arxiv.org/abs/2210.02747)) | a velocity field $v_{\theta}(a_t, t, o)$ | integrate an ODE from noise in $N$ steps | $N$ calls |

DDPM ("denoising diffusion") and flow matching are close relatives: both start from noise and need $N$ network calls per sample; flow matching regresses a velocity, DDPM the noise to remove.

## Experiment

Full script: [`experiments/30-diffusion-and-flow-policies/experiment.py`](../experiments/30-diffusion-and-flow-policies/experiment.py). NumPy 2.5.4, PyTorch 2.14.1 (CPU, 2 threads), Apple Silicon Mac (Apple M5), fixed seeds, deterministic. Parts A–B port the experiment of the earlier [notebook](../notebooks/30-diffusion-and-flow-policies.ipynb) and reproduce the numbers recorded for it, except the three-seed MSE check, which now gives somewhat different collision counts.

The observation is the start point $(x_0, y_0)$ with $x_0 \in [-0.05, 0.05]$; the action is the whole path, 40 waypoints relative to the start (80 numbers). A trial passes or hits the obstacle; "left/right" counts the collision-free trials by the side they passed. 200 trials per row, 95% Wilson intervals (lesson 25).

**Prediction:** the MSE path goes through the middle; the mixture and the flow policy split roughly 50/50.

```python
a0 = torch.randn(a1.shape)                          # noise
t = torch.rand(batch)
at = (1 - t[:, None]) * a0 + t[:, None] * a1        # point on the straight path
loss = ((net(at, t, o) - (a1 - a0)) ** 2).mean()    # regress the velocity, conditioned on o

a = torch.randn(n, 80)                              # sampling: Euler from t = 0 to 1
for i in range(n_steps):
    a = a + net(a, torch.full((n,), i / n_steps), o) / n_steps
```

**Part A.** NumPy MLP 64-64 tanh, 1,500 full-batch Adam epochs; mixture with $K = 2$, fixed $\sigma = 0.2$.

| model | collided | 95% CI | left | right |
|---|---|---|---|---|
| MSE, demos half left / half right | **200/200** | 98.1–100% | 0 | 0 |
| MSE, all demos on the left | 0/200 | 0–1.9% | 200 | 0 |
| Mixture (MDN, $K=2$) | **0/200** | 0–1.9% | 96 | 104 |

At the obstacle's height ($y = 0$) the demonstrations pass at $x = -0.55$ and $+0.55$; the MSE path passes at $x = +0.030$, inside the radius of 0.30. With three other seeds the MSE model collided 193, 184 and 200 times out of 200, the mixture 0 times in each.

**Part B: more data does not help.** Same MSE model, demonstrations half/half:

| demos | 10 | 20 | 50 | 100 | 200 | 500 |
|---|---|---|---|---|---|---|
| MSE collisions | 8.0% | 46.0% | 85.0% | 91.5% | 100% | 100% |

A likely reason (not instrumented): with 10 demonstrations the network ties tiny differences in $x_0$ to the side of the nearest demonstration and usually veers to one side; as data grows it approaches the true conditional mean, and the collision rate goes up.

**Part C: flow-matching policy** on the same 200 demonstrations: MLP 3×256 SiLU, input = noisy chunk + sinusoidal time embedding + observation, 4,000 Adam steps, batch 256. Sweep the number of Euler steps.

| Euler steps | collided | 95% CI | left | right | single-query time (ms) | mean $\lvert x\rvert$ at $y=0$ |
|---|---|---|---|---|---|---|
| 1 | 200/200 | 98.1–100% | 0 | 0 | 0.04 | 0.161 |
| 2 | 95/200 | 40.7–54.4% | 53 | 52 | 0.06 | 0.372 |
| 4 | 17/200 | 5.4–13.2% | 87 | 96 | 0.10 | 0.494 |
| 8 | 7/200 | 1.7–7.0% | 95 | 98 | 0.19 | 0.533 |
| 16 | 5/200 | 1.1–5.7% | 96 | 99 | 0.39 | 0.546 |
| 32 | **3/200** | 0.5–4.3% | 97 | 100 | 0.77 | 0.554 |

![Two panels against the number of Euler steps on a log scale from 1 to 32. Left: collisions fall from 100 percent at 1 step, the same as MSE regression, to 47.5 percent at 2 steps and 1.5 percent at 32 steps. Right: the mean lateral distance at the obstacle rises from 0.161, inside the obstacle radius of 0.30, toward the demonstrations' 0.55.](../experiments/30-diffusion-and-flow-policies/flow_steps.png)

Times are the fastest of 30 single-trajectory queries on a machine heavily loaded by other jobs; treat them as approximate. The network-evaluation count (= steps) is the machine-independent cost.

Observations: the flow policy splits left/right like the mixture model without being told there are two modes; the lateral offset at the obstacle approaches the demonstrations' 0.55 as steps increase; inference cost (network evaluations) grows linearly with the step count.

## Break it

Cut the integration steps. With 1 step the flow policy collides **200/200**, like MSE: the path's mean distance from the centre line at the obstacle is 0.161, inside the obstacle. That is the $t = 0$ argument above, measured; the paths are not exactly the MSE path ($x = +0.030$) because the learned field still depends a little on $a_0$. With 2 steps, about half the samples still end between the modes (95/200). Even at 32 steps 3/200 samples touch the obstacle: the exact flow would carry no probability mass between the modes, but the learned velocity field is approximate and Euler integration adds error, so a little remains.

The second break is the data, for the MSE model (200 demos, Part B sweep):

| fraction of left demos | 50% | 60% | 70% | 80% | 90% | 100% |
|---|---|---|---|---|---|---|
| MSE collisions | 100% | 91.5% | 63.0% | 36.5% | 0% | 0% |

MSE is safe only when the data is nearly unimodal (one dominant mode): the mean lateral offset $0.55\,\lvert 2p_{\text{left}} - 1\rvert$ must clear the radius 0.30 plus a margin.

## On the real robot

- **Diffusion Policy** uses a DDPM to model $p(A \mid O)$, the action sequence conditioned on observations, not their joint distribution; the CNN variant conditions through FiLM (a learned per-channel scale and shift computed from the observation). With DDIM (a faster DDPM sampler), 100 training iterations and 10 inference iterations give about 0.1 s inference on an Nvidia 3080 in their real-world setup; an action horizon of 8 steps worked best for most tasks; reported average improvement 46.9% over prior methods across 15 tasks from 4 benchmarks ([Chi et al. 2023](https://arxiv.org/abs/2303.04137)).
- **π0** uses conditional flow matching for actions, a 300M-parameter "action expert" on a 3B vision-language model (3.3B total), chunks of $H = 50$ actions, 10 integration steps; on an RTX 4090 the 10 action passes take 27 ms of a 73 ms on-board inference ([Black et al. 2024](https://arxiv.org/abs/2410.24164)). Same trade-off as Part C, at real scale.
- **SO-101 procedure.** LeRobot's diffusion policy config defaults to `horizon=64`, `n_action_steps=32`, a DDPM scheduler with 100 training timesteps ([`configuration_diffusion.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/diffusion/configuration_diffusion.py)). Record teleop demos of a task with two valid strategies (grasp a cube from either side), train an MSE behavior-cloning baseline and a diffusion policy on the same episodes, and count grasps that end between the two strategies over at least 50 trials each. Not yet measured on the real arm.

## Check yourself

1. **Easy.** Why does a 1-step flow policy behave like MSE regression?
2. **Medium.** With 80% left demonstrations, where does the MSE path pass at $y = 0$, and why do some trials still collide?
3. **Hard.** Write a closed-loop step policy (current position → next step) trained with MSE on the same demonstrations and measure its collision rate. Is the average flow at the centre line stable?

### Answers

1. At $t=0$ the input $a_0$ is independent of the target, so the optimal velocity is $\mathbb{E}[a_1\mid o] - a_0$ and one Euler step returns $\mathbb{E}[a_1\mid o]$. Measured: 200/200 collisions, same as MSE.
2. Around $0.8\cdot(-0.55) + 0.2\cdot 0.55 = -0.33$, just outside the radius 0.30; jitter in the learned path and in $x_0$ pushes many trials inside. Measured: 73/200 (36.5%).
3. Not computed here; run it.

## Takeaways and sources

- MSE regression returns the conditional mean; with two 50/50 modes it collided 200/200 in simulation, and more data (10 → 500 demos) raised the collision rate from 8% to 100%.
- A mixture model (0/200, 96 left / 104 right) and a flow-matching policy (3/200 at 32 steps, 97 / 100) represent both modes; the flow policy did not need to know the number of modes.
- Sampling steps are the knob: 1 step reproduced the mean (200/200), 8 steps reached 7/200, and cost grows linearly with steps.

Next: [lesson 31](31-act.md), predicting chunks of future actions and executing them.

**Sources**

- C. Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion", 2023. [arXiv 2303.04137](https://arxiv.org/abs/2303.04137).
- K. Black et al., "π0: A Vision-Language-Action Flow Model for General Robot Control", 2024. [arXiv 2410.24164](https://arxiv.org/abs/2410.24164).
- Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, M. Le, "Flow Matching for Generative Modeling", 2023. [arXiv 2210.02747](https://arxiv.org/abs/2210.02747).
- J. Ho, A. Jain, P. Abbeel, "Denoising Diffusion Probabilistic Models", 2020. [arXiv 2006.11239](https://arxiv.org/abs/2006.11239).
- LeRobot, [`configuration_diffusion.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/diffusion/configuration_diffusion.py).
- All tables: [`experiments/30-diffusion-and-flow-policies/output.txt`](../experiments/30-diffusion-and-flow-policies/output.txt) (2026-10-27).
