# 31 · Action chunking and ACT

<!-- meta -->
| | |
|---|---|
| **Part** | [7 · Learning from demonstration](README.md) |
| **Run type** | CPU · Mac or Colab CPU |
| **Prerequisites** | [28](28-behavior-cloning.md), [30](30-diffusion-and-flow-policies.md) |
| **Learn first** | [The Full Transformer — Encoder + Decoder](https://aiengineeringfromscratch.com/lesson?path=phases/07-transformers-deep-dive/05-full-transformer) · [Autoencoders & Variational Autoencoders (VAE)](https://aiengineeringfromscratch.com/lesson?path=phases/08-generative-ai/02-autoencoders-vae) |
| **Time** | ~75 min |
| **Code** | [`experiments/31-act/experiment.py`](../experiments/31-act/experiment.py) |
| **Status** | ✅ done |
<!-- /meta -->

> Predict the next k actions, not the next one: a chunk can reach past a pause that the observation cannot time.

## The problem

A behavior-cloning policy maps the current observation to the next action. Demonstrations contain moments where the observation stays the same while the right action changes with time: the arm waits while a gripper closes, then moves on. The ACT paper ([Zhao et al. 2023](https://arxiv.org/abs/2304.13705), Sec. I) says predicting action sequences "helps tackle temporally correlated confounders", such as pauses in demonstrations. In its chunk-size ablation (Fig. 8a: 2 simulated tasks × scripted/human data, averaged, no ensembling) success rises from 1% at $k = 1$ to 44% at $k = 100$.

Our version, in simulation: a planar arm reaches a target, holds still for 0.5 s, then goes to a drop point. A single-step policy trained on 100 demonstrations ends up **stuck at the target in 543 of 600 episodes**.

## Core idea

**Action chunking.** Instead of $\pi(a_t \mid o_t)$, learn $\pi(a_{t:t+k} \mid o_t)$. Executing a chunk and querying again every $k$ steps "reduces the effective horizon of the task by k-fold" (Sec. I): fewer decisions, fewer chances for errors to compound ([lesson 29](29-dagger.md)).

**Why a pause breaks a single-step policy.** During a pause the observation (joints at rest) barely changes, and in the data the next action is "hold" at every pause step but the last. ACT's L1 loss is minimized by the per-dimension median: hold, forever (L2 would give the mean, a slow creep; [lesson 30](30-diffusion-and-flow-policies.md)). In a chunk predicted from that observation, entry $i$ is "already moving" in a growing fraction of the demonstrations, so the median chunk holds for a while and then departs, if $k$ is long enough to contain the departure.

**Two ways to execute a chunk.**

| | open-loop chunks | temporal ensembling |
|---|---|---|
| query | every $k$ steps | every step |
| action at step $t$ | entry $t \bmod k$ of the last chunk | weighted mean of all chunk entries predicted for $t$ |
| reacts to disturbances | once per chunk | every step |
| chunk boundaries | jumps | smoothed |

Temporal ensembling (Sec. IV-A) averages the (up to $k$) predictions for the current step:

$$a_t = \frac{\sum_i w_i\, \hat a_t^{(i)}}{\sum_i w_i}, \qquad w_i = e^{-m i},$$

where $i = 0$ is the oldest prediction, so older predictions weigh more and $m \to 0$ is a plain average; in the paper's words, "a smaller m means faster incorporation" of new observations. The paper gives no value; LeRobot's config comment says the original uses $m = 0.01$ ([`configuration_act.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/act/configuration_act.py)).

**ACT's architecture** (Sec. IV; see AIEFS: [Full Transformer](https://aiengineeringfromscratch.com/lesson?path=phases/07-transformers-deep-dive/05-full-transformer), [VAE](https://aiengineeringfromscratch.com/lesson?path=phases/08-generative-ai/02-autoencoders-vae)):

| part | what it does |
|---|---|
| CVAE encoder (training only) | transformer over [CLS], joint positions and the demonstrated chunk → diagonal Gaussian over a "style variable" $z$ |
| backbone | ResNet18 per camera; four 480×640 images → 15×20×512 feature maps with 2D sinusoidal position embeddings |
| transformer encoder | image features + joint positions + $z$, each projected to 512 dimensions |
| transformer decoder | cross-attention, outputs a $k \times 14$ chunk (two 7-DoF arms) |
| loss, inference | L1 reconstruction + KL; at test time $z = 0$ (prior mean), deterministic |

About 80M parameters, ~5 hours of training per task on one RTX 2080 Ti, 50 Hz control, $k = 100$, 50 demonstrations for most real tasks. Without the CVAE objective success fell from 35.3% to 2% on human data, with almost no difference on scripted data (Fig. 8c, 2 simulated tasks); ensembling added 3.3% for ACT (Fig. 8b).

## Experiment

Full script: [`experiments/31-act/experiment.py`](../experiments/31-act/experiment.py). MuJoCo 3.17.0, NumPy 2.5.4, PyTorch 2.14.1 (CPU, 2 threads), Apple Silicon Mac (Apple M5), deterministic seeds. Wall times in `output.txt` are approximate (heavily loaded machine).

**Task.** A 2-link planar arm (0.30 m + 0.25 m, position servos `kp=20`, `kv=1.5`), 50 Hz control, random torque disturbance (std 0.2 N·m per joint, resampled every control step). The scripted demonstrator moves toward the target's IK solution (target lead clipped at 0.12 rad), starts an internal timer when within 1 cm, holds for 25 steps (0.5 s), then moves to a fixed drop pose. The policy sees $(q, \dot q, q_{goal})$: no timer, no phase. 100 demonstrations × 150 steps.

**Kept from ACT:** chunks of absolute joint targets, L1 loss, open-loop vs ensembling with $m = 0.01$. **Dropped:** images, transformer, CVAE (it mattered least on scripted data, which ours is). Policy: MLP 6-256-256-$2k$, 3,000 Adam steps, one model per $k$ and training seed (3 seeds).

```python
for t in range(len(o)):                                # dataset: o_t -> a_{t..t+k-1}
    X.append(o[t]); Y.append(a_pad[t:t + k].reshape(-1))

w = np.exp(-m * (k - 1 - np.arange(k)))                # chunk offset j: older prediction, larger weight
chunk = pol(obs)                                       # temporal ensembling: query every step
num[:, t:t + k] += w[None, :, None] * chunk
den[:, t:t + k] += w[None, :, None]
a = num[:, t] / den[:, t]
```

**Success:** within 3 cm of the target for ≥ 0.2 s, then within 3 cm of the drop point at 4.0 s; 200 new targets per seed, 600 episodes per row, same disturbances in every row. Failures: *miss* (never reached the target), *short* (left before 0.2 s), *stuck* (still there at 4.0 s). Pause: median longest stay at the target in successes. Jerk: RMS third difference of the commanded joint targets (rad/s³).

**Prediction:** small $k$ gets stuck; large open-loop chunks react too rarely; ensembling smooths.

| policy | success | 95% CI | miss | short | stuck | pause (s) | jerk |
|---|---|---|---|---|---|---|---|
| demonstrator | 200/200 | 98.1–100% | 0 | 0 | 0 | 0.66 | 3506 |
| $k=1$ | 18/600 | 1.9–4.7% | 2 | 26 | **543** | 0.44 | 2443 |
| $k=4$ open-loop | 20/600 | 2.2–5.1% | 0 | 27 | 535 | 0.33 | 1734 |
| $k=4$ ensembled | 14/600 | 1.4–3.9% | 1 | 23 | 545 | 0.59 | 677 |
| $k=8$ open-loop | 32/600 | 3.8–7.4% | 3 | 25 | 534 | 0.37 | 1694 |
| $k=8$ ensembled | 23/600 | 2.6–5.7% | 2 | 34 | 538 | 0.82 | 376 |
| $k=16$ open-loop | 431/600 | 68.1–75.3% | 62 | 87 | 6 | 0.48 | 3085 |
| $k=16$ ensembled | 160/600 | 23.3–30.3% | 5 | 1 | 362 | 0.81 | 543 |
| $k=32$ open-loop | 297/600 | 45.5–53.5% | 200 | 103 | 0 | 0.38 | 3875 |
| $k=32$ ensembled | **552/600** | 89.6–93.9% | 39 | 9 | 0 | 0.58 | **433** |

Why the jump between 8 and 16? Query each policy (seed 0) at rest exactly on the goal, for 200 goals, and find the first chunk entry that leaves the goal by more than 0.02 rad:

| $k$ | 1 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| chunks containing a departure | 0/200 | 0/200 | 8/200 | 200/200 | 200/200 |
| median first departure index | – | – | 6.0 | 9.0 | 8.0 |

The chunk predicted from the pause state holds for about 8–9 steps, then departs. A shorter chunk is all "hold": $k \le 8$ is stuck in 534–545 of 600 episodes under either scheme. Above that threshold the scheme decides:

- **$k=16$ open-loop (71.8%)** commits to the whole chunk, so its departure entries are executed. **$k=16$ ensembled (26.7%)** averages departing entries (from older queries) with holding entries (from newer queries at the same resting state). A hypothesis consistent with the 362 stuck episodes, not instrumented: the averaged command moves too little to leave the pause state, so the next queries say "hold" again.
- **$k=32$ open-loop (49.5%)** misses the target in 200 of 600 episodes: 0.64 s of blind commands cannot correct the disturbance at a 3 cm tolerance. Ensembling the same models gives **92.0%**, the best row.
- **Jerk.** Ensembling lowers command jerk at every $k$ (1734 → 677 at $k=4$, 3875 → 433 at $k=32$), below even the demonstrator (3506), which reacts to the noisy state every step.

## Break it

Lengthen the pause; keep $k = 32$ ensembled (seed 0, 200 episodes, budget = pause + 3.5 s):

| pause | 0.5 s | 1.0 s | 1.5 s | 2.0 s |
|---|---|---|---|---|
| success | 185/200 | 65/200 | 2/200 | 0/200 |
| stuck | 0 | 1 | 196 | 154 |

The chunk covers 0.64 s. Consistent with the median argument: once, from the resting state, most demonstrations are still holding at the end of the chunk, the chunk is all "hold" and the arm never leaves (196/200 stuck at 1.5 s). At 1.0 s, 126 of 200 episodes left the target but were not at the drop point when the budget ran out (not broken down further). Rule of thumb: the chunk must be long compared to the longest unobservable wait, or the observation must contain the missing state (a timer, a gripper signal, history).

## On the real robot

- **LeRobot.** `lerobot-train --policy.type=act --dataset.repo_id=...` trains ACT; the docs recommend it as the first policy to try (~80M parameters, "a few hours for 100k training steps on a single GPU") and evaluate on an SO-101 with `lerobot-rollout --robot.type=so101_follower ...` ([LeRobot ACT docs](https://huggingface.co/docs/lerobot/act)). Config defaults: `chunk_size=100`, `n_action_steps=100` (whole chunk open-loop), `temporal_ensemble_coeff=None`; ensembling requires `n_action_steps=1` (query every step); `kl_weight=10`, `latent_dim=32`, `n_decoder_layers=1` (per the comment, the original sets 7 but a bug uses only the first) ([`configuration_act.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/act/configuration_act.py)).
- **Chunk length is time.** 100 steps at 30 fps last 100/30 ≈ 3.3 s; by the break-it table, it must outlast the pauses in your demonstrations.
- **SO-101 procedure.** Record 50 episodes of a pick with a deliberate pause while the gripper closes. Train ACT with defaults and evaluate three settings on the same 20+ object placements: defaults, `n_action_steps=1` with `temporal_ensemble_coeff=0.01`, and a short `chunk_size`. Count successes and stuck episodes with Wilson intervals (lesson 25). Not yet measured on the real arm.

## Check yourself

1. **Easy.** Why does the $k=1$ policy stay at the target instead of leaving late?
2. **Medium.** With the measured departure index of about 9, why can $k = 16$ ensembled get stuck while $k = 16$ open-loop leaves?
3. **Hard.** Add the time since arrival (or the previous action) to the observation and rerun $k = 1$. Does the stuck count drop? When is chunking then needed?

### Answers

1. At rest on the goal, "hold" is the demonstrated next action at every pause step but the last; the L1-optimal output, the median, is hold. Measured: 543/600 stuck.
2. Likely mechanism (not instrumented): of the 16 predictions for the current step, only those made at least ~9 steps earlier contain the departure; the weighted mean with the holding ones moves little, the arm stays near the resting state, and fresh queries say hold. Open-loop executes its departure entries unmixed. Measured: 362/600 vs 6/600 stuck.
3. Not computed here; run it.

## Takeaways and sources

- A pause the observation cannot time makes an L1 single-step policy hold forever (543/600 stuck in simulation); a chunk reaches past it only if $k$ exceeds the learned hold (8–9 steps here).
- Open-loop chunks commit (71.8% at $k=16$) but react rarely (200 misses at $k=32$); ensembling at $k=32$ gave the best result (92.0%) and lowered jerk at every $k$.
- The chunk must outlast the longest unobservable wait: with a 0.64 s chunk, success fell from 185/200 at a 0.5 s pause to 0/200 at 2.0 s.

Next: [lesson 32](../part-8-rl-for-robots/32-rl-environment.md), wrapping a MuJoCo task as an RL environment.

**Sources**

- T. Z. Zhao, V. Kumar, S. Levine, C. Finn, "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware", 2023. [arXiv 2304.13705](https://arxiv.org/abs/2304.13705).
- LeRobot, [ACT documentation](https://huggingface.co/docs/lerobot/act) and [`configuration_act.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/act/configuration_act.py).
- All tables: [`experiments/31-act/output.txt`](../experiments/31-act/output.txt) (2026-10-27).
