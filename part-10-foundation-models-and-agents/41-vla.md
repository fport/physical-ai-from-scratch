# 41 · Vision-language-action models

<!-- meta -->
| | |
|---|---|
| **Part** | [10 · Foundation models and agents](README.md) |
| **Run type** | ART · shipped results, runs on CPU |
| **Prerequisites** | [31](../part-7-learning-from-demonstration/31-act.md) |
| **Learn first** | [Embodied VLAs: RT-2, OpenVLA, π0, GR00T](https://aiengineeringfromscratch.com/lesson?path=phases/12-multimodal-ai/21-embodied-vlas-openvla-pi0-groot) · [The Full Transformer — Encoder + Decoder](https://aiengineeringfromscratch.com/lesson?path=phases/07-transformers-deep-dive/05-full-transformer) |
| **Time** | ~60 min |
| **Code** | [`experiments/41-vla/experiment.py`](../experiments/41-vla/experiment.py) |
| **Status** | ✅ done |
<!-- /meta -->

> A VLA is a policy whose observation includes a sentence. For the robot, the decisive design choice is on the output side: one token per joint, or a chunk of continuous actions.

## The problem

LeRobot's SmolVLA guide tells an SO-101 owner to record about 50 episodes, fine-tune `lerobot/smolvla_base` and run it with `--task="Grasp a lego block and put it in the bin."` ([docs](https://huggingface.co/docs/lerobot/smolvla)). Before spending GPU hours on that, an engineer needs to know what the network actually reads (which cameras, what state, how much text), what it writes (one action or fifty, tokens or real numbers), and where its parameters sit. Those answers are in public files you can inspect without downloading the weights. This lesson reads them for SmolVLA, π0 and OpenVLA, and then measures what the token-based output costs on the scripted SO-101 pick of [lesson 16](../part-4-dynamics-and-control/16-grasping-and-pick-place.md).

## Core idea

A vision-language-action model (VLA) is a policy whose observation is

$$o_t = [I_t^1,\dots,I_t^n,\ \ell_t,\ q_t]$$

camera images, the tokens of a language instruction and the proprioceptive state (joint angles), and whose output is an action chunk $A_t = [a_t, \dots, a_{t+H-1}]$ (π0 paper, [arXiv 2410.24164](https://arxiv.org/abs/2410.24164)). The body is a pretrained vision-language model (VLM): a vision encoder, a projector into the language model's embedding space, and a transformer language model (see AIEFS: [Embodied VLAs](https://aiengineeringfromscratch.com/lesson?path=phases/12-multimodal-ai/21-embodied-vlas-openvla-pi0-groot)). The robotics-specific part is the action head.

**Discrete action tokens (OpenVLA).** Each action dimension is discretized "separately into one of 256 bins", with the bin width dividing the interval between the 1st and 99th quantile of the training actions; the 256 least-used tokens of the Llama tokenizer are overwritten with these bins and the model is trained with the usual next-token loss ([arXiv 2406.09246](https://arxiv.org/abs/2406.09246), §3.2). One 7-D action is 7 tokens, decoded one after another.

**Continuous chunks by flow matching (π0, SmolVLA).** A separate "action expert" transformer attends to the VLM's features and learns a velocity field that moves Gaussian noise to the action chunk. With $A^\tau = \tau A + (1-\tau)\epsilon$, $\epsilon \sim \mathcal N(0, I)$:

$$\mathcal L(\theta) = \mathbb E\,\big\|v_\theta(A^\tau, o) - (A - \epsilon)\big\|^2, \qquad A^{\tau+\delta} = A^\tau + \delta\, v_\theta(A^\tau, o),\ \ A^0 \sim \mathcal N(0, I)$$

π0 uses 10 Euler steps (δ = 0.1) and H = 50, a 3B PaliGemma VLM plus a 300M action expert, 3.3B in total ([arXiv 2410.24164](https://arxiv.org/abs/2410.24164)). SmolVLA keeps the idea at 450M parameters (about 100M in the expert), uses only the first 16 layers of SmolVLM-2, 64 visual tokens per frame, chunks of n = 50, and interleaves cross- and self-attention in the expert ([arXiv 2506.01844](https://arxiv.org/abs/2506.01844)).

| | OpenVLA | π0 | SmolVLA |
|---|---|---|---|
| action head | 256-bin tokens, autoregressive | flow-matching expert | flow-matching expert |
| output per call | 1 action | chunk, H = 50 | chunk, n = 50 |
| pretraining data | 970k episodes, Open X-Embodiment | 10,000 h from 7 robot configurations and 68 tasks, plus OXE, DROID, Bridge | 481 community datasets, 22.9k episodes, 10.6M frames |

**Cross-embodiment.** Robots differ in action dimension. The configs below pad state and action to a fixed `max_action_dim` = 32, so a 6-D SO-101 and a higher-dimensional bimanual arm share one network. **Fine-tuning** then adapts the pretrained model to one robot and task: SmolVLA's config freezes the vision encoder and trains only the expert; OpenVLA's LoRA matches full fine-tuning "while training only 1.4% of the model parameters" ([arXiv 2406.09246](https://arxiv.org/abs/2406.09246), §5.3).

## Experiment

**A. Read the artifacts.** For each model we fetch `config.json` and only the JSON header of each `.safetensors` file (an 8-byte length plus a table of tensor names and shapes, via an HTTP Range request), then sum parameters per component.

```python
n = struct.unpack("<Q", fetch(repo, fn, (0, 7)))[0]      # header length
header = json.loads(fetch(repo, fn, (8, 7 + n)))           # {tensor: {dtype, shape, offsets}}
counts[group_of(name)] += prod(shape)
```

*Predict:* where do most of SmolVLA's 450M parameters sit, in vision, language or action?

Read from the configs: `smolvla_base` expects a 6-D state and three 3 × 256 × 256 cameras, at most 48 language tokens, outputs a 6-D action padded to 32, `chunk_size` 50, `n_action_steps` 50, 10 flow steps. `pi0_base`: three 224 × 224 cameras, 32-D state and action, chunk 50, 10 inference steps, `gemma_2b` + `gemma_300m`. `openvla-7b`: 224 × 224 images, `n_action_bins` 256, 7-D actions, normalization statistics for 25 datasets.

| parameters (from tensor shapes) | smolvla_base | pi0_base | openvla-7b |
|---|---|---|---|
| language model | 204.6M | 1981.9M | 6607.6M |
| vision encoder | 86.4M | 412.4M | 730.9M |
| action expert | 98.2M | 311.5M | – |
| LM output head(s) | 47.3M | 790.0M | 131.3M |
| vision-language connector | 11.8M | 2.4M | 71.4M |
| action / state / time projections | 1.6M | 3.2M | – |
| **total** | **450.0M** | **3501.4M** | **7541.2M** |
| header bytes fetched | 73,056 | 129,872 | 126,432 |

The totals match the papers' 450M and 7B. The stored π0 checkpoint holds 3501.4M against the paper's 3.3B; most of the difference is the action expert's 263.3M vocabulary head, which π0 does not use (actions leave through the small `action_out_proj`): 3501.4M − 263.3M = 3238.1M. PaliGemma's 526.6M head is needed: the header stores no separate token-embedding tensor, so it doubles as the tied input embedding. The three headers add up to 329,360 bytes; no weights were downloaded. Note where SmolVLA's budget goes: 204.6M language model, 98.2M action expert. Per inference call SmolVLA and π0 return 50 × 32 continuous values after 10 flow steps; OpenVLA returns 7 tokens, so 7 decoder passes per control step.

**B. Sweep the number of bins.** Take lesson 16's scripted pick (MuJoCo, 229 actions of 6 joint targets at 50 Hz), tokenize each dimension OpenVLA-style, de-tokenize to bin centres, and replay the result open loop in the same scene with the physical success test (cube bottom ≥ 4 cm, both jaws in contact). Tip error is the gripper-frame position from forward kinematics. Full script: [`experiments/41-vla/experiment.py`](../experiments/41-vla/experiment.py). MuJoCo 3.17.0, NumPy 2.5.4, Python 3.14, Apple M5, deterministic.

*Predict:* the arm joints span roughly one radian, so 256 bins should give errors of a fraction of a degree.

| bins | bits/dim | RMS arm-joint err | max arm-joint err | mean tip err | max tip err | replay | cube bottom |
|---|---|---|---|---|---|---|---|
| 1024 | 10 | 0.012° | 0.034° | 0.07 mm | 0.17 mm | lifted | 7.24 cm |
| 256 | 8 | 0.048 | 0.136 | 0.26 | 0.60 | lifted | 7.21 |
| 64 | 6 | 0.209 | 0.542 | 1.24 | 2.00 | lifted | 6.96 |
| 16 | 4 | 0.745 | 2.168 | 3.51 | 9.54 | lifted | 6.68 |
| 8 | 3 | 1.566 | 4.336 | 8.83 | 19.75 | lifted | 5.19 |
| 4 | 2 | 3.161 | 8.672 | 15.69 | 40.36 | **not lifted** | 0.00 |

The original demonstration lifted the cube to 7.37 cm; its un-tokenized 50 Hz replay to 7.25 cm. At 256 bins the error is 0.26 mm at the gripper, well below anything else in the loop: the price of tokens is not resolution. It is that one action costs one decoder pass per dimension, and that the clipping range is frozen from the training data. Here 0.00% of values fell outside it: this demonstration holds its extreme poses for many steps, so its 1st and 99th percentiles equal its minimum and maximum.

## Break it

Lower the bins. Each bit removed roughly doubles the error, as uniform quantization predicts (arm-joint RMS 0.745° → 1.566° → 3.161° for 16 → 8 → 4 bins). The grasp survives down to 8 bins with a 19.75 mm worst-case tip error, though the cube ends lower (5.19 cm). At 4 bins the worst-case tip error, 40.36 mm, is larger than the 30 mm cube, and the replay does not lift it. A token head is a quantizer with a known error bound; pick the bin count from the tolerance of the task (here: a few millimetres at the jaw), not from the vocabulary size.

## On the real robot

- **SO-101 with SmolVLA.** LeRobot's procedure: record ~50 episodes ("10 episodes" for each of 5 cube positions in their reference dataset; 25 episodes "was not enough"), fine-tune with `lerobot-train --policy.path=lerobot/smolvla_base` (20k steps "~4 hrs on a single A100"), deploy with `lerobot-rollout --robot.type=so101_follower` ([docs](https://huggingface.co/docs/lerobot/smolvla)). The GPU part is in [Appendix B](../appendix/b-smolvla-finetuning.md). **Not yet measured on our arm.**
- **Latency is a budget.** π0 reports 73 ms per chunk on an RTX 4090 with three cameras ([arXiv 2410.24164](https://arxiv.org/abs/2410.24164), Table I); OpenVLA "approximately 6Hz" on an RTX 4090 ([arXiv 2406.09246](https://arxiv.org/abs/2406.09246)). Chunking is what lets a slow model drive a 30–50 Hz servo loop; SmolVLA adds asynchronous inference so the next chunk is computed while the current one executes ([arXiv 2506.01844](https://arxiv.org/abs/2506.01844)). Mac numbers: [Appendix C](../appendix/c-inference-speed-on-mac.md).
- **Hardware for π0.** openpi lists > 8 GB GPU memory for inference, > 22.5 GB for LoRA fine-tuning, and states it is tested only on Ubuntu 22.04 ([openpi](https://github.com/Physical-Intelligence/openpi)). The safety layer of [lesson 15](../part-4-dynamics-and-control/15-safety.md) sits between any of these outputs and the servos.

## Check yourself

1. **Easy.** With OpenVLA-style tokens, how many tokens per control step does a 6-D SO-101 action need, and how many decoder passes per second at 30 Hz?
2. **Medium.** In the 256-bin row the largest joint error is 0.136°. If that is half a bin of the joint with the widest range, roughly how wide is that range?
3. **Hard.** Replace uniform bins with quantile bins (equal number of training values per bin) and rerun part B. Predict before running: which rows change most, and why?

### Answers

1. 6 tokens per step; 6 × 30 = 180 decoder passes per second (with no chunking).
2. One bin ≈ 2 × 0.136° = 0.272°, so the range ≈ 0.272° × 256 ≈ 70°.
3. Not computed here; run it.

## Takeaways and sources

- A VLA maps (images, instruction, joint state) to actions; π0 and SmolVLA emit 50-step continuous chunks from a flow-matching expert, OpenVLA emits 256-bin tokens one dimension at a time.
- The public configs and safetensors headers (329,360 bytes of headers for three models) give inputs, chunk length and the parameter budget: SmolVLA 450.0M with 98.2M in the action expert.
- On the SO-101 pick, 256 bins cost 0.26 mm of tip error; 4 bins (40.36 mm worst case) fail the grasp.

Next: [lesson 42 · LLM agents with robot tools](42-llm-agents.md).

Sources:
- Kim et al., [OpenVLA](https://arxiv.org/abs/2406.09246) (arXiv 2406.09246); config [`openvla/openvla-7b`](https://huggingface.co/openvla/openvla-7b).
- Black et al., [π0: A Vision-Language-Action Flow Model](https://arxiv.org/abs/2410.24164) (arXiv 2410.24164); config [`lerobot/pi0_base`](https://huggingface.co/lerobot/pi0_base); [openpi](https://github.com/Physical-Intelligence/openpi).
- Shukor et al., [SmolVLA](https://arxiv.org/abs/2506.01844) (arXiv 2506.01844); [LeRobot SmolVLA docs](https://huggingface.co/docs/lerobot/smolvla); config [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base) (Apache-2.0).
- All measured numbers: [`experiments/41-vla/output.txt`](../experiments/41-vla/output.txt).
