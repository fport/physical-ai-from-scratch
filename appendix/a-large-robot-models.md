# Appendix A · Large robot models: π0, GR00T, Gemini Robotics

> Three families of large robot policies, read from their papers: what they consume, what they output, how fast, and what you can run yourself.

All three are vision-language-action models in the sense of [lesson 41](../part-10-foundation-models-and-agents/41-vla.md): a pretrained vision-language backbone plus an action module that emits chunks of actions. Numbers below are quoted from the primary sources; none were measured by us except where lesson 41 is cited.

| | π0 (Physical Intelligence) | GR00T N1 (NVIDIA) | Gemini Robotics (Gemini Robotics Team) |
|---|---|---|---|
| backbone | PaliGemma, 3B | Eagle-2 VLM ("System 2"), 1.34B | Gemini 2.0 |
| action module | flow-matching action expert, 300M | diffusion transformer trained with flow matching ("System 1") | VLA backbone in the cloud + local action decoder |
| size | 3.3B | 2.2B (released GR00T-N1-2B) | not stated in the sections we read |
| chunk | H = 50 | H = 16 | action chunks |
| speed | 73 ms per chunk on an RTX 4090 (3 cameras); control up to 50 Hz | 63.9 ms per 16-action chunk on an L40 (bf16); System 2 at 10 Hz, System 1 actions at 120 Hz | backbone < 160 ms, ~250 ms observation-to-chunk, effective 50 Hz |
| data | 10,000 h dexterous data, 7 robot configurations, 68 tasks, plus OXE, DROID, Bridge | real-robot trajectories, human videos, synthetic data | large teleoperated dataset from a fleet of ALOHA 2 robots |
| hardware shown | single-arm, dual-arm, mobile manipulators | Fourier GR-1 humanoid, bimanual tasks | bimanual ALOHA 2; adapted to a bi-arm platform and a humanoid |

**π0.** Black et al. ([arXiv 2410.24164](https://arxiv.org/abs/2410.24164)). The code and checkpoints are in [openpi](https://github.com/Physical-Intelligence/openpi): base models π0, π0-FAST and π0.5, "pre-trained on 10k+ hours of robot data", plus fine-tuned checkpoints (e.g. DROID, ALOHA, LIBERO). Requirements listed there: > 8 GB GPU memory for inference, > 22.5 GB for LoRA fine-tuning, > 70 GB for full fine-tuning; tested only on Ubuntu 22.04. The LeRobot port `lerobot/pi0_base` stores 3501.4M parameters in float32 with 50-step chunks of 32-D actions (read from its files in [lesson 41](../part-10-foundation-models-and-agents/41-vla.md), [output](../experiments/41-vla/output.txt)).

**GR00T N1.** NVIDIA ([arXiv 2503.14734](https://arxiv.org/abs/2503.14734)), "an open foundation model for generalist humanoid robots". The dual-system split is the engineering idea worth copying: a slow reasoning module (10 Hz) conditions a fast action module (120 Hz) that produces 16-step chunks, trained jointly end to end.

**Gemini Robotics.** Gemini Robotics Team ([arXiv 2503.20020](https://arxiv.org/abs/2503.20020)). Two models: Gemini Robotics, a VLA that controls robots directly and can be fine-tuned to new embodiments, and Gemini Robotics-ER, which adds embodied reasoning (object detection, pointing, trajectory and grasp prediction, 3D bounding boxes). The latency split (cloud backbone, onboard decoder, chunking) shows how a large model meets a 50 Hz loop.

**What transfers to an SO-101.** All three need an NVIDIA-class GPU for the sizes above; none has been run on our arm. The small open option is SmolVLA (450M, [Appendix B](b-smolvla-finetuning.md)). Whatever model you use, its output goes through the validation layer of [lesson 42](../part-10-foundation-models-and-agents/42-llm-agents.md) and the limits of [lesson 15](../part-4-dynamics-and-control/15-safety.md).

Sources: [π0, arXiv 2410.24164](https://arxiv.org/abs/2410.24164) · [openpi](https://github.com/Physical-Intelligence/openpi) · [GR00T N1, arXiv 2503.14734](https://arxiv.org/abs/2503.14734) · [Gemini Robotics, arXiv 2503.20020](https://arxiv.org/abs/2503.20020).
