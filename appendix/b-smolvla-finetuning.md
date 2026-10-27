# Appendix B · Fine-tuning SmolVLA

> A procedure, not a result: how LeRobot's documentation fine-tunes SmolVLA on your own SO-101 data. We have not run it, so this page contains no numbers of ours.

**What you fine-tune.** `lerobot/smolvla_base` is a 450M-parameter VLA (about 100M in a flow-matching action expert) on SmolVLM-2, pretrained on 481 community datasets ([arXiv 2506.01844](https://arxiv.org/abs/2506.01844); architecture and our parameter count in [lesson 41](../part-10-foundation-models-and-agents/41-vla.md)). Its [`config.json`](https://huggingface.co/lerobot/smolvla_base/blob/main/config.json) sets `freeze_vision_encoder: true` and `train_expert_only: true`, `chunk_size` 50, learning rate 1e-4 with 1000 warm-up and 30,000 decay steps: by default fine-tuning updates the action expert, not the vision-language backbone.

**Procedure** (from the [LeRobot SmolVLA guide](https://huggingface.co/docs/lerobot/smolvla)):

1. Install LeRobot, then `pip install -e ".[smolvla]"`.
2. Record a dataset with the SO-101 leader/follower ([recording guide](https://huggingface.co/docs/lerobot/il_robots)). The guide recommends "~50 episodes"; its reference dataset has 50 episodes over 5 cube positions, 10 per position, and notes that 25 episodes "was not enough". Cover every variation (e.g. object position) with several episodes.
3. Fine-tune on a CUDA GPU:

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=${HF_USER}/mydataset \
  --batch_size=64 --steps=20000 \
  --output_dir=outputs/train/my_smolvla \
  --job_name=my_smolvla_training \
  --policy.device=cuda --wandb.enable=true
```

   The guide states that 20k steps take "~4 hrs on a single A100" and links a Colab notebook for users without a GPU. Start with a smaller batch if memory is short.
4. Run it on the arm with `lerobot-rollout --robot.type=so101_follower ... --task="<the same task text you recorded>" --policy.path=${HF_USER}/<model>`; the task string must match the recording.

**Engineering checks before step 4.** Keep the language instruction identical to the recordings; verify the camera names and resolution match the dataset; evaluate first in a held-out position with the safety limits of [lesson 15](../part-4-dynamics-and-control/15-safety.md) (`max_relative_target` set); report success as k/n with a confidence interval ([lesson 25](../part-7-learning-from-demonstration/25-evaluating-policies.md)). **Not yet run or measured by us.**

Sources: [LeRobot SmolVLA docs](https://huggingface.co/docs/lerobot/smolvla) · [SmolVLA, arXiv 2506.01844](https://arxiv.org/abs/2506.01844) · [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base) (Apache-2.0).
