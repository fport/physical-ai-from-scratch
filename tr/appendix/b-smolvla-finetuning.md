# Ek B · SmolVLA fine-tuning

> Bir sonuç değil, bir prosedür: LeRobot dokümantasyonu SmolVLA'yı kendi SO-101 verinle nasıl fine-tune ediyor. Biz koşturmadık; bu sayfada bize ait sayı yok.

**Neyi fine-tune ediyorsun.** `lerobot/smolvla_base`, SmolVLM-2 üzerine kurulu, 481 community dataset'iyle pretrain edilmiş 450M parametrelik bir VLA (yaklaşık 100M'si bir flow-matching action expert'te) ([arXiv 2506.01844](https://arxiv.org/abs/2506.01844); mimari ve bizim parametre sayımız [ders 41](../part-10-foundation-models-and-agents/41-vla.md)'de). [`config.json`](https://huggingface.co/lerobot/smolvla_base/blob/main/config.json)'u `freeze_vision_encoder: true` ve `train_expert_only: true`, `chunk_size` 50, 1000 warm-up ve 30.000 decay adımıyla learning rate 1e-4 ayarlıyor: varsayılan olarak fine-tuning vision-language backbone'unu değil, action expert'i güncelliyor.

**Prosedür** ([LeRobot SmolVLA kılavuzu](https://huggingface.co/docs/lerobot/smolvla)'ndan):

1. LeRobot'u kur, sonra `pip install -e ".[smolvla]"`.
2. SO-101 leader/follower ile bir dataset kaydet ([kayıt kılavuzu](https://huggingface.co/docs/lerobot/il_robots)). Kılavuz "~50 episodes" öneriyor; referans dataset'leri 5 küp pozisyonunda, pozisyon başına 10 olmak üzere 50 episode içeriyor ve 25 episode'un "was not enough" olduğunu not ediyor. Her varyasyonu (örn. nesne konumu) birkaç episode ile kapsa.
3. Bir CUDA GPU'da fine-tune et:

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=${HF_USER}/mydataset \
  --batch_size=64 --steps=20000 \
  --output_dir=outputs/train/my_smolvla \
  --job_name=my_smolvla_training \
  --policy.device=cuda --wandb.enable=true
```

   Kılavuz 20k adımın "~4 hrs on a single A100" sürdüğünü belirtiyor ve GPU'su olmayanlar için bir Colab notebook'una bağlantı veriyor. Bellek yetmezse daha küçük bir batch ile başla.
4. Kolda `lerobot-rollout --robot.type=so101_follower ... --task="<kaydettiğin aynı görev metni>" --policy.path=${HF_USER}/<model>` ile çalıştır; görev metni kayıtla aynı olmalı.

**4. adımdan önce mühendislik kontrolleri.** Dil talimatını kayıtlarla birebir aynı tut; kamera adlarının ve çözünürlüğünün dataset'le eşleştiğini doğrula; önce ayrılmış bir pozisyonda, [ders 15](../part-4-dynamics-and-control/15-safety.md)'in safety limitleriyle (`max_relative_target` ayarlı) değerlendir; başarıyı güven aralığıyla k/n olarak raporla ([ders 25](../part-7-learning-from-demonstration/25-evaluating-policies.md)). **Henüz bizim tarafımızdan koşturulmadı ya da ölçülmedi.**

Kaynaklar: [LeRobot SmolVLA docs](https://huggingface.co/docs/lerobot/smolvla) · [SmolVLA, arXiv 2506.01844](https://arxiv.org/abs/2506.01844) · [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base) (Apache-2.0).
