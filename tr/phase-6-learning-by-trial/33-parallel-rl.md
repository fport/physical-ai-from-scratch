# 33 · Binlerce robot aynı anda düşerse: PPO sezgisi

> **Durum: yakında.** Bu sayfa iskelet. Deney henüz koşturulmadı; aşağıdaki deney bir **plandır**, sonuç değil. Sayılar yalnız koşudan sonra yazılır.

| | |
|---|---|
| **Faz** | [6 · Deneyerek öğrenmek](README.md) |
| **Önkoşul** | [32](../phase-6-learning-by-trial/32-q-learning.md) |
| **Araç** | MuJoCo Playground (Colab GPU) |
| **Durum** | 🚧 yakında |

## Öğrenme hedefi

Policy'nin denemeyle iyileştiğini (policy gradient → PPO sezgisi) ve GPU'da binlerce paralel robotla training'in dakikalara indiğini görmek.

## Ders sorusu

Paralel environment sayısı artınca wall-clock süresi ve reward nasıl değişir?

## Neden önemli

Unitree'nin açık training kodu `unitree_rl_gym` default olarak 4.096 paralel environment ile eğitir (`num_envs = 4096`). Kaynak: [legged_robot_config.py](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/base/legged_robot_config.py).

> Not: Playground locomotion notebook'unun ücretsiz Colab GPU'da çalıştığı henüz doğrulanmadı.

## Planlanan deney

Playground locomotion notebook'u; paralel environment sayısı sweep'i → wall-clock süresi ve reward.

Her dersteki gibi bir **sweep**: tek bir parametre değişir, tek bir sayı ölçülür. Sonuç `results/33/result.json` dosyasına yazılır.

## Kaynaklar

- [MuJoCo Playground](https://playground.mujoco.org/) · [arXiv 2502.08844](https://arxiv.org/abs/2502.08844)
- [PPO, arXiv 1707.06347](https://arxiv.org/abs/1707.06347)

---

_Bu dersin yazılmasına katkı vermek için: [Katkı](../contributing.md)._
