# 37 · Sim'i bilerek bozmak: domain randomization

> **Durum: yakında.** Bu sayfa iskelet. Deney henüz koşturulmadı; aşağıdaki deney bir **plandır**, sonuç değil. Sayılar yalnız koşudan sonra yazılır.

| | |
|---|---|
| **Faz** | [7 · Sim'den masaya](README.md) |
| **Önkoşul** | [33](../phase-6-learning-by-trial/33-parallel-rl.md), [36](../phase-7-sim-to-real/36-system-identification.md) |
| **Araç** | MuJoCo veya Colab GPU |
| **Durum** | 🚧 yakında |

## Öğrenme hedefi

Simülasyonu bilerek bozarak (domain randomization) robust bir policy eğitmek ve fazla bozmanın zararını görmek.

## Ders sorusu

Randomization aralığı genişledikçe ayrı bir test sim'inde başarı nasıl değişir?

## Neden önemli

Unitree'nin training kodunda her environment'ın friction'ı 0,5–1,25 aralığından seçilir (`randomize_friction = True`, `friction_range = [0.5, 1.25]`). Kaynak: [legged_robot_config.py](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/base/legged_robot_config.py). Bkz. [ders 04](../phase-0-world-frame-by-frame/04-friction.md).

## Planlanan deney

Friction/kütle/latency randomization genişliği sweep'i → "gerçek" yerine geçen ayrı test sim'inde başarı.

Her dersteki gibi bir **sweep**: tek bir parametre değişir, tek bir sayı ölçülür. Sonuç `results/37/result.json` dosyasına yazılır.

## Kaynaklar

- [Tobin et al. 2017, arXiv 1703.06907](https://arxiv.org/abs/1703.06907)
- [Tan et al. 2018, arXiv 1804.10332](https://arxiv.org/abs/1804.10332)
- [NVIDIA · Train an SO-101 Robot From Sim-to-Real](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/index.html)

---

_Bu dersin yazılmasına katkı vermek için: [Katkı](../contributing.md)._
