# 26 · Gerçek kolda end-to-end imitation learning

> **Durum: yakında.** Bu sayfa iskelet. Deney henüz koşturulmadı; aşağıdaki deney bir **plandır**, sonuç değil. Sayılar yalnız koşudan sonra yazılır.

| | |
|---|---|
| **Faz** | [4 · Göstererek öğretmek](README.md) |
| **Önkoşul** | 19–25 |
| **Araç** | LeRobot + gerçek SO-101 + kiralık GPU |
| **Durum** | 🚧 yakında |

## Öğrenme hedefi

Gerçek kolda recording → training → 20+ deneme ile evaluation; data distribution'ın etkisi.

## Ders sorusu

25 demonstration mı yeter, 50 mi?

## Neden önemli

LeRobot en az 50 episode ve konum başına 10 öneriyor; SmolVLA belgesi 25 episode'un yetmediğini yazıyor. Hardware guide'a göre Apple Silicon (MPS) üzerinde ACT training'i ~6–14 saat sürüyor. Kaynaklar: [LeRobot · imitation learning on real robots](https://huggingface.co/docs/lerobot/il_robots), [LeRobot · SmolVLA](https://huggingface.co/docs/lerobot/smolvla), [LeRobot · hardware guide](https://huggingface.co/docs/lerobot/hardware_guide).

> Not: ACT için 25 ile 50 episode karşılaştırması kendi koşumuz yapılmadan söylenmeyecek.

## Planlanan deney

50 episode (5 konum × 10) ile ACT, karşılaştırma için 25 episode. Training Colab GPU / HF Jobs'ta. Evaluation: `lerobot-rollout`, 20 deneme + confidence interval.

Her dersteki gibi bir **sweep**: tek bir parametre değişir, tek bir sayı ölçülür. Sonuç `results/26/result.json` dosyasına yazılır.

## Kaynaklar

- [LeRobot · imitation learning on real robots](https://huggingface.co/docs/lerobot/il_robots)
- [LeRobot · hardware guide](https://huggingface.co/docs/lerobot/hardware_guide)
- [LeRobot · SmolVLA](https://huggingface.co/docs/lerobot/smolvla)

---

_Bu dersin yazılmasına katkı vermek için: [Katkı](../contributing.md)._
