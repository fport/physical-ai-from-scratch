# 39 · Noise'dan yol çıkarmak: diffusion ve flow matching

> **Durum: yakında.** Bu sayfa iskelet. Deney henüz koşturulmadı; aşağıdaki deney bir **plandır**, sonuç değil. Sayılar yalnız koşudan sonra yazılır.

| | |
|---|---|
| **Faz** | [8 · Büyük beyinler](README.md) |
| **Önkoşul** | [24](../phase-4-learning-from-demonstration/24-two-right-answers.md), [38](../phase-8-big-models/38-vla.md) |
| **Araç** | NumPy/PyTorch (Colab CPU) |
| **Durum** | 🚧 yakında |

## Öğrenme hedefi

Noise'dan action üreten modellerin (diffusion / flow matching) multimodality'yi nasıl çözdüğünü sezgiyle görmek.

## Ders sorusu

Ders 24'ün aynı verisinde küçük bir flow/diffusion modeli engele çarpar mı?

## Neden önemli

_Dünyadan çalışan bir örnekle açılacak (kaynaklı). Yazılacak._

## Planlanan deney

Ders 24'ün 2D verisi: küçük bir flow/diffusion modeli iki yolu da ayrı ayrı üretir mi; MLP'ye karşı çarpma oranı.

Her dersteki gibi bir **sweep**: tek bir parametre değişir, tek bir sayı ölçülür. Sonuç `results/39/result.json` dosyasına yazılır.

## Kaynaklar

- [Diffusion Policy, arXiv 2303.04137](https://arxiv.org/abs/2303.04137)
- [π0, arXiv 2410.24164](https://arxiv.org/abs/2410.24164)

---

_Bu dersin yazılmasına katkı vermek için: [Katkı](../contributing.md)._
