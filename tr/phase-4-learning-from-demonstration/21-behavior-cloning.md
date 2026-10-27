# 21 · Kimse kod yazmadı: behavior cloning

> **Durum: yakında.** Bu sayfa iskelet. Deney henüz koşturulmadı; aşağıdaki deney bir **plandır**, sonuç değil. Sayılar yalnız koşudan sonra yazılır.

| | |
|---|---|
| **Faz** | [4 · Göstererek öğretmek](README.md) |
| **Önkoşul** | [07](../phase-1-space-and-body/07-forward-kinematics.md), [20](../phase-4-learning-from-demonstration/20-datasets.md) |
| **Araç** | MuJoCo + PyTorch (Mac) |
| **Durum** | 🚧 yakında |

## Öğrenme hedefi

Behavior cloning'i sıfırdan, 30 satırlık bir MLP ile yazmak.

## Ders sorusu

Demonstration sayısı 5'ten 50'ye çıkınca başarı nasıl değişir?

## Neden önemli

_Dünyadan çalışan bir örnekle açılacak (kaynaklı). Yazılacak._

## Planlanan deney

Sim'de state-based "küpe uzan" görevi; demonstration sayısı sweep'i (5/10/20/50) → başarı.

Her dersteki gibi bir **sweep**: tek bir parametre değişir, tek bir sayı ölçülür. Sonuç `results/21/result.json` dosyasına yazılır.

## Kaynaklar

- [Berkeley CS285](https://rail.eecs.berkeley.edu/deeprlcourse/) (behavior cloning)
- [Robot Learning: A Tutorial, arXiv 2510.12403](https://arxiv.org/abs/2510.12403)

---

_Bu dersin yazılmasına katkı vermek için: [Katkı](../contributing.md)._
