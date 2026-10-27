# 25 · Robot her frame'de fikir değiştiriyordu: action chunking

> **Durum: yakında.** Bu sayfa iskelet. Deney henüz koşturulmadı; aşağıdaki deney bir **plandır**, sonuç değil. Sayılar yalnız koşudan sonra yazılır.

| | |
|---|---|
| **Faz** | [4 · Göstererek öğretmek](README.md) |
| **Önkoşul** | [23](../phase-4-learning-from-demonstration/23-distribution-shift.md), [24](../phase-4-learning-from-demonstration/24-two-right-answers.md) |
| **Araç** | LeRobot ACT (sim) |
| **Durum** | 🚧 yakında |

## Öğrenme hedefi

Tek tek adım yerine action chunk tahmin etmenin titremeyi azalttığını görmek (ACT fikri).

## Ders sorusu

Chunk size 1'den 100'e çıkınca jerk ve başarı nasıl değişir?

## Neden önemli

LeRobot belgeleri ACT'yi ilk önerilen policy olarak sunar. Kaynak: [LeRobot · ACT](https://huggingface.co/docs/lerobot/act).

## Planlanan deney

Chunk size sweep'i (1/10/50/100) → jerk ve başarı, sim'de LeRobot ACT.

Her dersteki gibi bir **sweep**: tek bir parametre değişir, tek bir sayı ölçülür. Sonuç `results/25/result.json` dosyasına yazılır.

## Kaynaklar

- [ACT, arXiv 2304.13705](https://arxiv.org/abs/2304.13705)
- [LeRobot · ACT](https://huggingface.co/docs/lerobot/act)

---

_Bu dersin yazılmasına katkı vermek için: [Katkı](../contributing.md)._
