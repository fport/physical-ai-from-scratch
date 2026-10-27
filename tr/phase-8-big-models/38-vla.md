# 38 · Bir VLA'nın içine bakmak: image + text → action

> **Durum: yakında.** Bu sayfa iskelet. Deney henüz koşturulmadı; aşağıdaki deney bir **plandır**, sonuç değil. Sayılar yalnız koşudan sonra yazılır.

| | |
|---|---|
| **Faz** | [8 · Büyük beyinler](README.md) |
| **Önkoşul** | [25](../phase-4-learning-from-demonstration/25-action-chunking.md), [26](../phase-4-learning-from-demonstration/26-end-to-end-imitation.md) |
| **Araç** | LeRobot (Mac/MPS ya da Colab GPU) |
| **Durum** | 🚧 yakında |

## Öğrenme hedefi

Bir VLA'nın input ve output'unu (image + cümle + açılar → action chunk) tensor shape'leriyle okumak.

## Ders sorusu

SmolVLA bir adımda ne alır, ne verir, kaç ms sürer?

## Neden önemli

_Dünyadan çalışan bir örnekle açılacak (kaynaklı). Yazılacak._

> Not: SmolVLA'nın Mac MPS'de çalışıp çalışmadığı henüz doğrulanmadı.

## Planlanan deney

SmolVLA (450M) inference: input/output shape'leri, chunk uzunluğu, adım başına ms.

Her dersteki gibi bir **sweep**: tek bir parametre değişir, tek bir sayı ölçülür. Sonuç `results/38/result.json` dosyasına yazılır.

## Kaynaklar

- [SmolVLA, arXiv 2506.01844](https://arxiv.org/abs/2506.01844)
- [LeRobot · SmolVLA](https://huggingface.co/docs/lerobot/smolvla)

---

_Bu dersin yazılmasına katkı vermek için: [Katkı](../contributing.md)._
