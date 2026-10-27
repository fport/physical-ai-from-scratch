# 22 · %90 success rate dedim, aslında ne kadar? Evaluation

> **Durum: yakında.** Bu sayfa iskelet. Deney henüz koşturulmadı; aşağıdaki deney bir **plandır**, sonuç değil. Sayılar yalnız koşudan sonra yazılır.

| | |
|---|---|
| **Faz** | [4 · Göstererek öğretmek](README.md) |
| **Önkoşul** | [21](../phase-4-learning-from-demonstration/21-behavior-cloning.md) |
| **Araç** | MuJoCo + NumPy |
| **Durum** | 🚧 yakında |

## Öğrenme hedefi

Success rate'in bir tahmin olduğunu ve kaç denemenin ne kadar güven verdiğini öğrenmek.

## Ders sorusu

Aynı policy'yi 10, 50 ve 200 kez denersen confidence interval ne kadar daralır?

## Neden önemli

_Dünyadan çalışan bir örnekle açılacak (kaynaklı). Yazılacak._

> Not: Ders 19'un sonuç dosyası 8 denemelik oranlar için Wilson %95 interval'ı veriyor (ör. 1/8 → 0,02–0,47); bu ders kavramı ayrıntılı işler.

## Planlanan deney

Aynı policy'yi 10, 50, 200 kez dene; Wilson interval'ın daralması.

Her dersteki gibi bir **sweep**: tek bir parametre değişir, tek bir sayı ölçülür. Sonuç `results/22/result.json` dosyasına yazılır.

## Kaynaklar

- [LeRobot · imitation learning on real robots](https://huggingface.co/docs/lerobot/il_robots) (evaluation with `lerobot-rollout`)

---

_Bu dersin yazılmasına katkı vermek için: [Katkı](../contributing.md)._
