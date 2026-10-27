# 15 · Noise'u temizledim, robot geç kaldı: filtering ve latency

> **Durum: yakında.** Bu sayfa iskelet. Deney henüz koşturulmadı; aşağıdaki deney bir **plandır**, sonuç değil. Sayılar yalnız koşudan sonra yazılır.

| | |
|---|---|
| **Faz** | [3 · Duyular](README.md) |
| **Önkoşul** | [10](../phase-2-muscle/10-position-servo.md) |
| **Araç** | MuJoCo + gerçek SO-101 |
| **Durum** | 🚧 yakında |

## Öğrenme hedefi

Her sensor'ün noisy olduğunu ve filtering'in latency getirdiğini görmek.

## Ders sorusu

Moving average window'u büyüdükçe noise ne kadar azalır, latency ne kadar artar?

## Neden önemli

_Dünyadan çalışan bir örnekle açılacak (kaynaklı). Yazılacak._

> Not: STS3215 encoder çözünürlüğü henüz doğrulanmadı; ders yazılmadan önce Feetech belgesinden bakılacak.

## Planlanan deney

Eklem okumasına noise ekle → window sweep'i: noise ↓, latency (ms) ↑. Gerçek kol okumasıyla karşılaştırma.

Her dersteki gibi bir **sweep**: tek bir parametre değişir, tek bir sayı ölçülür. Sonuç `results/15/result.json` dosyasına yazılır.

## Kaynaklar

- [Tedrake, *Underactuated Robotics*](https://underactuated.mit.edu/) (state estimation)

---

_Bu dersin yazılmasına katkı vermek için: [Katkı](../contributing.md)._
