# 36 · Simdeki motor gerçeğinden farklıydı: system identification

> **Durum: yakında.** Bu sayfa iskelet. Deney henüz koşturulmadı; aşağıdaki deney bir **plandır**, sonuç değil. Sayılar yalnız koşudan sonra yazılır.

| | |
|---|---|
| **Faz** | [7 · Sim'den masaya](README.md) |
| **Önkoşul** | [10](../phase-2-muscle/10-position-servo.md), [12](../phase-2-muscle/12-gravity.md), [35](../phase-7-sim-to-real/35-latency-and-control-rate.md) |
| **Araç** | Gerçek SO-101 + MuJoCo |
| **Durum** | 🚧 yakında |

## Öğrenme hedefi

Sim motorunu gerçeğine benzetmek için ölçüm yapmak (system identification).

## Ders sorusu

Aynı step komutunda sim kol ile gerçek kol arasındaki tracking error, ayar öncesi ve sonrası ne kadar?

## Neden önemli

NVIDIA'nın SO-101 sim-to-real kursu dördüncü stratejisini actuator gap'i ölçmeye ayırır. Kaynak: [NVIDIA · Train an SO-101 Robot From Sim-to-Real](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/index.html).

## Planlanan deney

Aynı step komutu: gerçek kol ve sim kol; kp/damping grid search → tracking error önce/sonra.

Her dersteki gibi bir **sweep**: tek bir parametre değişir, tek bir sayı ölçülür. Sonuç `results/36/result.json` dosyasına yazılır.

## Kaynaklar

- [NVIDIA · Train an SO-101 Robot From Sim-to-Real](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/index.html)
- [Tedrake, *Underactuated Robotics*](https://underactuated.mit.edu/) (system identification)

---

_Bu dersin yazılmasına katkı vermek için: [Katkı](../contributing.md)._
