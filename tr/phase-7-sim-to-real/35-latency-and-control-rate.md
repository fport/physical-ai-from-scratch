# 35 · Robotun beyni kaç milisaniye geride? Latency ve control rate

> **Durum: yakında.** Bu sayfa iskelet. Deney henüz koşturulmadı; aşağıdaki deney bir **plandır**, sonuç değil. Sayılar yalnız koşudan sonra yazılır.

| | |
|---|---|
| **Faz** | [7 · Sim'den masaya](README.md) |
| **Önkoşul** | [11](../phase-2-muscle/11-pd-control.md), [15](../phase-3-senses/15-noise-and-latency.md), [26](../phase-4-learning-from-demonstration/26-end-to-end-imitation.md) |
| **Araç** | MuJoCo + gerçek SO-101 |
| **Durum** | 🚧 yakında |

## Öğrenme hedefi

Control rate ve latency'nin stability'yi nasıl bozduğunu görmek.

## Ders sorusu

Control loop'a 0–100 ms latency eklenince overshoot ve başarı nasıl değişir?

## Neden önemli

LeRobot ~30 fps ile kaydeder; yavaş policy'ler için ayrı bir `rtc` inference modu vardır. Kaynak: [LeRobot · imitation learning on real robots](https://huggingface.co/docs/lerobot/il_robots).

## Planlanan deney

Sim'de PID / policy loop'una 0–100 ms latency ekle → overshoot ve başarı; gerçek kolda ölçülen loop Hz.

Her dersteki gibi bir **sweep**: tek bir parametre değişir, tek bir sayı ölçülür. Sonuç `results/35/result.json` dosyasına yazılır.

## Kaynaklar

- [LeRobot · imitation learning on real robots](https://huggingface.co/docs/lerobot/il_robots)

---

_Bu dersin yazılmasına katkı vermek için: [Katkı](../contributing.md)._
