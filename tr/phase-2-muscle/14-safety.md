# 14 · Robotun ilk kuralı: kendini durdurabilmek

> **Durum: yakında.** Bu sayfa iskelet. Deney henüz koşturulmadı; aşağıdaki deney bir **plandır**, sonuç değil. Sayılar yalnız koşudan sonra yazılır.

| | |
|---|---|
| **Faz** | [2 · Kas](README.md) |
| **Önkoşul** | [10](../phase-2-muscle/10-position-servo.md), [13](../phase-2-muscle/13-trajectory.md) |
| **Araç** | Gerçek SO-101 |
| **Durum** | 🚧 yakında |

## Öğrenme hedefi

Gerçek kola dokunmadan önce joint limit'lerin, torque/velocity limit'in ve E-STOP'un kurulması gerektiğini öğrenmek.

## Ders sorusu

E-STOP komutu ile kolun gerçekten durması arasında kaç ms geçer?

## Neden önemli

_Dünyadan çalışan bir örnekle açılacak (kaynaklı). Yazılacak._

> Not: Müfredat araştırması LeRobot belgelerinde ayrı bir safety sayfası bulamadı (doğrulanmadı); bu ders mühendislik pratiğine dayanır, kaynakları yazılırken eklenecek.

## Planlanan deney

Software limit + köpüğe çarpma testi (torque limit açık/kapalı), E-STOP latency'si (ms).

Her dersteki gibi bir **sweep**: tek bir parametre değişir, tek bir sayı ölçülür. Sonuç `results/14/result.json` dosyasına yazılır.

## Kaynaklar

- [LeRobot · SO-101](https://huggingface.co/docs/lerobot/so101)

---

_Bu dersin yazılmasına katkı vermek için: [Katkı](../contributing.md)._
