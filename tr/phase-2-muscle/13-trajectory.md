# 13 · Eklemleri düz oynattım, uç eğri çizdi: trajectory

> **Durum: yakında.** Bu sayfa iskelet. Deney henüz koşturulmadı; aşağıdaki deney bir **plandır**, sonuç değil. Sayılar yalnız koşudan sonra yazılır.

| | |
|---|---|
| **Faz** | [2 · Kas](README.md) |
| **Önkoşul** | [09](../phase-1-space-and-body/09-inverse-kinematics.md), [11](../phase-2-muscle/11-pd-control.md) |
| **Araç** | MuJoCo + gerçek SO-101 |
| **Durum** | 🚧 yakında |

## Öğrenme hedefi

Hedefe nasıl gidildiğinin (trajectory'nin) önemli olduğunu öğrenmek.

## Ders sorusu

Joint space'te düz çizgi, uçta da düz çizgi midir?

## Neden önemli

_Dünyadan çalışan bir örnekle açılacak (kaynaklı). Yazılacak._

## Planlanan deney

Joint-space ve Cartesian interpolation karşılaştırması; ani hedef ile yumuşak (min-jerk) profil arasında en büyük ivme farkı.

Her dersteki gibi bir **sweep**: tek bir parametre değişir, tek bir sayı ölçülür. Sonuç `results/13/result.json` dosyasına yazılır.

## Kaynaklar

- [Lynch & Park, *Modern Robotics*](https://hades.mech.northwestern.edu/index.php/Modern_Robotics) (Ch. 9, trajectory generation)

---

_Bu dersin yazılmasına katkı vermek için: [Katkı](../contributing.md)._
