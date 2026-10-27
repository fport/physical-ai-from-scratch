# 07 · İki açı, bir nokta: forward kinematics

> **Durum: yakında.** Bu sayfa iskelet. Deney henüz koşturulmadı; aşağıdaki deney bir **plandır**, sonuç değil. Sayılar yalnız koşudan sonra yazılır.

| | |
|---|---|
| **Faz** | [1 · Uzay ve beden](README.md) |
| **Önkoşul** | [05](../phase-1-space-and-body/05-coordinate-frames.md) |
| **Araç** | MuJoCo + NumPy |
| **Durum** | 🚧 yakında |

## Öğrenme hedefi

Eklem açılarından uç konumunu forward kinematics (FK) ile hesaplamak.

## Ders sorusu

10 satırlık bir FK fonksiyonu simülatörün kendi hesabından kaç mm sapar?

## Neden önemli

_Dünyadan çalışan bir örnekle açılacak (kaynaklı). Yazılacak._

## Planlanan deney

2 eklemli FK ile simülasyondaki uç karşılaştırılır; ardından 5 eklemli SO-101'de sim zincirinden FK. Ölçü: fark (mm).

Her dersteki gibi bir **sweep**: tek bir parametre değişir, tek bir sayı ölçülür. Sonuç `results/07/result.json` dosyasına yazılır.

## Kaynaklar

- [Lynch & Park, *Modern Robotics*](https://hades.mech.northwestern.edu/index.php/Modern_Robotics) (Ch. 4)
- [Hugging Face Robotics Course](https://huggingface.co/learn/robotics-course) (Unit 2, classical robotics)

---

_Bu dersin yazılmasına katkı vermek için: [Katkı](../contributing.md)._
