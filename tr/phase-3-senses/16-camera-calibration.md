# 16 · Kameran düz çizgiyi eğri görüyor: camera calibration

> **Durum: yakında.** Bu sayfa iskelet. Deney henüz koşturulmadı; aşağıdaki deney bir **plandır**, sonuç değil. Sayılar yalnız koşudan sonra yazılır.

| | |
|---|---|
| **Faz** | [3 · Duyular](README.md) |
| **Önkoşul** | [05](../phase-1-space-and-body/05-coordinate-frames.md) |
| **Araç** | OpenCV (Mac / Colab) |
| **Durum** | 🚧 yakında |

## Öğrenme hedefi

Kameranın 3D'yi 2D'ye izdüşüren bir matrix olduğunu ve calibration'ın bu matrix'i bulduğunu öğrenmek.

## Ders sorusu

Distortion düzeltilince reprojection error kaç pixel'den kaç pixel'e iner?

## Neden önemli

_Dünyadan çalışan bir örnekle açılacak (kaynaklı). Yazılacak._

## Planlanan deney

Checkerboard ile OpenCV calibration; distortion düzeltilmeden/düzeltildikten sonra reprojection error (pixel). Telefon kamerası yeter.

Her dersteki gibi bir **sweep**: tek bir parametre değişir, tek bir sayı ölçülür. Sonuç `results/16/result.json` dosyasına yazılır.

## Kaynaklar

- [Tedrake, *Robotic Manipulation*](https://manipulation.mit.edu/) (Ch. 4, geometric pose estimation)

---

_Bu dersin yazılmasına katkı vermek için: [Katkı](../contributing.md)._
