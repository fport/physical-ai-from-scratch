# 18 · Lambayı açtım, robot kırmızıyı göremedi: color thresholding ve open-vocabulary detection

> **Durum: yakında.** Bu sayfa iskelet. Deney henüz koşturulmadı; aşağıdaki deney bir **plandır**, sonuç değil. Sayılar yalnız koşudan sonra yazılır.

| | |
|---|---|
| **Faz** | [3 · Duyular](README.md) |
| **Önkoşul** | [16](../phase-3-senses/16-camera-calibration.md) |
| **Araç** | OpenCV + Hugging Face (Mac CPU/MPS) |
| **Durum** | 🚧 yakında |

## Öğrenme hedefi

Color thresholding'in ışığa kırılgan olduğunu, open-vocabulary detection'ın daha sağlam ama yavaş olduğunu görmek.

## Ders sorusu

Üç ışık koşulunda HSV thresholding ve OWL-ViT nesneyi kaç kez bulur, frame başına kaç ms harcar?

## Neden önemli

_Dünyadan çalışan bir örnekle açılacak (kaynaklı). Yazılacak._

> Not: OWL-ViT'in Mac'teki hızı henüz ölçülmedi.

## Planlanan deney

3 ışık koşulu × (HSV, OWL-ViT) → detection rate ve frame başına ms.

Her dersteki gibi bir **sweep**: tek bir parametre değişir, tek bir sayı ölçülür. Sonuç `results/18/result.json` dosyasına yazılır.

## Kaynaklar

- [LeRobot docs](https://huggingface.co/docs/lerobot/index) (lighting during recording)
- [OWL-ViT, arXiv 2205.06230](https://arxiv.org/abs/2205.06230)

---

_Bu dersin yazılmasına katkı vermek için: [Katkı](../contributing.md)._
