# 09 · Aynı noktaya iki farklı dirsekle: inverse kinematics

> **Durum: yakında.** Bu sayfa iskelet. Deney henüz koşturulmadı; aşağıdaki deney bir **plandır**, sonuç değil. Sayılar yalnız koşudan sonra yazılır.

| | |
|---|---|
| **Faz** | [1 · Uzay ve beden](README.md) |
| **Önkoşul** | [07](../phase-1-space-and-body/07-forward-kinematics.md), [08](../phase-1-space-and-body/08-workspace.md) |
| **Araç** | MuJoCo + NumPy |
| **Durum** | 🚧 yakında |

## Öğrenme hedefi

IK'yı "küçük adımlarla hedefe sürünmek" (Jacobian ile numerical IK) olarak kurmak ve birden çok çözümü görmek.

## Ders sorusu

Başlangıç açısını değiştirince aynı hedefe farklı bir çözümle varılır mı?

## Neden önemli

_Dünyadan çalışan bir örnekle açılacak (kaynaklı). Yazılacak._

## Planlanan deney

Başlangıç konfigürasyonu sweep'i → aynı hedefe farklı çözümler; iteration sayısı ve kalan hata.

Her dersteki gibi bir **sweep**: tek bir parametre değişir, tek bir sayı ölçülür. Sonuç `results/09/result.json` dosyasına yazılır.

## Kaynaklar

- [Lynch & Park, *Modern Robotics*](https://hades.mech.northwestern.edu/index.php/Modern_Robotics) (Ch. 5–6, Jacobian and IK)
- [Hugging Face Robotics Course](https://huggingface.co/learn/robotics-course) (Unit 2)

---

_Bu dersin yazılmasına katkı vermek için: [Katkı](../contributing.md)._
