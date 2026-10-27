# 05 · Robot için "sol" diye bir şey yok: coordinate frames

> **Durum: yakında.** Bu sayfa iskelet. Deney henüz koşturulmadı; aşağıdaki deney bir **plandır**, sonuç değil. Sayılar yalnız koşudan sonra yazılır.

| | |
|---|---|
| **Faz** | [1 · Uzay ve beden](README.md) |
| **Önkoşul** | [01](../phase-0-world-frame-by-frame/01-world-as-text.md) |
| **Araç** | MuJoCo + Menagerie |
| **Durum** | 🚧 yakında |

## Öğrenme hedefi

Her noktanın bir frame'e göre tanımlandığını; transform matrix'in "bakış açısı değiştirmek" olduğunu öğrenmek.

## Ders sorusu

Kol döndüğünde küpün koordinatlarından hangisi değişir: world, base ya da gripper frame'indeki?

## Neden önemli

_Dünyadan çalışan bir örnekle açılacak (kaynaklı). Yazılacak._

## Planlanan deney

Aynı küpün world, base ve gripper frame'indeki koordinatları; kol dönünce yalnız biri değişir. Menagerie `robotstudio_so101` modeli.

Her dersteki gibi bir **sweep**: tek bir parametre değişir, tek bir sayı ölçülür. Sonuç `results/05/result.json` dosyasına yazılır.

## Kaynaklar

- [Lynch & Park, *Modern Robotics*](https://hades.mech.northwestern.edu/index.php/Modern_Robotics) (Ch. 3, rigid-body motions)
- [Tedrake, *Robotic Manipulation*](https://manipulation.mit.edu/) (App. A, spatial algebra)
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)

---

_Bu dersin yazılmasına katkı vermek için: [Katkı](../contributing.md)._
