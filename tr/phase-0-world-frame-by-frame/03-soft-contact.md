# 03 · Simülasyonda masa aslında bir sünger: soft contact

> **Durum: yakında.** Bu sayfa iskelet. Deney henüz koşturulmadı; aşağıdaki deney bir **plandır**, sonuç değil. Sayılar yalnız koşudan sonra yazılır.

| | |
|---|---|
| **Faz** | [0 · Dünya kare kare](README.md) |
| **Önkoşul** | [02](../phase-0-world-frame-by-frame/02-timestep.md) |
| **Araç** | MuJoCo (Mac / Colab CPU) |
| **Durum** | 🚧 yakında |

## Öğrenme hedefi

Simülasyondaki contact'ın sert değil yumuşak bir yay olduğunu ve nesnelerin birbirine biraz girdiğini anlamak.

## Ders sorusu

Masada duran bir küp masaya kaç milimetre gömülür, ve bunu hangi sayı belirler?

## Neden önemli

_Dünyadan çalışan bir örnekle açılacak (kaynaklı). Yazılacak._

> Not: Ders 01'in koşusunda default soft contact küpü birkaç mm masaya gömüyor; bu ders o gözlemi ölçüye çeviriyor.

## Planlanan deney

`solref` (contact stiffness/damping) sweep'i: küpün masaya penetration derinliği (mm) ve stiffness arttıkça sıçrama.

Her dersteki gibi bir **sweep**: tek bir parametre değişir, tek bir sayı ölçülür. Sonuç `results/03/result.json` dosyasına yazılır.

## Kaynaklar

- [MuJoCo · Computation](https://mujoco.readthedocs.io/en/stable/computation/index.html)
- [MuJoCo XML reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html) (`solref`, `solimp`)

---

_Bu dersin yazılmasına katkı vermek için: [Katkı](../contributing.md)._
