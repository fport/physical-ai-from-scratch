# 12 · Kol zayıf değil, matematik onu sarkıtıyor: gravity compensation

> **Durum: yakında.** Bu sayfa iskelet. Deney henüz koşturulmadı; aşağıdaki deney bir **plandır**, sonuç değil. Sayılar yalnız koşudan sonra yazılır.

| | |
|---|---|
| **Faz** | [2 · Kas](README.md) |
| **Önkoşul** | [11](../phase-2-muscle/11-pd-control.md) |
| **Araç** | MuJoCo + gerçek SO-101 |
| **Durum** | 🚧 yakında |

## Öğrenme hedefi

Yerçekiminin uzanan kolda tork yarattığını ve sadece P control'ün bu yüzden hedefin altında kaldığını anlamak.

## Ders sorusu

Kol yatay uzanınca kaç derece sarkar, ve gravity compensation bunu ne kadar kapatır?

## Neden önemli

_Dünyadan çalışan bir örnekle açılacak (kaynaklı). Yazılacak._

> Not: Ders 11'deki Go2 deneyinde gövde hedef duruşun ~7 cm altında kaldı; bu dersin konusu olan sarkmanın bir örneği.

## Planlanan deney

Kol yatay uzanınca sarkma açısı: sim (kütle sweep'i) ve gerçek kol; gravity compensation açık/kapalı.

Her dersteki gibi bir **sweep**: tek bir parametre değişir, tek bir sayı ölçülür. Sonuç `results/12/result.json` dosyasına yazılır.

## Kaynaklar

- [Lynch & Park, *Modern Robotics*](https://hades.mech.northwestern.edu/index.php/Modern_Robotics) (Ch. 8, dynamics; Ch. 11)

---

_Bu dersin yazılmasına katkı vermek için: [Katkı](../contributing.md)._
