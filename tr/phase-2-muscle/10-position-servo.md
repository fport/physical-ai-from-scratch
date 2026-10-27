# 10 · Motora kuvvet değil, açı söylüyorsun: position servo

> **Durum: yakında.** Bu sayfa iskelet. Deney henüz koşturulmadı; aşağıdaki deney bir **plandır**, sonuç değil. Sayılar yalnız koşudan sonra yazılır.

| | |
|---|---|
| **Faz** | [2 · Kas](README.md) |
| **Önkoşul** | [06](../phase-1-space-and-body/06-degrees-of-freedom.md) |
| **Araç** | LeRobot + gerçek SO-101, MuJoCo |
| **Durum** | 🚧 yakında |

## Öğrenme hedefi

SO-101'in motorlarının açı alan position servo'lar olduğunu ve motorun içinde bir control loop bulunduğunu anlamak.

## Ders sorusu

Bir servo'ya hedef açı gönderdiğinde okunan konum ve load nasıl değişir?

## Neden önemli

SO-101'in eklemleri Feetech STS3215 servo kullanır; LeRobot belgesi 1/345 dişli oranını listeler. Kaynak: [LeRobot · SO-101](https://huggingface.co/docs/lerobot/so101).

## Planlanan deney

Gerçek kol: hedef açı gönder, okunan konumu ve load'u kaydet. Aynısını sim actuator'üyle yap.

Her dersteki gibi bir **sweep**: tek bir parametre değişir, tek bir sayı ölçülür. Sonuç `results/10/result.json` dosyasına yazılır.

## Kaynaklar

- [LeRobot · SO-101](https://huggingface.co/docs/lerobot/so101)
- [Lynch & Park, *Modern Robotics*](https://hades.mech.northwestern.edu/index.php/Modern_Robotics) (Ch. 11, robot control)

---

_Bu dersin yazılmasına katkı vermek için: [Katkı](../contributing.md)._
