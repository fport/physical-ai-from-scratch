# 06 · 6 motor var, 6 degrees of freedom yok

> **Durum: yakında.** Bu sayfa iskelet. Deney henüz koşturulmadı; aşağıdaki deney bir **plandır**, sonuç değil. Sayılar yalnız koşudan sonra yazılır.

| | |
|---|---|
| **Faz** | [1 · Uzay ve beden](README.md) |
| **Önkoşul** | [05](../phase-1-space-and-body/05-coordinate-frames.md) |
| **Araç** | MuJoCo + gerçek SO-101 |
| **Durum** | 🚧 yakında |

## Öğrenme hedefi

Degrees of freedom ile motor sayısının farkını ve 5 eksenli bir kolun her pose'u alamayacağını anlamak.

## Ders sorusu

SO-101 gripper'ı yatay tutarak bir küpün etrafında dönebilir mi?

## Neden önemli

SO-101'de 6 motor var: 5 eklem (shoulder pan, shoulder lift, elbow flex, wrist flex, wrist roll) ve 1 gripper. Kaynak: [LeRobot · SO-101](https://huggingface.co/docs/lerobot/so101).

## Planlanan deney

Gripper'ı yatay tutarak küpün etrafında dönmeyi dene; istenen ile ulaşılan orientation arasındaki açı hatasını ölç. Gerçek kolla insan katmanı.

Her dersteki gibi bir **sweep**: tek bir parametre değişir, tek bir sayı ölçülür. Sonuç `results/06/result.json` dosyasına yazılır.

## Kaynaklar

- [LeRobot · SO-101](https://huggingface.co/docs/lerobot/so101)
- [Lynch & Park, *Modern Robotics*](https://hades.mech.northwestern.edu/index.php/Modern_Robotics) (Ch. 2, configuration space)

---

_Bu dersin yazılmasına katkı vermek için: [Katkı](../contributing.md)._
