# 20 · Bir robot dataset'i neye benzer?

> **Durum: yakında.** Bu sayfa iskelet. Deney henüz koşturulmadı; aşağıdaki deney bir **plandır**, sonuç değil. Sayılar yalnız koşudan sonra yazılır.

| | |
|---|---|
| **Faz** | [4 · Göstererek öğretmek](README.md) |
| **Önkoşul** | [19](../phase-4-learning-from-demonstration/19-replay.md) |
| **Araç** | LeRobot + gerçek SO-101 |
| **Durum** | 🚧 yakında |

## Öğrenme hedefi

Bir robot dataset'inin neye benzediğini (frame + state + action + zaman) okumak.

## Ders sorusu

50 episode'luk bir kayıt kaç frame eder, her frame'de ne saklanır?

## Neden önemli

LeRobot hardware guide bir hesap örneği verir: 50 episode × 30 fps × 30 sn ≈ 45.000 frame. Kaynak: [LeRobot · hardware guide](https://huggingface.co/docs/lerobot/hardware_guide).

## Planlanan deney

HF Hub'da `lerobot/svla_so100_pickplace` incelenir; kendi 5 episode'luk kaydımız LeRobot visualizer'ı ile açılır.

Her dersteki gibi bir **sweep**: tek bir parametre değişir, tek bir sayı ölçülür. Sonuç `results/20/result.json` dosyasına yazılır.

## Kaynaklar

- [LeRobot · hardware guide](https://huggingface.co/docs/lerobot/hardware_guide)
- [LeRobot · imitation learning on real robots](https://huggingface.co/docs/lerobot/il_robots)

---

_Bu dersin yazılmasına katkı vermek için: [Katkı](../contributing.md)._
