# 23 · Mükemmel taklitçi ilk hatada batar mı? Distribution shift

> **Durum: yakında.** Bu sayfa iskelet. Deney henüz koşturulmadı; aşağıdaki deney bir **plandır**, sonuç değil. Sayılar yalnız koşudan sonra yazılır.

| | |
|---|---|
| **Faz** | [4 · Göstererek öğretmek](README.md) |
| **Önkoşul** | [21](../phase-4-learning-from-demonstration/21-behavior-cloning.md), [22](../phase-4-learning-from-demonstration/22-evaluation.md) |
| **Araç** | MuJoCo + PyTorch |
| **Durum** | 🚧 yakında |

## Öğrenme hedefi

Taklit eden policy'nin küçük bir hatada bilmediği state'lere düştüğünü ve hatanın biriktiğini (distribution shift) görmek; DAgger fikri.

## Ders sorusu

Rollout sırasında küçük bir itme başarıyı ne kadar düşürür, düzeltme demonstration'ları ne kadar toparlar?

## Neden önemli

_Dünyadan çalışan bir örnekle açılacak (kaynaklı). Yazılacak._

## Planlanan deney

Rollout sırasında küçük itme ekle → başarı düşüşü; düzeltme demonstration'ları ekleyince toparlanma.

Her dersteki gibi bir **sweep**: tek bir parametre değişir, tek bir sayı ölçülür. Sonuç `results/23/result.json` dosyasına yazılır.

## Kaynaklar

- [DAgger, arXiv 1011.0686](https://arxiv.org/abs/1011.0686)
- [LeRobot · imitation learning on real robots](https://huggingface.co/docs/lerobot/il_robots) (`dagger` strategy)
- [Berkeley CS285](https://rail.eecs.berkeley.edu/deeprlcourse/)

---

_Bu dersin yazılmasına katkı vermek için: [Katkı](../contributing.md)._
