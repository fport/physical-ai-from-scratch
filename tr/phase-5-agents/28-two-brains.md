# 28 · Büyük model kolu oynatmıyor, emir veriyor

> **Durum: yakında.** Bu sayfa iskelet. Deney henüz koşturulmadı; aşağıdaki deney bir **plandır**, sonuç değil. Sayılar yalnız koşudan sonra yazılır.

| | |
|---|---|
| **Faz** | [5 · Ajan](README.md) |
| **Önkoşul** | [18](../phase-3-senses/18-color-vs-open-vocabulary.md), [26](../phase-4-learning-from-demonstration/26-end-to-end-imitation.md), [27](../phase-5-agents/27-agent-loop.md) |
| **Araç** | Ollama + MuJoCo |
| **Durum** | 🚧 yakında |

## Öğrenme hedefi

Agent'ın tool'larının önceki fazlarda kurulan beceriler olduğunu görmek: büyük model ne yapılacağını, küçük beceri nasıl yapılacağını bilir.

## Ders sorusu

3 adımlı bir görevde hata en çok hangi adımda çıkar?

## Neden önemli

_Dünyadan çalışan bir örnekle açılacak (kaynaklı). Yazılacak._

## Planlanan deney

`find` (ders 18) + `move_to` + `pick` (ders 26 policy'si): 3 adımlı görevde adım başına hata.

Her dersteki gibi bir **sweep**: tek bir parametre değişir, tek bir sayı ölçülür. Sonuç `results/28/result.json` dosyasına yazılır.

## Kaynaklar

- _Yazılırken eklenecek._

---

_Bu dersin yazılmasına katkı vermek için: [Katkı](../contributing.md)._
