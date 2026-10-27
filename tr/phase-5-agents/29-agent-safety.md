# 29 · Agent'a "masanın altına uzan" dedim: tool call validation

> **Durum: yakında.** Bu sayfa iskelet. Deney henüz koşturulmadı; aşağıdaki deney bir **plandır**, sonuç değil. Sayılar yalnız koşudan sonra yazılır.

| | |
|---|---|
| **Faz** | [5 · Ajan](README.md) |
| **Önkoşul** | [14](../phase-2-muscle/14-safety.md), [27](../phase-5-agents/27-agent-loop.md) |
| **Araç** | Ollama + MuJoCo |
| **Durum** | 🚧 yakında |

## Öğrenme hedefi

LLM'in hatalı ya da tehlikeli tool call üretebileceğini ve tool katmanında validation gerektiğini öğrenmek.

## Ders sorusu

Erişilemez ya da limit dışı hedefler verildiğinde validation katmanı kaç çağrıyı reddeder?

## Neden önemli

_Dünyadan çalışan bir örnekle açılacak (kaynaklı). Yazılacak._

## Planlanan deney

Erişilemez/limit dışı hedefler listesi: validation kapalı/açık iken reddedilen çağrı sayısı (ders 14 limitleri).

Her dersteki gibi bir **sweep**: tek bir parametre değişir, tek bir sayı ölçülür. Sonuç `results/29/result.json` dosyasına yazılır.

## Kaynaklar

- _Yazılırken eklenecek._

---

_Bu dersin yazılmasına katkı vermek için: [Katkı](../contributing.md)._
