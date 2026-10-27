# 27 · Agent loop'u 25 satırda yazmak

> **Durum: yakında.** Bu sayfa iskelet. Deney henüz koşturulmadı; aşağıdaki deney bir **plandır**, sonuç değil. Sayılar yalnız koşudan sonra yazılır.

| | |
|---|---|
| **Faz** | [5 · Ajan](README.md) |
| **Önkoşul** | [09](../phase-1-space-and-body/09-inverse-kinematics.md) |
| **Araç** | Ollama + MuJoCo |
| **Durum** | 🚧 yakında |

## Öğrenme hedefi

Bir LLM agent'ının "düşün → tool çağır → gözle" loop'unu sıfırdan yazmak.

## Ders sorusu

10 komutta local bir model doğru tool'u kaç kez çağırır?

## Neden önemli

_Dünyadan çalışan bir örnekle açılacak (kaynaklı). Yazılacak._

## Planlanan deney

Ollama tool calling + `move_to(x, y)` tool'u (IK, ders 09); 10 komutta doğru tool call oranı.

Her dersteki gibi bir **sweep**: tek bir parametre değişir, tek bir sayı ölçülür. Sonuç `results/27/result.json` dosyasına yazılır.

## Kaynaklar

- [Ollama · tool calling](https://ollama.com/blog/tool-support)

---

_Bu dersin yazılmasına katkı vermek için: [Katkı](../contributing.md)._
