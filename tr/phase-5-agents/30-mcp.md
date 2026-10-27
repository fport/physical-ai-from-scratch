# 30 · Aynı kolu üç farklı agent kullansın: MCP

> **Durum: yakında.** Bu sayfa iskelet. Deney henüz koşturulmadı; aşağıdaki deney bir **plandır**, sonuç değil. Sayılar yalnız koşudan sonra yazılır.

| | |
|---|---|
| **Faz** | [5 · Ajan](README.md) |
| **Önkoşul** | [27](../phase-5-agents/27-agent-loop.md), [29](../phase-5-agents/29-agent-safety.md) |
| **Araç** | MCP + MuJoCo |
| **Durum** | 🚧 yakında |

## Öğrenme hedefi

Kolu bir MCP server olarak açıp herhangi bir agent'a tool olarak vermek.

## Ders sorusu

MCP üzerinden bir tool call kaç ms gecikme ekler?

## Neden önemli

_Dünyadan çalışan bir örnekle açılacak (kaynaklı). Yazılacak._

## Planlanan deney

MCP server: `get_state`, `move_to`; iki client'tan bağlanma, tool call latency'si (ms).

Her dersteki gibi bir **sweep**: tek bir parametre değişir, tek bir sayı ölçülür. Sonuç `results/30/result.json` dosyasına yazılır.

## Kaynaklar

- [Model Context Protocol](https://modelcontextprotocol.io/)

---

_Bu dersin yazılmasına katkı vermek için: [Katkı](../contributing.md)._
