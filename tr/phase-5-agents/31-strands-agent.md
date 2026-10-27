# 31 · Robota konuşarak iş yaptırmak: Strands

> Agent'ın cevabına değil, ölçüme bak.

> **Durum: taslak.** Deney koşturuldu ve notebook hazır. Bu ders yeni sırada 31. (Koddan Kasa videosunda künyesi "Levha 32" idi; notebook içinde hâlâ öyle geçiyor.) Önkoşul dersleri 27–30 henüz yazılmadı.

| | |
|---|---|
| **Faz** | [5 · Ajan](README.md) |
| **Tür** | Deney |
| **Önkoşul** | [27–30](27-agent-loop.md) (deney için şart değil) |
| **Süre** | ~60 dk (model indirme dahil) |
| **Araç** | `strands-robots` 0.5.3 · `strands-agents` · MuJoCo **3.16.0** · Ollama `qwen3.5:9b` ya da Claude API |
| **Kod** | [notebook](../../notebooks/31-strands-agent.ipynb) |
| **Durum** | 📝 taslak · notebook hazır |

## Neden önemli

Dil modelleri artık tool calling yapabiliyor. `strands-robots` (AWS Strands Labs, Apache-2.0) simüle bir SO-100 kolu, bir LLM agent'ına **tek bir tool** olarak veriyor. Bu tool'un içinde 77 action var ("Actions (77 total)"). Birkaç satır kodla bir robota "kırmızı küpü al" diyebiliyorsun. Soru şu: agent "aldım" dediğinde küp gerçekten kalkmış mı?

## Kavram

Agent loop: **düşün → tool çağır → gözle → düzelt**. Gözlem (observation) adımı atlanır ya da yok sayılırsa agent yanıldığını fark etmez. Bu derste sonucu agent'ın cümlesinden değil, simülasyonun ölçümünden (küpün yüksekliği) okuyacağız.

## Deney (notebook'ta)

- Sahne README örneğiyle aynı: `Robot("so100")`, 5 cm kırmızı küp, ön kamera.
- Agent modeli: Ollama ile yerel `qwen3.5:9b` (Mac, API anahtarı yok). Prompt: "pick up the red cube" ve Türkçesi.
- Her tool call, eklem açıları ve küp konumu kaydedilir.

## Sayılarla sonuç

`qwen3.5:9b` ile 6 deneme:

| Sonuç | Kaç deneme |
|---|---|
| Küp kalktı (52,15 cm ve 35,08 cm) | 2 |
| Küp masadayken agent "başardım" dedi | 1 |
| Çöktü (tool call hatası) | 1 |
| Vazgeçti / sahte policy'yi çağırıp durdu | 2 |

`qwen2.5:7b` ile 1 denemede hiç tool çağrılmadı.

Başarılı koşunun anatomisi (`en-qwen35-2`): **16 tool call** (3'ü hata). Kapatınca tool "Closed on nothing" dedi; agent 171. saniyede küpün yüksekliğini okudu (2,49 cm), hatayı gördü, yeniden indi, `attach_bodies` ile küpü gripper'a bağladı ve 50 cm'ye kaldırdı. Simülasyon saati **5,91 sn** ilerledi, duvar saati (wall-clock) **351 sn** sürdü: robot ~6 saniye hareket etti, model ~6 dakika düşündü.

## Sınırlar

- **Simde grasp gerçek değil.** Gripper küpe 29,80 N bastırıyor ama sürtünme küpün 0,98 N'luk ağırlığının en fazla 0,00 N'unu taşıyor. Agent küpü `attach_bodies(mode="weld")` ile gripper'a "yapıştırıyor". Bu bir **grasp assist**, fiziksel grasp değil; gerçek kolda karşılığı yok. Weld'i kütüphane değil, agent çağırdı.
- `strands-robots` 0.5.3, MuJoCo 3.17 ile `attach_bodies(weld)` çağrısında `TypeError` veriyor (`mjNEQDATA` 11 → 12). Notebook MuJoCo 3.16.0'a sabitli.
- Koşular yerel 9B bir modelle yapıldı, güçlü bir bulut modeliyle değil. 6 denemelik oran bir tahmin; güven aralığı geniş.

## Alıştırmalar

1. **Kolay.** Aynı prompt'u 5 kez koştur. Kaçında küp kalkıyor, kaçında agent yanlışlıkla "başardım" diyor?
2. **Orta.** System prompt'a "kaldırdıktan sonra `get_body_state` ile küpün yüksekliğini kontrol et" ekle. Success rate ve yanlış iddia oranı değişiyor mu?
3. **Zor.** `attach_bodies`'i agent'tan gizle (tool listesinden çıkar). Agent ne yapıyor? Bu, gerçek kol için ne anlama geliyor?

## Kaynaklar

- [`strands-robots` (PyPI)](https://pypi.org/project/strands-robots/) (Apache-2.0) ve README örneği.
- [Strands Agents](https://strandsagents.com/).
- [Hugging Face Robotics Course](https://huggingface.co/learn/robotics-course).
- Sayılar Koddan Kasa 32 koşu kayıtlarından ve iç doğrulama notundan (25.10.2026).
