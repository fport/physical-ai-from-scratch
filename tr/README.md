# Fiziksel AI Mühendisliği — Sıfırdan

Bir dizüstü bilgisayarla başlayıp bir robot kola kendi başına iş yaptırmaya kadar giden, Türkçe ve açık kaynak bir kurs kitabı. Her ders gerçekten koşturulmuş bir deneye dayanır. Her sayı o deneyden ya da kaynağı gösterilen bir belgeden gelir.

> 40 ders · 10 faz · 8 ek ders. Python, MuJoCo, LeRobot. Önce kendi Mac'inde, sonra gerçek kolda.

**[Kurulum](getting-started/setup.md)** · **[Bir ders nasıl okunur](getting-started/how-to-read-a-lesson.md)** · **[Okuma sırası](getting-started/reading-order.md)** · **[İlk ders: küp masanın içinden neden geçti?](phase-0-world-frame-by-frame/02-timestep.md)** · **[English](../README.md)**

## Bu kitap ne?

Fiziksel AI, yazdığın kodun bir motoru, bir eklemi, bir "kası" oynattığı yapay zekâ. Bu kitap onu en alttan kurar: önce physics engine'in dünyayı nasıl hesapladığını, sonra bir kolun uzayda nerede olduğunu ve motorlarının nasıl döndüğünü, sonra robotun nasıl gördüğünü, demonstration'dan ve denemeden nasıl öğrendiğini, en sonda da bir dil modeliyle nasıl konuşarak iş yaptırıldığını.

Her ders aynı soruyu sorar: **bu satır hangi kası oynatıyor?**

## Kimler için?

- Python okuyabilen, robotiğe sıfırdan başlamak isteyen herkes.
- Lise fiziği (F = m·a, tork) ve biraz lineer cebir işini kolaylaştırır, ama dersler gerektiğinde bunları baştan anlatır.
- İlk dört faz **yalnız bilgisayar** ister. Gerçek kol (SO-101) 6. dersten itibaren "insan katmanı" olarak girer; kolu olmayan okur her dersin simülasyon karşılığını koşturabilir.

## Harita

| Faz | Ad | Soru | Ders |
|---|---|---|---|
| 0 | [Dünya kare kare](phase-0-world-frame-by-frame/README.md) | Bir fizik motoru dünyayı nasıl hesaplar? | 01–04 |
| 1 | [Uzay ve beden](phase-1-space-and-body/README.md) | Kol nerede, neyi nerede görüyor? | 05–09 |
| 2 | [Kas](phase-2-muscle/README.md) | Kod motoru nasıl oynatıyor? | 10–14 |
| 3 | [Duyular](phase-3-senses/README.md) | Robot ne görüyor, ne hissediyor? | 15–18 |
| 4 | [Göstererek öğretmek](phase-4-learning-from-demonstration/README.md) | Göster, öğrensin. | 19–26 |
| 5 | [Ajan](phase-5-agents/README.md) | Konuşarak iş yaptırmak | 27–31 |
| 6 | [Deneyerek öğrenmek](phase-6-learning-by-trial/README.md) | Ödülü sen yaz. | 32–34 |
| 7 | [Sim'den masaya](phase-7-sim-to-real/README.md) | Simde çalıştı, masada neden çalışmıyor? | 35–37 |
| 8 | [Büyük beyinler](phase-8-big-models/README.md) | Robot modelleri | 38–39 |
| 9 | [Bitirme](phase-9-capstone/README.md) | Söyle → gör → planla → tut | 40 |
| EK | [Ek dersler](appendix/README.md) | Derinleşmeler ve Mac'te koşmayan konular | A–H |

## Durum

Kitap yazılıyor. Her dersin başında durumu yazar:

| İşaret | Anlamı | Dersler |
|---|---|---|
| ✅ tam | Metin, deney, sayılar ve alıştırmalar hazır | 01, 02, 04 |
| 📝 taslak · notebook hazır | Deney koşturuldu, notebook ve doğrulanmış sayılar var; metin tamamlanmadı | 11, 19, 24, 31 |
| 🚧 yakında | İskelet: öğrenme hedefi, ders sorusu, planlanan deney, kaynaklar. Sonuç yok | geri kalanlar |

## Nasıl çalışılır?

1. **Oku.** Her ders gerçek dünyadan çalışan bir örnekle açılır (bir robot köpeğin eğitim kodu, bir fabrika kolu, bir makale), sonra kavramı kurar.
2. **Koştur.** Her dersin bir deneyi var: Mac'te `python experiments/.../experiment.py` ya da Colab'de notebook. Deneyler bir **sweep**'tir: tek bir sayı değişir, tek bir sonuç ölçülür.
3. **Değiştir.** Alıştırmalar o sayıyı senin değiştirmeni ister. Önce tahmin et, sonra koştur.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python experiments/02-timestep/experiment.py
```

Kitabı web sitesi olarak yerelde açmak için: `cd site && npm install && npm run dev` ([site/README.md](../site/README.md)).

## Videolar

Bu kitap, Instagram, TikTok ve YouTube Shorts'ta yayınlanan **Koddan Kasa · Sıfırdan Fiziksel AI** serisinin uzun biçimli eşidir. Her video bir dersin özeti; video açıklaması o derse bağlanır. Video 40 saniyede merak uyandırır, kitap öğretir.

## Dürüstlük

- Videodaki ve kitaptaki her sayı ya bizim koşturduğumuz bir deneyden (sürüm, tohum ve sonuç dosyasıyla) ya da bağlantısı verilen bir kaynaktan gelir. Kurallar: [Sayı ve kaynak kuralları](getting-started/numbers-and-sources.md).
- Koşturmadığımız bir şeyi koşturmuş gibi göstermeyiz. GPU ya da donanım isteyen dersler bunu başlıkta söyler.
- Teleoperation ve replay öğrenme değildir; kitap bu farkı bir derse ayırır ([ders 19](phase-4-learning-from-demonstration/19-replay.md)).

## Teşekkür ve ilham

Yapısı, Rohit Ghumare ve katkıcılarının [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) müfredatından (MIT) esinlendi: fazlara bölünmüş numaralı dersler, her derste "problem → kavram → kendin kur → kullan → alıştırma → terimler → kaynaklar" akışı, "framework'ten önce ham hâli" ilkesi. Metin ve kod oradan kopyalanmadı; fiziksel AI'ye özgü yeni bir müfredat olarak yazıldı. Müfredatın dayandığı kaynaklar [Kaynakça](references.md)'da.

## Lisans

Henüz seçilmedi.
