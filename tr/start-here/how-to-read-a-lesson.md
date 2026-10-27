# Bir ders nasıl okunur

Her dersin şekli aynıdır; teorinin, kodun ve kanıtın nerede olduğunu hep bilirsin.

## Künye

| Satır | Anlamı |
|---|---|
| **Bölüm** | Dersin ait olduğu bölüm |
| **Çalışma türü** | **CPU** Mac'te ya da Colab CPU'da koşar · **GPU** Colab GPU ister · **ART** bizim verdiğimiz sonuçları (log, checkpoint) kullanır, CPU'da koşar · **HW** gerçek SO-101 ister |
| **Önkoşul** | Bu kitabın gereken önceki dersleri |
| **Önce öğren** | Genel arka planı (matematik, ML, deep learning, RL) anlatan [AI Engineering from Scratch](https://aiengineeringfromscratch.com/index.html) dersleri |
| **Süre** | Okuma ve koşturma, yaklaşık |
| **Kod** | Deney script'i (varsa notebook) |
| **Durum** | ✅ tam · 📝 taslak · 🚧 yakında |

## Bölümler

| Bölüm | Ne verir |
|---|---|
| **Problem** | Dersin çözdüğü somut bir hata ya da soru |
| **Ana fikir** | Gereken en az teori: tanımlar ve birkaç denklem |
| **Deney** | Tahmin et, koştur, gözle. Kısa kod, ardından ölçülmüş sayılardan bir tablo |
| **Boz** | Yöntem bozulana kadar zorlanan bir parametre ve nedeni |
| **Gerçek robotta** | Donanımda neyin değiştiği, yalnız kaynaklı bilgilerle |
| **Kendini dene** | Kolaydan zora üç alıştırma, cevaplarıyla |
| **Akılda kalanlar ve kaynaklar** | Üç madde, sıradaki ders, birincil kaynaklar |

## Her deney bir sweep

Bir parametre değişir, bir büyüklük ölçülür; sonuç tek bir anekdot değil bir tablodur. Her koşunun tam çıktısı script'inin yanında `output.txt` olarak saklanır; dersteki sayılar oradan gelir.
