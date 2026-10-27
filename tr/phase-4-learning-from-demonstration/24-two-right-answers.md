# 24 · İki doğru cevabın ortalaması yanlıştır

> Robot engelin solundan da sağından da geçmeyi öğrendi. Tam ortasına çarptı.

> **Durum: taslak.** Deney koşturuldu, notebook hazır ve sayılar doğrulandı. Metin henüz kitap formatında tamamlanmadı.

| | |
|---|---|
| **Faz** | [4 · Göstererek öğretmek](README.md) |
| **Tür** | Deney |
| **Önkoşul** | [21 · Behavior cloning](21-behavior-cloning.md) |
| **Süre** | ~40 dk |
| **Araç** | NumPy · Colab CPU ya da Mac (GPU gerekmez, ~1 dk) |
| **Kod** | [notebook](../../notebooks/24-two-right-answers.ipynb) |
| **Durum** | 📝 taslak · notebook hazır |

## Neden önemli

2023'te Columbia, Toyota Research Institute ve MIT'den araştırmacılar *Diffusion Policy*'yi yayımladı ([Chi vd., arXiv 2303.04137](https://arxiv.org/abs/2303.04137)). Makalenin çıkış noktalarından biri **multimodal action dağılımları**: aynı durumda birden fazla doğru hareket olabilir ve tek bir ortalama öğrenen yöntemler bu durumda bozulur. Bu ders o fikri, bir engelin iki yanından geçen 200 demonstration ile, 2 boyutta ve bir dakikada gösteriyor.

## Kavram

Karesel hatayı (MSE) en aza indiren tahmin, verinin **koşullu ortalamasıdır** (conditional mean; ör. Bishop, *Pattern Recognition and Machine Learning* §1.5.5). Aynı başlangıçtan yarı yarıya soldan ve sağdan geçen demonstration'ların ortalaması, engelin tam ortasından geçen bir yoldur.

## Deney (notebook'ta)

Kuşbakışı 2D düzlemde bir nokta aşağıdan yukarıya gider, arada yuvarlak bir engel var. 200 demonstration'ın 100'ü soldan, 100'ü sağdan geçer; taraf, başlangıç noktasından bağımsız seçilir.

- **MSE modeli:** başlangıca bakıp yolun tamamını (40 waypoint) tahmin eden küçük bir MLP (64-64, tanh).
- **Mixture model (MDN, 2 bileşen):** aynı ağ, çıkışı iki olası yol ve ağırlıkları. Denemede önce bir bileşen seçilir, sonra o yol izlenir.

## Sayılarla sonuç

NumPy 2.5.4, seed 0, 1500 epoch, 200 deneme.

| | Çarpma |
|---|---|
| MSE, demonstration'lar yarı sol / yarı sağ | **200/200** |
| MSE, demonstration'ların hepsi soldan | 0/200 |
| Mixture model (MDN) | **0/200** (96 sol, 104 sağ) |

Engel hizasında (y = 0) demonstration'lar x = −0,55 ve +0,55'ten geçiyor, MSE yolu x = +0,03'ten (engel yarıçapı 0,30).

**Daha çok demonstration düzeltmiyor**, tersine: MSE çarpma oranı 10 demonstration'da %8, 20'de %46, 50'de %85, 100'de %91,5, 200 ve 500'de %100. Az demonstration'da ağ, başlangıçtaki küçük farkları demonstration'ın tarafına bağlayıp ezberliyor (overfitting) ve çoğu zaman bir yana sapıyor; veri arttıkça gerçek ortalamaya yaklaşıyor.

Üç farklı seed'de MSE %97 / %93,5 / %100 çarptı, mixture model üçünde de 0.

## Sınırlar

- "Robot" 2D bir nokta.
- Sonuç, gözlemin (observation) iki yolu ayırt edemediği düzende geçerli (burada: başlangıçtan bütün yolu tahmin etmek). Adım adım, closed-loop bir MSE policy'si ((x, y) → sonraki adım) aynı veride **0/200** çarptı: ortadaki ortalama akış kararsız, küçük bir fark robotu bir yana itiyor.
- Mixture model'de σ sabit (0,2) ve bileşenler iki uzak demonstration ile başlatılıyor; öğrenilen σ ile ilk denemede bileşenler çöktü.
- Oyuncak model Diffusion Policy ile aynı değil. Ortak fikir: tek ortalama yerine birden çok olası hareketi temsil etmek. ACT ([arXiv 2304.13705](https://arxiv.org/abs/2304.13705)) CVAE ile eğitilir ama inference'ta latent değişkeni sıfırlar; "olasılıkları örnekler" demek yanlış olur.

## Alıştırmalar

1. **Kolay.** Demonstration'ların %80'i soldan olsa MSE modeli çarpar mı? Notebook'ta `LEFT_FRACS` ile dene. (Cevap notebook'ta.)
2. **Orta.** Engeli küçült (`OBS_R = 0.1`). Ortalama yol artık engelin yanından geçiyor mu?
3. **Zor.** Adım adım bir policy yaz ve neden çoğu zaman kurtulduğunu bir akış alanı (flow field) çizerek göster. Bu kurtuluş güvenilir mi?

## Kaynaklar

- [Chi vd., *Diffusion Policy*, arXiv 2303.04137](https://arxiv.org/abs/2303.04137).
- [Zhao vd., *ACT*, arXiv 2304.13705](https://arxiv.org/abs/2304.13705).
- Bishop, *Pattern Recognition and Machine Learning* (2006), §1.5.5.
- Sayılar Koddan Kasa 24 deneyinin sonuç dosyasından ve iç doğrulama notundan (27.10.2026).
