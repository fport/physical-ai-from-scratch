# Bir ders nasıl okunur

Her ders aynı iskeleti izler. Bir derse nereden girersen gir, nerede olduğunu bilirsin.

## Künye

Sayfanın başındaki tablo: faz, tür, önkoşul, süre, araç, kod ve **durum** (✅ tam · 📝 taslak · 🚧 yakında).

## Bölümler

| Bölüm | Ne yapar |
|---|---|
| **Neden önemli** | Dünyada çalışan bir şeyle açılır: bir robot köpeğin açık eğitim kodundaki tek bir sayı, bir fabrika kolunun özelliği, bir makalenin çıkış noktası. Her iddia kaynaklı. |
| **Kavram** | Kodsuz sezgi: benzetme, bir tablo, iki satırlık formül. |
| **Kendin kur** | Adım adım deney. Her kod bloğu kendi başına koşar. Tam script `experiments/` altında, notebook `notebooks/` altında. |
| **Sayılarla sonuç** | Deneyin gerçek çıktısı: sürüm, donanım, tablo. Ders sorusunun cevabı burada. |
| **Sınırlar** | Bu sonucun neyi göstermediği. Simülasyon nerede gerçekten ayrılıyor, neyi ölçmedik. |
| **Alıştırmalar** | Kolay → orta → zor. Çoğu, deneydeki tek bir sayıyı değiştirmeni ister. Önce tahmin et, sonra koştur. |
| **Terimler** | Terim · sık duyulan · aslında ne. |
| **Kaynaklar** | Birincil kaynaklar ve bu sayfadaki sayıların nereden geldiği. |

## Her deney bir sweep

Bir parametre değişir, bir sayı ölçülür. Ders 02'de timestep 2'den 50 ms'ye, ders 04'te rampa açısı 0,05° adımlarla, ders 24'te demonstration sayısı 10'dan 500'e taranır. Dersin "ters köşesi" bu sweep'in verisinden çıkar: 12 ms'de küp kalır ama 10'da geçer; 100 kat ağır küp aynı açıda kayar; daha çok demonstration çarpmayı artırır.

## Sonuç dosyası

Notebook'lar sonuçlarını `results/NN/result.json` dosyasına yazar. Kitaptaki ve videodaki sayılar bu dosyadan okunur.
