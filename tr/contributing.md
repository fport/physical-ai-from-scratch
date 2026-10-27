# Katkı

Bu kitap yazılıyor ve katkıya açık: GitHub'da bir issue ya da pull request aç.

## Bir dersi tamamlamak

Bir iskelet dersi (🚧) tama çevirmek için:

1. **Deneyi koştur.** Dersin "Planlanan deney" bölümündeki sweep'i yaz; en az 5 değer dene. Sürümleri yazdır, varsa seed'i sabitle, sonucu `results/NN/result.json` dosyasına yaz.
2. **Sayfayı şablona göre yaz:** [Bir ders nasıl okunur](getting-started/how-to-read-a-lesson.md). Hazır örnekler: [01](phase-0-world-frame-by-frame/01-world-as-text.md), [02](phase-0-world-frame-by-frame/02-timestep.md), [04](phase-0-world-frame-by-frame/04-friction.md).
3. **Her iddiaya kaynak koy.** [Sayı ve kaynak kuralları](getting-started/numbers-and-sources.md).
4. **Kodu ekle:** Mac'te koşan script `experiments/NN-slug/experiment.py`, Colab notebook'u `notebooks/NN-slug.ipynb`.
5. Künyede durumu güncelle ve [SUMMARY.md](SUMMARY.md) ile faz sayfasındaki durumu eşitle.
6. **İki dil.** Varsayılan dil İngilizce (repo kökü); Türkçe sürüm aynı dosya adlarıyla `tr/` altında. Türkçede teknik terimler İngilizce kalır.

## Kod kuralları

- Kod, değişken adları ve yorumlar İngilizce.
- Bağımlılık sürümleri sabit (`mujoco==3.17.0`).
- Bir script tek bir sweep koşturur ve sonucu tablo olarak yazdırır.

## Hata bildirmek

Bir sayı tutmuyorsa: dersi, kullandığın sürümleri ve çıktını yaz. Kitaptaki her sayının yeniden üretilebilir olması gerekiyor; tutmayan bir sayı bir hatadır.
