# Ek C · Mac'te inference hızı

> Bir Mac'te küçük bir policy çoğu zaman CPU'da GPU'dan daha hızlıdır: GPU, yalnızca büyük batch'lerin ya da büyük modellerin amorti ettiği sabit bir başlatma-ve-senkronizasyon bedeli öder.

PyTorch'un MPS backend'i tensor'ları Metal Performance Shaders üzerinden Apple GPU'sunda çalıştırır; macOS 14.0+ ister ve `torch.device("mps")` ile seçilir ([PyTorch MPS notları](https://docs.pytorch.org/docs/2.14/notes/mps.html)). Robot için önemli sayı, batch 1'de tek bir forward geçişinin latency'sidir, çünkü kontrol döngüsü her seferinde bir action (ya da bir chunk) ister.

**Ölçüm.** Bölüm 7'nin eğittiği boyutta iki policy: bir MLP (64 → 256 → 256 → 6, 83.974 parametre) ve 3 × 96 × 96'lık bir görüntüden 50 × 6'lık bir chunk çıkaran küçük bir CNN (352.492 parametre). 10 ısınma geçişinden sonra 50 forward geçişinin medyanı (minimumu), `torch.no_grad()`, MPS süreleri `torch.mps.synchronize()` dahil. PyTorch 2.14.1, Python 3.14, Apple M5, macOS 26.3.1, 4 CPU thread'i. Script: [`experiments/appendix-c-inference-speed/experiment.py`](../../experiments/appendix-c-inference-speed/experiment.py), çıktı: [`output.txt`](../../experiments/appendix-c-inference-speed/output.txt).

**Makine bu koşu sırasında ağır yüklüydü** (paralel çalışan çok sayıda başka iş vardı). Bu sayılar ödünleşimin şeklini gösteriyor, benchmark değil; alıntılamadan önce boş bir makinede yeniden koştur.

| policy | batch | CPU [ms] | MPS [ms] |
|---|---|---|---|
| MLP | 1 | 0,010 (0,010) | 0,344 (0,274) |
| MLP | 8 | 0,022 (0,021) | 0,338 (0,282) |
| MLP | 64 | 0,039 (0,034) | 0,433 (0,312) |
| MLP | 256 | 0,290 (0,131) | 0,439 (0,407) |
| CNN | 1 | 0,534 (0,408) | 1,159 (0,979) |
| CNN | 8 | 3,722 (2,748) | 2,714 (2,408) |
| CNN | 64 | 473,753 (278,449) | 4,280 (3,333) |
| CNN | 256 | 1894,905 (1054,068) | 11,946 (11,118) |

**Okuma.** MLP denediğimiz her batch boyutunda CPU'da daha hızlı; MPS 0,338 ile 0,439 ms (medyan) arasında kalıyor, kabaca çağrı başına sabit bedeli. CNN batch 1'de de CPU'da daha hızlı (0,534'e karşı 1,159 ms); batch 8'den itibaren MPS kazanıyor. Batch 64 ve 256'daki CPU satırlarını bir ölçekleme yasası gibi okuma: batch 8'in 8 katı iş 127 kat uzun sürdü. Şüphelilerden biri yarışan süreçler, ama tek CPU thread'iyle hızlı bir yeniden koşu batch 8'in üstünde benzer bir sıçrama gösterdi; yani büyük batch'lerde daha yavaş bir CPU convolution yolu da olası. Sebebi ayırmadık. Batch 1'deki kontrolü etkilemiyor.

**Kontrol döngüsü için.** 30 Hz'de bütçe adım başına 33 ms; iki policy de batch 1'de iki cihaza da sığıyor. Bir VLA için çağrı başına bedel çok daha büyük (π0: RTX 4090'da chunk başına 73 ms, [arXiv 2410.24164](https://arxiv.org/abs/2410.24164)); bu yüzden chunking ve asenkron inference ([SmolVLA, arXiv 2506.01844](https://arxiv.org/abs/2506.01844)) cihaz seçiminden daha önemli. Bu Mac'te bir VLA ölçmedik.

**Pratik kurallar.** Tam modeli batch 1'de iki cihazda da senkronizasyonla ölç; modeli tek bir cihazda tut (her adımda tensor'ları CPU ile MPS arasında taşımak kopya ekler); yukarıdaki ölçüm script'ini boş bir makinede yeniden kullan.
