# 25 · Robot policy'lerini evaluate etmek

<!-- meta -->
| | |
|---|---|
| **Bölüm** | [7 · Learning from demonstration](README.md) |
| **Çalışma türü** | CPU · Mac ya da Colab CPU |
| **Önkoşul** | [16](../part-4-dynamics-and-control/16-grasping-and-pick-place.md) |
| **Önce öğren** | [Statistics for Machine Learning](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/15-statistics-for-ml) · [Model Evaluation](https://aiengineeringfromscratch.com/lesson?path=phases/02-ml-fundamentals/09-model-evaluation) |
| **Süre** | ~40 dk |
| **Kod** | [`experiments/25-evaluating-policies/experiment.py`](../../experiments/25-evaluating-policies/experiment.py) |
| **Durum** | ✅ tam |
<!-- /meta -->

> Deneme sayısı olmayan bir başarı oranı ölçüm değildir. k/n'yi, bir Wilson aralığını ve her denemeye karar veren fiziksel testi raporla.

## Problem

Bölüm 7 policy'ler eğitiyor ve her birine tek bir soru soruyor: küpü kaldırıyor mu? "%90 başarı" gibi cevaplar genellikle 10 ya da 20 rollout'a dayanır. 10 denemeyle, uzun vadeli oranı 0,74 olan bir policy beş evaluation'dan birinde 9/10 ya da daha iyisini yapar (aşağıda ölçüldü); yani tek bir "9/10", 0,74'lük bir policy'yi 0,88'lik birinden ayıramaz. Bir şey eğitmeden önce bu ders kitabın geri kalanının kullandığı evaluation protokolünü kuruyor: neyin başarı sayıldığı, kaç deneme gerektiği, hangi aralığın raporlanacağı ve iki policy'nin gerçekten ne zaman farklı olduğu.

## Ana fikir

**Başarı bir Bernoulli sonucudur.** Deneme $i$, fiziksel bir testin karar verdiği $X_i \in \{0, 1\}$ ile biter. Bizimki [ders 16](../part-4-dynamics-and-control/16-grasping-and-pick-place.md)'dan: sonda küpün altı zeminden ≥ 4 cm yukarıda ve iki çene de küpe değiyor; weld ya da bağlantı yok. Denemeler bağımsız ve aynı oran $p$'ye sahipse $k = \sum X_i$ binom dağılır ve $\hat p = k/n$ (bkz. AIEFS: [Statistics for Machine Learning](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/15-statistics-for-ml)).

**Wilson aralığı.** Ders kitabındaki (Wald) aralık $\hat p \pm z\sqrt{\hat p(1-\hat p)/n}$, $k = 0$ ya da $k = n$'de sıfır genişliğe çöker ve küçük $n$'de gerektiğinden az kapsar. Wilson (1927) bunun yerine score testini tersine çevirir:

$$\frac{\hat p + \frac{z^2}{2n}}{1 + \frac{z^2}{n}} \;\pm\; \frac{z}{1 + \frac{z^2}{n}}\sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}},\qquad \%95 \text{ için } z = 1{,}96.$$

Brown, Cai ve DasGupta (2001), Wald aralığının kapsamasının düzensiz olduğunu gösterir ve küçük $n$ için Wilson'ı (ya da Jeffreys'i) önerir.

**Gereken deneme sayısı.** Yarı genişlik $1/\sqrt n$ gibi küçülür: yarıya indirmek dört kat deneme ister.

**Varyansın birden fazla kaynağı var.** Binom aralığı yalnız evaluation gürültüsünü kapsar. Öğrenilmiş bir policy training seed'leri ve veri alt kümeleri arasında da değişir; bir *yöntem* hakkındaki iddia, her biri aynı test konumlarında evaluate edilen birkaç training koşusu ister. Ders 28 seed başına ve birleşik sonuçları raporluyor.

**İki policy'yi karşılaştırmak.** Örtüşen aralıklar eşitliği kanıtlamaz ve ayrık %95 aralıkları doğrudan bir karşılaştırmadan daha katı bir testtir. $k_A/n_A$ ile $k_B/n_B$ için iki örneklemli bir test kullan, örneğin 2 × 2 tablo üzerinde Fisher'ın exact testi ([SciPy `fisher_exact`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html)).

## Deney

Oranı bilinmeyen bir "policy": ders 16'nın scripted pick'i, küpün gürültülü bir tahminine nişan alıyor; gerçek konum + eksen başına $\mathcal N(0, \sigma^2)$, sanki kusurlu bir algı modülünden geliyormuş gibi. Küp konumları x ∈ [16, 28] cm, y ∈ [−8, 8] cm içinde düzgün dağılımlı. Policy A'da σ = 9 mm, policy B'de σ = 6 mm. Her biri 300 simüle denemelik bir havuz alıyor; havuzun oranı uzun vadeli oran $p$'nin yerine geçiyor. $n$ denemelik evaluation'lar sonra havuzdan iadeli çekiliyor. Tam script: [`experiments/25-evaluating-policies/experiment.py`](../../experiments/25-evaluating-policies/experiment.py), çıktı: [`output.txt`](../../experiments/25-evaluating-policies/output.txt). MuJoCo 3.17.0, NumPy 2.5.4, SciPy 1.18.1, Apple Silicon Mac (Apple M5), deterministik, yaklaşık 1 dk CPU (yaklaşık; çok yüklü bir makinede duvar saati süresi birkaç kat uzundu).

```python
def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)
```

**A. Havuzlar.**

| Policy | Nişan gürültüsü | Başarı | Wilson %95 | "Çene küpe kapandı" |
|---|---|---|---|---|
| A | 9 mm | 221/300 (0,737) | [0,684, 0,783] | 259/300 (0,863) |
| B | 6 mm | 263/300 (0,877) | [0,835, 0,909] | 288/300 (0,960) |

Son sütun aynı denemelerde sayılan daha gevşek bir kriter: çene, iki çene de küpe değerek oturdu. A'ya 38, B'ye 25 "başarı" ekliyor; bunlar dokunulan ya da sıkılan ama kaldırılıp tutulamayan küpler. Başarı testi evaluation'dan önce yazılmalı ve fiziksel olmalı.

**B. Deneme sayısı.** *Tahmin et:* $n = 10$'da $p \approx 0{,}74$ etrafındaki bir %95 aralığı ne kadar geniş? Satır başına 2000 evaluation, havuz A'dan:

| n | ortalama Wilson genişliği | Wilson p'yi kapsıyor | Wald p'yi kapsıyor | exact Wilson | exact Wald | k/n'nin %5–95'i |
|---|---|---|---|---|---|---|
| 5 | 0,577 | 0,987 | 0,764 | 0,981 | 0,764 | 0,40–1,00 |
| 10 | 0,463 | 0,971 | 0,928 | 0,975 | 0,927 | 0,50–0,90 |
| 20 | 0,354 | 0,967 | 0,919 | 0,962 | 0,908 | 0,55–0,90 |
| 50 | 0,235 | 0,951 | 0,916 | 0,947 | 0,914 | 0,62–0,84 |
| 100 | 0,169 | 0,942 | 0,942 | 0,946 | 0,945 | 0,66–0,81 |

"Exact", $p$ = 0,737'de binom dağılımından hesaplanan kapsama; yeniden örneklenen sütunlar onunla uyuşuyor. 10 denemede aralık 0,46 genişliğinde ve gözlenen oran 0,50 ile 0,90 arasında herhangi bir yere düşüyor. Wilson %95'in yaklaşık bir puan yakınında kalıyor (0,942–0,987; küçük $n$'de biraz temkinli); Wald $n = 5$'te 0,76'ya düşüyor.

**C. "9/10" ne demek.** 9/10 için Wilson %95 [0,596, 0,982]; 10/10 için [0,722, 1,000]. ≥ 9/10 görme olasılığı:

| Policy | uzun vadeli p | P(≥ 9/10), binom | havuzdan çekilen |
|---|---|---|---|
| A | 0,737 | 0,215 | 0,208 |
| B | 0,877 | 0,645 | 0,634 |

Bir 9/10 iki policy ile de tutarlı ve B evaluation'ların yaklaşık üçte birinde 9/10 üretemiyor.

**D. Hedef yarı genişlik için deneme sayısı** (Wilson %95):

| gözlenen oran | ±0,10 | ±0,05 | ±0,02 |
|---|---|---|---|
| 0,50 | 93 | 381 | 2398 |
| 0,70 | 78 | 320 | 2014 |
| 0,90 | 34 | 141 | 864 |
| 0,95 | 24 | 82 | 465 |

**E. B, A'dan iyi mi?** Gerçek fark 0,14. Her biri $n$ denemelik 1000 evaluation çifti:

| policy başına n | Fisher p < 0,05 | Wilson aralıkları ayrık | B ≤ A skor yapıyor |
|---|---|---|---|
| 10 | 0,037 | 0,006 | 0,286 |
| 20 | 0,113 | 0,025 | 0,170 |
| 50 | 0,335 | 0,152 | 0,041 |
| 100 | 0,673 | 0,423 | 0,005 |

Her birinde 20 denemeyle daha iyi policy zamanın %17'sinde aynı ya da daha kötü skor yapıyor ve fark yalnız %11 oranında anlamlı çıkıyor. Her birinde 100 deneme bile 14 puanlık bir farkı ancak üç seferde iki kez yakalıyor.

## Boz

$n$'yi düşür, $p$'yi 1'e yaklaştır; Wald aralığı bozulur. $n = 5$'te gerçek oranı evaluation'ların %95'i yerine %76'sında kapsıyor. 10/10'da [1,000, 1,000] raporluyor, yani on denemeden kesinlik iddiası; Wilson ise [0,722, 1,000] veriyor. Mekanizma: Wald varyansa $\hat p$'yi koyar, bu yüzden tahmin edilen varyans tam da $\hat p$ 0 ya da 1 olduğunda sıfırdır, ve $np$ ya da $n(1-p)$ küçükken normal yaklaşım kötüdür. Wilson aralığı $z^2/2n$ kadar 0,5'e doğru merkezler ve hiç kaybolmayan bir $z^2/4n^2$ terimi tutar.

## Gerçek robotta

- **Protokol** (SO-101'imizde henüz koşulmadı): masada işaretlenmiş bir ızgarada 20 küp konumu sabitle ve karşılaştırılan her policy için aynı konumları kullan. Başarı testini koşmadan önce yaz: küp ≥ 4 cm kaldırıldı ve 1 s tutuldu, videodan değerlendirildi. Küpü işarete elle geri koy. k/n'yi, Wilson %95'i ve başarısızlık türlerini (ıskaladı, itti, düşürdü) raporla.
- **İlk bozulan varsayım bağımsızlıktır.** Servo ısınması, aşınan bir küp, kayan bir kamera ve operatörün reset'leri denemeleri birbirine bağlar. Önce A'nın hepsini koşmak yerine policy'leri sırayla dönüşümlü koş (A, B, A, B, …).
- LeRobot'un `lerobot-rollout`'u evaluation episode'larını bir dataset'e kaydedebilir (örn. `--strategy.type=sentry`; `base` kayıt yapmadan çalışır) ([il_robots](https://huggingface.co/docs/lerobot/il_robots)). Başarı kararını ve aralığı tanımlamak ve raporlamak yine sana kalır.

## Kendini dene

1. **Kolay.** Bir policy 18/20 yapıyor. Yukarıdaki kodla Wilson %95 aralığını ver.
2. **Orta.** D tablosuna göre gözlenen oran 0,70'te ±0,05'lik bir iddia yaklaşık kaç deneme ister? 0,95'te neden daha az?
3. **Zor.** Aynı yöntemin 3 training seed'ini koş ve her birini aynı 20 konumda evaluate et. Sonucu hem seed varyansı hem deneme varyansı görünecek şekilde nasıl raporlayacağını öner.

### Cevaplar

1. [0,699, 0,972] (`wilson(18, 20)`; çıktının C kısmında da yazıyor).
2. 320. Binom varyansı $p(1-p)$ 0,70'te 0,21, 0,95'te 0,0475; aynı hassasiyet kenarlara yakın daha az deneme ister.
3. Kabul edilebilir bir cevap: her seed'in k/20'sini Wilson aralığıyla, artı birleşik k/60'ı aralığıyla ve seed'ler arasındaki min–max'ı raporla. Tek başına birleşik aralık seed varyansını gizler.

## Akılda kalanlar ve kaynaklar

- k/n'yi, bir Wilson %95 aralığını ve fiziksel başarı testini raporla. Simülasyonda daha gevşek bir "çene kapandı" testi policy A'yı 0,737'den 0,863'e şişirdi.
- On deneme yaklaşık 0,46 genişliğinde bir aralık verir; 0,74'lük bir policy evaluation'ların yaklaşık %21'inde ≥ 9/10 üretir.
- 0,74'lük bir policy'yi 0,88'likten güvenilir biçimde ayırmak her biri için 100'ün üzerinde deneme ister; her birinde 20 denemeyle Fisher testi farkı %11 oranında yakaladı.

Sonraki: [ders 26 · Teleoperation, replay ve data collection](26-teleop-replay-data.md).

Kaynaklar:
- E. B. Wilson, ["Probable inference, the law of succession, and statistical inference"](https://doi.org/10.1080/01621459.1927.10502953), *J. Amer. Statist. Assoc.* 22 (158), 209–212, 1927.
- L. D. Brown, T. T. Cai, A. DasGupta, ["Interval Estimation for a Binomial Proportion"](https://doi.org/10.1214/ss/1009213286), *Statistical Science* 16 (2), 101–133, 2001.
- SciPy, [`scipy.stats.fisher_exact`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html).
- LeRobot, [Imitation learning on real-world robots](https://huggingface.co/docs/lerobot/il_robots).
- Ölçülen tüm sayılar: [`experiments/25-evaluating-policies/output.txt`](../../experiments/25-evaluating-policies/output.txt).
