# 18 · Kalman filter

<!-- meta -->
| | |
|---|---|
| **Bölüm** | [5 · Perception ve estimation](README.md) |
| **Çalışma türü** | CPU · Mac ya da Colab CPU |
| **Önkoşul** | [17](17-sensors-noise-filtering.md) |
| **Önce öğren** | [Probability and Distributions](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/06-probability-and-distributions) · [Bayes' Theorem](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/07-bayes-theorem) |
| **Süre** | ~60 dk |
| **Kod** | [`experiments/18-kalman-filter/experiment.py`](../../experiments/18-kalman-filter/experiment.py) |
| **Durum** | ✅ tam |
<!-- /meta -->

> Kalman filter, eklemin nasıl hareket ettiğini bilen bir moving average'dır: geride kalmadan yumuşatır ve ne kadar yanıldığını düşündüğünü de söyler.

## Problem

Ders 17, aynı noisy SO-101 eklem sinyali üzerinde ölçülmüş bir çelişkiyle bitti: en iyi konum filtresinin delay'i 4 ms'ydi (5 örneklik moving average, 5,17 mrad), en iyi hız filtresi 64 ms istiyordu (65 örnek, 0,12 rad/s). Bir controller ikisini de tek bir estimator'dan, gecikmesiz ve onlara ne kadar güveneceğini söyleyen bir sayıyla birlikte ister. Hareketin doğrusal bir modelini yazabiliyorsan Kalman'ın 1960'taki recursive filtresi ([Kalman 1960](https://doi.org/10.1115/1.3662552)) bunu yapar.

## Ana fikir

State $x_k\in\mathbb R^n$, ölçüm $z_k$ ([Welch & Bishop, TR 95-041](https://techreports.cs.unc.edu/papers/95-041.pdf), denklem 1.1–1.2):

$$x_k = F x_{k-1} + w_{k-1},\quad w\sim\mathcal N(0,Q) \qquad z_k = H x_k + v_k,\quad v\sim\mathcal N(0,R)$$

**Predict:** $\hat x^-_k = F\hat x_{k-1}$, $\;P^-_k = F P_{k-1} F^\top + Q$.
**Update:** $K_k = P^-_k H^\top (H P^-_k H^\top + R)^{-1}$, $\;\hat x_k = \hat x^-_k + K_k(z_k - H\hat x^-_k)$, $\;P_k = (I - K_k H)P^-_k$.

Bayes yorumu AIEFS'te (bkz. AIEFS: [Bayes' Theorem](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/07-bayes-theorem)). Burada mühendislik hâli lazım.

**Tek eklem için constant-velocity modeli.** $x = [q,\ \dot q]^\top$; encoder yalnızca $q$'yu görür:

$$F = \begin{bmatrix}1 & \Delta t\\ 0 & 1\end{bmatrix},\quad H = [1\ \ 0],\quad Q = q_c\begin{bmatrix}\Delta t^3/3 & \Delta t^2/2\\ \Delta t^2/2 & \Delta t\end{bmatrix}$$

$Q$, ivmeyi yoğunluğu $q_c$ (rad²/s³) olan beyaz noise olarak modellemekten gelir. $R$ encoder varyansıdır: burada $\sigma^2 + \Delta^2/12$.

| Ayar | Fiziksel anlamı | Nasıl belirlenir |
|---|---|---|
| $R$ | ölçüm noise varyansı | ölç: sensor'ü durağanken ya da bilinen bir hareket boyunca logla ([Welch & Bishop](https://techreports.cs.unc.edu/papers/95-041.pdf), §1 "Filter Parameters and Tuning") |
| $q_c$ | eklemin habersizce ne kadar sert ivmelenebileceği | ayarla: tek gerçek tasarım seçimi |

Deneyden önce bilinmesi gereken iki gerçek: $Q$ ve $R$ sabitse $P_k$ ve $K_k$ sabitlere yakınsar; yani kararlı hâldeki filtre sabit bir doğrusal filtredir (Welch & Bishop da bunu söyler). Ve başlangıç geçişi bittikten sonra tahmin yalnızca $q_c/R$ oranına bağlıdır; mutlak ölçek sadece raporlanan $P$'yi değiştirir.

## Deney

**Kurulum.** Ders 17 ile aynı sinyal: MuJoCo'da SO-101 `shoulder_pan`, 2 ms timestep, aynı trajectory, 12-bit encoder modeli artı 10 mrad beyaz noise, aynı seed; noise dizisi birebir aynı. $R = 1{,}002\times10^{-4}$ rad² (σ 10,01 mrad). $q_c$'yi altı dekad boyunca sweep ediyor; konum ve hız RMS hatasını, delay'i ve **2σ coverage**'ı ölçüyoruz: gerçek konumun $\pm2\sqrt{P_{00}}$ içinde kaldığı örneklerin oranı. Hatası iddia ettiği $P$ ile uyumlu Gauss bir filtre için coverage yaklaşık %95'tir.

```python
F = np.array([[1.0, DT], [0.0, 1.0]]); H = np.array([[1.0, 0.0]])
Q = q_psd * np.array([[DT**3 / 3, DT**2 / 2], [DT**2 / 2, DT]])
for k in range(len(z)):
    x = F @ x;  P = F @ P @ F.T + Q                       # predict
    S = H @ P @ H.T + r;  K = (P @ H.T) / S               # gain
    x = x + (K * (z[k] - H @ x)).ravel();  P = (np.eye(2) - K @ H) @ P   # update
```

**Tahmin.** $q_c$ çok küçükse filtre constant-velocity modeline güvenir, çok yumuşatır, geride kalır ve aşırı emin olur. Çok büyükse noise'u takip eder.

**Çalıştır:** `python experiments/18-kalman-filter/experiment.py` (MuJoCo 3.17.0, NumPy 2.5.4, Apple M5, deterministic).

| $q_c$ | Konum RMS (mrad) | Delay (ms) | Hız RMS (rad/s) | $K_q$ | $K_{\dot q}$ | 2σ coverage (%) |
|---|---|---|---|---|---|---|
| 0,001 | 20,71 | 16,0 | 0,250 | 0,0235 | 0,140 | 8,0 |
| 0,01 | 6,91 | 4,0 | 0,138 | 0,0414 | 0,437 | 32,4 |
| 0,1 | 3,23 | 0,0 | **0,084** | 0,0724 | 1,361 | 91,0 |
| **1** | **3,15** | 0,0 | 0,093 | 0,1251 | 4,179 | 97,8 |
| 10 | 4,02 | 0,0 | 0,203 | 0,2116 | 12,545 | 98,1 |
| 100 | 5,22 | 0,0 | 0,482 | 0,3448 | 36,165 | 97,8 |
| 1000 | 6,61 | 0,0 | 1,136 | 0,5285 | 97,018 | 97,3 |

Aynı veride ders 17 baseline'ları: MA N=5, 4,0 ms delay ile 5,17 mrad; MA N=65, 0,120 rad/s hız hatası.

**Gözlem.**

1. $q_c = 1$'de filtre, ders 17'nin en iyi konum filtresini sıfır ölçülen delay ile geçer (3,15'e karşı 5,17 mrad); $q_c = 0{,}1$'de en iyi hız filtresini geçer (0,084'e karşı 0,120 rad/s), hem de tek bir filtreyle. Baseline'lar bir örnek ileriye bakan `np.gradient` kullandı; Kalman filter tamamen causal ve yine kazanıyor.
2. Sıfır delay sihir değil: predict adımı tahmin edilen hızla ileriye extrapolate eder; yumuşak bir hareket "geçmişe doğru ortalanmaz".
3. Başlangıç geçişinden sonra gain'ler sabittir (tablo son $K$'yı gösteriyor). Kararlı hâldeki Kalman filter, gain'lerini koda gömebileceğin sabit, iki state'li doğrusal bir filtredir.
4. Coverage dürüstlük testidir. $q_c = 0{,}001$'de filtre 16 ms geride kalırken küçük bir $P$ iddia ediyor: örneklerin yalnızca %8,0'ı kendi 2σ bandının içinde. Ayarı bozuk bir filtrenin $P$'sini outlier eleme ya da sensor füzyonu gibi kararlar için asla kullanma.

## Boz

**Yanlış R.** Gerçek noise sabit; filtreye $R_{\text{used}} = f\cdot R$ söyleniyor, $q_c = 1$:

| $R_{\text{used}}/R$ | Konum RMS (mrad) | Hız RMS (rad/s) | 2σ coverage (%) |
|---|---|---|---|
| 0,01 | 5,22 | 0,482 | 17,5 |
| 0,1 | 4,02 | 0,203 | 53,9 |
| 1 | 3,15 | 0,093 | 97,8 |
| 10 | 3,23 | 0,084 | 100,0 |
| 100 | 6,91 | 0,138 | 100,0 |

RMS sütunları $q_c$ sweep'ini birebir tekrarlar: $q_c=1$'de $R\times0{,}01$, gerçek $R$ ile $q_c = 100$'ün verdiği 5,22 mrad / 0,482 rad/s'yi verir. Tahmin yalnızca $q_c/R$'yi görür. Coverage ise tekrarlanmaz: $R$'yi küçük tahmin etmek filtreyi aşırı emin yapar (%17,5), büyük tahmin etmek karamsar yapar (%100). Yani RMS hatası iyi görünürken covariance yanlış olabilir.

**Modellenmemiş ivme.** $t = 5$ s'de eklem komutuna 0,3 rad'lık bir basamak ekle. Position actuator 4,05 rad/s'lik gerçek tepe hızla cevap verir; constant-velocity modeli bu ivmeye "noise" der.

| $q_c$ | Basamaktan sonraki 0,5 s'de en büyük konum hatası (mrad) | En büyük hız hatası (rad/s) | Basamaktan önce konum RMS (mrad) |
|---|---|---|---|
| 1 | 44,1 | 2,92 | 3,22 |
| 100 | 19,6 | 2,57 | 5,26 |
| 10000 | 22,3 | 7,74 | 7,98 |

Ayarlı filtre basamakta 44,1 mrad'a kadar geride kalır. $q_c$'yi yüz kat artırmak geçişi yarıya indirir ama yumuşak hareketteki doğruluğu düşürür (3,22'ye karşı 5,26 mrad); $10^4$'te filtre noise'u izler ve hız hatası patlar. Mekanizma: modelin bilinen bir komutu temsil etme yolu yok. Çözüm daha büyük $Q$ değil, daha iyi model: komut edilen ivmeyi ya da motor torkunu control input $Bu$ olarak ver (Welch & Bishop denklem 1.1) ya da bilinen geçişlerde $Q$'yu değiştir.

## Gerçek robotta

- $R$ tahmin edilmez, ölçülür: her SO-101 eklemini sabit bir pozda ve yavaş bir rampa boyunca logla (prosedür [ders 17](17-sensors-noise-filtering.md)'de). Gerçek kolda henüz ölçülmedi.
- Seri bus okumaları jitter'lı gelir. $F$ ve $Q$'yu nominal $\Delta t$ ile değil, son örnekten bu yana geçen gerçek süreyle kur; ikisi de yukarıda açıkça $\Delta t$'nin fonksiyonu.
- SO-101 servoları kendi position loop'larını içeride kapatır (LeRobot onları position moduna alır ve PID gain'lerini yazar, [`so_follower.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/so_follower.py)); host tarafındaki bir Kalman filter servonun iç döngüsünü değil, logladığın, policy'ye verdiğin ya da collision tespiti için kullandığın hız ve state tahminlerini iyileştirir.
- Her ekleme gönderdiğin komutu biliyorsan (biliyorsun), $Q$'yu büyütmeden önce onu modele koy. Yukarıdaki basamak testi, aksi hâlde her hızlı kalkış ve duruşta göreceğin hatadır.

## Kendini dene

1. **Kolay.** Tabloda yalnızca hızı önemseseydin hangi $q_c$'yi seçerdin? Yalnızca konumu?
2. **Orta.** $q_c = 1$'de $R_{\text{used}} = 0{,}1R$'nin, gerçek $R$ ile $q_c = 10$'la neden birebir aynı RMS'yi verdiğini açıkla.
3. **Zor.** State'i $[q,\dot q,\ddot q]$'ya genişlet (constant-acceleration modeli) ve basamak testini tekrar çalıştır. Geçiş, yumuşak kısmı bozmadan küçülüyor mu?

### Cevaplar

1. Hız: $q_c = 0{,}1$ (0,084 rad/s). Konum: $q_c = 1$ (3,15 mrad).
2. $Q$ ve $R$'yi (ve etkisi geçişten sonra kaybolan başlangıç $P$'sini) aynı katsayıyla ölçeklemek $P$'yi o katsayıyla ölçekler ve $K = P^-H^\top(HP^-H^\top+R)^{-1}$'yi değiştirmez. $q_c=1, R/10$ ile $q_c=10, R$ aynı orandır.
3. Burada hesaplanmadı; çalıştır. Basamakta daha küçük bir gecikme ve yumuşak kısımda daha noisy bir hız bekle, çünkü ek state de yalnızca konumdan tahmin edilmek zorunda.

## Akılda kalanlar ve kaynaklar

- Constant-velocity Kalman filter, tek bir noisy encoder'dan konumu ve hızı causal olarak ve ortalama filtrelerinin delay'i olmadan verir (burada 3,15 mrad ve 0,084–0,093 rad/s).
- Tahmini yalnızca $q_c/R$ belirler; mutlak değerler covariance'ı belirler. $P$'ye güvenmeden önce coverage'a bak.
- Modellenmemiş ivme geçici gecikme olarak görünür; $Q$'yu şişirmeden önce modeli düzelt (control input).

Sıradaki: [ders 19](19-camera-calibration.md), camera model ve calibration.

Kaynaklar:
- R. E. Kalman, "A New Approach to Linear Filtering and Prediction Problems", *Journal of Basic Engineering* 82(1):35–45, 1960. [doi:10.1115/1.3662552](https://doi.org/10.1115/1.3662552)
- G. Welch, G. Bishop, "An Introduction to the Kalman Filter", TR 95-041, UNC Chapel Hill (2004 güncellemesi). [PDF](https://techreports.cs.unc.edu/papers/95-041.pdf)
- Sinyal ve noise modeli: [ders 17](17-sensors-noise-filtering.md).
- Yukarıdaki tüm sayılar: [`experiment.py`](../../experiments/18-kalman-filter/experiment.py), çıktı [`output.txt`](../../experiments/18-kalman-filter/output.txt).
