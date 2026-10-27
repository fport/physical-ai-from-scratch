# 17 · Sensor'ler, noise ve filtering

<!-- meta -->
| | |
|---|---|
| **Bölüm** | [5 · Perception ve estimation](README.md) |
| **Çalışma türü** | CPU · Mac ya da Colab CPU |
| **Önkoşul** | [12](../part-4-dynamics-and-control/12-joint-control.md) |
| **Önce öğren** | [Probability and Distributions](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/06-probability-and-distributions) · [Stochastic Processes](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/22-stochastic-processes) |
| **Süre** | ~45 dk |
| **Kod** | [`experiments/17-sensors-noise-filtering/experiment.py`](../../experiments/17-sensors-noise-filtering/experiment.py) |
| **Durum** | ✅ tam |
<!-- /meta -->

> Her filtre daha az noise'u delay karşılığında satın alır; doğru pencere, elindeki hareket için iki hatanın eşitlendiği penceredir.

## Problem

Bir joint controller ([ders 12](../part-4-dynamics-and-control/12-joint-control.md)) eklemin konumuna ve hızına ihtiyaç duyar. Gerçek bir kol ikisini de doğrudan vermez. Küçük basamaklarla zıplayan ve elektriksel noise ile titreyen encoder okumaları verir; hızı ise ancak ardışık okumalardan sen hesaplarsan alırsın. Okumaları yumuşatmak titremeyi giderir ama onları geç bırakır; feedback loop'u ise geç veriye kötü tepki verir ([ders 37](../part-9-sim-to-real/37-latency.md)). Bu ders o ödünleşimi simüle edilmiş bir SO-101 ekleminde ölçer ve filtre seçmek için bir kuralla biter.

## Ana fikir

### Sade anlatımla

Titreyen bir hız göstergesi düşün. İbreyi bir kez okursan noisy bir sayı alırsın. Son on okumanın ortalamasını alırsan sayı sakinleşir, ama artık arabanın şu anki değil biraz önceki hızını anlatır. Ne kadar çok okumanın ortalamasını alırsan sayı o kadar sakin ve o kadar eski olur.

Yani her yumuşatma filtresinde ters yöne çeken iki hata vardır:

- **Noise hatası**: titremeden geriye kalan. Daha çok ortalama aldıkça küçülür.
- **Gecikme hatası**: ortalamasını aldığın okumalardan bu yana eklemin ne kadar ilerlediği. Daha çok ortalama aldıkça büyür; eklem hızlı hareket ediyorsa daha da hızlı büyür.

En iyi filtre, ikisinin toplamının en küçük olduğu yerdir. Bu yerin neresi olduğu eklemin ne kadar hızlı hareket ettiğine bağlıdır. Dersin tamamı bu.

### Sinyal zinciri

```mermaid
flowchart TD
    Q["gerçek açı q(t)"] --> E["encoder<br/>basamağa yuvarlar"]
    E --> N["+ elektriksel noise"]
    N --> F["filtre<br/>(ortalama / low-pass)"]
    F --> C["controller"]
    F -. "iki örneğin<br/>farkı" .-> V["hız tahmini"]
    V --> C
```

### İki filtre, kod olarak

Bir sayı akışı üzerinde rolling average yazdıysan ilk filtreyi zaten biliyorsun. Okumalar her `dt` saniyede bir gelir (burada `dt = 0.002`, yani saniyede 500); Sıfırdan yaz iki filtreyi tam olarak yazar.

```python
# Moving average: the mean of the last n readings
y[k] = mean(z[k - n + 1 : k + 1])
# First-order low-pass: move a fraction alpha toward each new reading
alpha = dt / (tau + dt)                      # tau = 1 / (2 pi fc); small alpha = heavy smoothing
y[k] = y[k - 1] + alpha * (z[k] - y[k - 1])
```

`n` okumalık bir moving average penceresinin ortasındaki, yani $(n-1)/2$ örnek önceki değeri raporlar; low-pass filtre yaklaşık zaman sabiti `tau` kadar geç kalır.

| Filtre | Ayar | Delay | Kalan noise (std) |
|---|---|---|---|
| Moving average | pencere $n$ | $\frac{n-1}{2}\,\Delta t$ | $\sigma/\sqrt{n}$ |
| Birinci derece low-pass | cutoff $f_c$ | $\approx \tau = \frac{1}{2\pi f_c}$ | $f_c$ düştükçe azalır |

Burada $\Delta t$ iki okuma arasındaki süre, $\sigma$ tek bir okumanın noise standart sapmasıdır. Gecikme hatası kabaca *hız × delay*'dir: 0,6 rad/s ile dönen bir eklem 16 ms'lik bir filtreden bakınca yaklaşık $0{,}6 \times 0{,}016 \approx 0{,}01$ rad geride görünür.

<details>
<summary>Matematik ayrıntısı: σ/√n ve hata ödünleşimi nereden geliyor</summary>

**Ortalama almak noise'u √n kadar azaltır.** $n$ okuma, varyansı $\sigma^2$ olan bağımsız noise'lar $v_i$ taşıyorsa, ortalamalarındaki noise $\frac1n\sum v_i$'dir ve varyansı $\frac{1}{n^2}\cdot n\sigma^2 = \sigma^2/n$'dir. Standart sapma $\sigma/\sqrt n$ olarak düşer. Bağımsızlık önemlidir: korelasyonlu noise (örneğin yavaş bir drift) bu yolla gitmez.

**Moving average'ın delay'i.** Yavaş değişen bir $q(t)$ sinyali için pencere ortalamasının şu ana göre Taylor açılımı $\frac1n\sum_{i=0}^{n-1} q(t - i\Delta t) \approx q\big(t - \tfrac{n-1}{2}\Delta t\big)$ verir: çıkış, yarım pencere kadar gecikmiş sinyaldir.

**Low-pass filtre.** $y_k = y_{k-1} + \alpha(z_k - y_{k-1})$, $\tau\dot y = z - y$ denkleminin ayrık hâlidir. $f_c = 1/(2\pi\tau)$'nun çok altındaki frekanslarda phase delay'i $\tau$'ya yaklaşır; $f_c$ civarında ve üstünde daha küçüktür. Deney bunu 0,5 Hz'de gösteriyor.

**Ödünleşim.** Delay $t_d$ iken filtrelenmiş konum kabaca $q(t - t_d)$ artı kalan noise'tur; dolayısıyla

$$e_{\text{rms}}^2 \approx \underbrace{\sigma^2\, g(t_d)}_{\text{noise, delay ile düşer}} + \underbrace{\overline{\dot q^{\,2}}\; t_d^2}_{\text{gecikme, delay ile büyür}}$$

burada $g$, $t_d$ büyüdükçe azalır (moving average için $t_d = \frac{n-1}{2}\Delta t$ iken $g = 1/n$). Düşen bir terim ile büyüyen bir terimin toplamının bir minimumu vardır.

**Hız daha zordur.** İki okumadan hız, $(z_{k+1} - z_{k-1})/(2\Delta t)$, onların noise'unu küçük bir sayıya böler: noise yaklaşık $1/\Delta t$ katı büyür. Bu yüzden hız için noise terimi çok daha büyüktür ve minimumu çok daha uzun bir delay'dedir.

Daha derine (random variable, varyans, noise süreçleri): [Probability and Distributions](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/06-probability-and-distributions), [Stochastic Processes](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/22-stochastic-processes).
</details>

<!-- figure: 17-moving-average -->

### Kısaca sensor'ler

| Sensor | Ne raporlar | Tipik hata |
|---|---|---|
| Encoder (turda $N$ sayım) | $2\pi/N$ basamaklarına yuvarlanmış açı | quantization (basamak boyu) + elektriksel noise |
| Gyroscope | açısal hız | yavaş değişen **bias** + noise; bias'ı integre etmek doğrusal olarak kayan bir açı verir |
| Accelerometer | specific force | durağan hâlde sıfır değil **yerçekimi** okur |

MuJoCo sensor'leri (`jointpos`, `jointvel`, `gyro`, `accelerometer`, …) `mjData.sensordata` dizisine tam değerler yazar. `noise` attribute'u yalnızca saklanan metadata'dır: dokümantasyon simülasyonu etkilemediğini söyler ([MuJoCo Modeling · sensors](https://mujoco.readthedocs.io/en/stable/modeling.html#sensor-noise)). Noise'u sen eklersin.

## Sıfırdan yaz

Şimdi sensor modelini ve iki filtreyi yaklaşık 40 satır NumPy ile kendin yazacak, her birini SciPy'a ya da Ana fikir'deki bir formüle karşı kontrol edeceksin. MuJoCo'dan yalnızca eklemin gerçek hareketi gelir (Deney'in `simulate` fonksiyonu); böylece Sıfırdan yaz ile Deney aynı 5000 okumayı puanlar.

Adımları kendi dosyana yaz ve her adımdan sonra çalıştır. Bitmiş fonksiyonlar [`physai/estimation.py`](../../physai/estimation.py) içinde; `uv run python experiments/17-sensors-noise-filtering/build.py` beş adımı kontrolleriyle çalıştırır (çıktı: [`build_output.txt`](../../experiments/17-sensors-noise-filtering/build_output.txt)).

### Adım 1 · Encoder

Encoder açıyı tam sayımlara yuvarlar; elektronik üstüne noise ekler:

```python
def encoder_reading(q, counts, sigma, rng):
    """What an encoder reports for true angles q: rounded to steps of 2 pi / counts, plus white noise."""
    step = 2 * np.pi / counts                       # one count, in rad
    return np.round(q / step) * step + rng.normal(0, sigma, np.shape(q))
```

$\Delta$ basamaklarına yuvarlamak $-\Delta/2$ ile $\Delta/2$ arasında eşit dağılmış bir hata bırakır; standart sapması $\Delta/\sqrt{12}$'dir. Bunu önce bir milyon rastgele açıda, sonra dersin sinyalinde kontrol et:

```text
== Step 1. The encoder: 4096 counts/rev, sigma 10 mrad ==
step = 1.534 mrad, max |rounding error| = 0.767 mrad (step / 2 = 0.767)
rounding error std 0.4427 mrad vs step / sqrt(12) = 0.4428 mrad
lesson signal: max |ours - Exp. encoder| = 0.0e+00 rad
expected reading error sqrt(sigma^2 + step^2/12) = 10.01 mrad, measured 9.95 mrad
```

Yuvarlama payı (0,44 mrad) 10 mrad'lık noise'un yanında küçük; beklenen okuma hatası 10,01 mrad. Bu seed'in noise çekilişi %0,6 düşük çıkıyor; Deney'in 9,95 mrad ölçmesinin sebebi bu.

### Adım 2 · Moving average

Kümülatif toplam her pencere ortalamasını tek bir çıkarmaya indirir; maliyet $n$ ile büyümez:

```python
def moving_average(x, n):
    """Causal moving average: the mean of the last n samples (fewer at the start), as in lesson 17."""
    c = np.cumsum(np.insert(x, 0, 0.0))
    i = np.arange(len(x))
    lo = np.maximum(0, i + 1 - n)
    return (c[i + 1] - c[lo]) / (i + 1 - lo)
```

Üç kontrol: pencereler üzerinde düz bir döngüyle aynı sayılar, saf white noise'ta kalan noise'un $1/\sqrt n$ ile karşılaştırması ve delay. Rampa delay'i kesinleştirir: bir doğrunun son $n$ örneğinin ortalaması, onun $(n-1)/2$ örnek önceki değeridir.

```text
== Step 2. Moving average: window mean, noise left, delay on a ramp ==
n = 5: max |ours - loop over windows| = 2.2e-16
    n  noise std left  1/sqrt(n)  ramp lag samples  (n-1)/2
    5          0.4459     0.4472               2.0      2.0
   17          0.2415     0.2425               8.0      8.0
   65          0.1226     0.1240              32.0     32.0
```

### Adım 3 · Birinci derece low-pass

```python
def low_pass(x, fc, dt):
    """First-order low-pass with cutoff fc (Hz): move a fraction alpha toward each new sample."""
    tau = 1.0 / (2 * np.pi * fc)                    # time constant (s)
    alpha = dt / (tau + dt)                         # small alpha = heavy smoothing
    y = np.empty(len(x))
    y[0] = x[0]                                     # start at the first reading
    for k in range(1, len(x)):
        y[k] = y[k - 1] + alpha * (x[k] - y[k - 1])
    return y
```

Bu recursive bir filtre, $y_k = \alpha z_k + (1-\alpha) y_{k-1}$; `scipy.signal.lfilter` onu `b = [alpha]`, `a = [1, alpha - 1]` katsayılarıyla çalıştırır. SciPy'ı `lfilter_zi(b, a) * z[0]` ile başlatmak, seninki gibi ilk okumadan başlamasını sağlar. Kontrol edilecek iki formül: birim varyanslı white noise varyansının $\alpha/(2-\alpha)$'sını bırakır, rampada çıkış $(1-\alpha)/\alpha$ örnek geride kalır; $\alpha = \Delta t/(\tau + \Delta t)$ ile bu tam olarak $\tau/\Delta t$'dir.

```text
== Step 3. First-order low-pass: vs scipy.signal.lfilter, noise left, delay on a ramp ==
fc Hz   alpha  max |ours - lfilter|  noise std left  sqrt(a/(2-a))  ramp lag samples  tau/dt
   50  0.3859               2.2e-16          0.4878         0.4889              1.59    1.59
   10  0.1116               3.3e-16          0.2423         0.2431              7.96    7.96
    2  0.0245               1.8e-15          0.1103         0.1114             39.79   39.79
```

Rampa gecikmesi her cutoff'ta iki ondalığa kadar $\tau$. Kalan noise formülden yaklaşık %1 içinde (fark, sonlu noise örneğinden).

### Adım 4 · Farklardan hız

```python
def central_difference(y, dt):
    """Velocity from samples: (y[k+1] - y[k-1]) / 2 dt inside, one-sided at the two ends."""
    v = np.empty(len(y))
    v[1:-1] = (y[2:] - y[:-2]) / (2 * dt)
    v[0], v[-1] = (y[1] - y[0]) / dt, (y[-1] - y[-2]) / dt
    return v
```

`np.gradient(y, dt)` de bunu yapar; kontrol son bitine kadar doğruluyor. Bağımsız 10,01 mrad hatalı iki okuma, $2\Delta t$ = 4 ms'ye bölününce ham hız noise'unu öngörür:

```text
== Step 4. Velocity by central differences ==
max |ours - np.gradient| = 0.0e+00 rad/s
raw velocity error: predicted sqrt(2) * 10.01 mrad / (2 dt) = 3.54 rad/s, measured 3.50 rad/s (true velocity RMS 0.63)
```

Öngörülen 3,54, ölçülen 3,50 rad/s: Adım 1'deki aynı %0,6 düşük çekiliş. Noise gerçek hızın beş katından fazla; hızın konumdan çok daha ağır filtre istemesinin sebebi bu.

### Adım 5 · Delay'i ve hataları ölç

Deney bir filtrenin delay'ini, filtrelenmiş sinyali gerçeğiyle en iyi hizalayan kaydırma olarak tanımlar:

```python
def delay_samples(y, ref, max_lag=400, skip=500):
    """The shift (in samples) that best aligns a filtered signal y with the true signal ref."""
    errs = [np.mean((y[skip + lag:] - ref[skip:len(ref) - lag]) ** 2) for lag in range(max_lag)]
    return int(np.argmin(errs))
```

Deney'deki her filtreyi kendi fonksiyonlarından geçir ve onun sayılarıyla karşılaştır:

```text
== Step 5. Every filter on the lesson's signal vs the Experiment ==
      filter  delay ms  theory ms  pos RMS mrad  vel RMS rad/s  max |ours - Exp.|
      MA n=1       0.0        0.0          9.95           3.50            0.0e+00
      MA n=3       2.0        2.0          5.91           1.65            0.0e+00
      MA n=5       4.0        4.0          5.17           1.01            0.0e+00
      MA n=9       8.0        8.0          6.03           0.55            0.0e+00
     MA n=17      16.0       16.0         10.22           0.29            0.0e+00
     MA n=33      32.0       32.0         19.76           0.16            0.0e+00
     MA n=65      64.0       64.0         39.01           0.12            0.0e+00
    MA n=129     128.0      128.0         76.52           0.19            0.0e+00
   LPF 50 Hz       4.0        3.2          5.30           1.35            0.0e+00
   LPF 20 Hz       8.0        8.0          6.02           0.71            0.0e+00
   LPF 10 Hz      16.0       15.9         10.16           0.39            0.0e+00
    LPF 2 Hz      78.0       79.6         47.33           0.14            0.0e+00
  LPF 0.5 Hz     274.0      318.3        152.59           0.35            0.0e+00
lag error of MA n=17: speed x delay = 0.61 rad/s x 16 ms = 9.8 mrad
```

Her satır Deney tablosuna son basamağına kadar eşit. Ölçülen delay'ler her pencere için $(n-1)\Delta t/2$ ile örtüşüyor. Low-pass'te ise yalnızca $f_c$ hareketin çok üstündeyken $\tau$ ile örtüşüyor (delay'ler tam 2 ms'lik örneklerdir, bu yüzden 50 Hz satırı 3,2 ms'yi 4,0 gösteriyor; 0,5 Hz satırı 318,3 değil 274 ms). Son satır, MA n=17 için Ana fikir'deki gecikme hatası: ölçülen 10,22 mrad'ın 9,8 mrad'ı.

### Elinde kalan

`physai.estimation` artık `encoder_reading`, `moving_average`, `low_pass`, `central_difference` ve `delay_samples` içeriyor. [Ders 18](18-kalman-filter.md) aynı encoder modelini ve baseline olarak `moving_average`'ı kullanır, iki filtrenin yerine öngören tek bir filtre koyar; [ders 37](../part-9-sim-to-real/37-latency.md) delay'e bir feedback döngüsü içinde döner. [`tests/test_17_sensors_noise_filtering.py`](../../tests/test_17_sensors_noise_filtering.py) onları SciPy'a sabitler; bir şeyi değiştirdikten sonra `uv run pytest tests/test_17_sensors_noise_filtering.py` çalıştır.

## Kütüphaneyle kullan

SciPy iki filtreyi de tek çağrıyla çalıştırır: `scipy.signal.lfilter(b, a, x)`, yani $a_0 y_k = \sum b_i x_{k-i} - \sum_{j\ge1} a_j y_{k-j}$ fark denklemi:

```python
from scipy.signal import group_delay, lfilter, lfilter_zi

y_ma = lfilter(np.ones(n) / n, 1.0, z)                    # moving average (first n-1 outputs differ)
alpha = dt / (1 / (2 * np.pi * fc) + dt)
b, a = [alpha], [1.0, alpha - 1.0]
y_lp, _ = lfilter(b, a, z, zi=lfilter_zi(b, a) * z[0])   # low-pass, starting at the first reading
_, gd = group_delay((b, a), w=[0.2], fs=1 / dt)          # delay in samples at 0.2 Hz
```

`group_delay` delay'i sinyal olmadan yalnızca katsayılardan verir: moving average için $(n-1)/2$ örnek, low-pass için düşük frekansta $\tau/\Delta t$; ikisi de testlerde kontrol ediliyor. Başka araçlar aynı pencereyi farklı yerleştirir:

| Çağrı | Pencere | Başlangıç |
|---|---|---|
| physai `moving_average(x, n)` | son $n$ örnek (causal) | o ana kadar görülen örneklerin ortalaması |
| `scipy.signal.lfilter(np.ones(n)/n, 1, x)` | son $n$ örnek (causal) | başlangıçtan önceki örnekler 0 sayılır ("initial rest") |
| `scipy.ndimage.uniform_filter1d(x, n)` | örneğe ortalanmış (`origin=0`), geleceği kullanır | kenarlarda `mode='reflect'`; `origin=(n-1)//2` onu causal yapar (testlerde kontrol edildi) |
| `np.gradient(y, dt)` | bir önceki ve bir sonraki örnek (central) | iki uçta tek taraflı farklar |

Kaynaklar: [`lfilter`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html), [`group_delay`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.group_delay.html), [`uniform_filter1d`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.uniform_filter1d.html), [`np.gradient`](https://numpy.org/doc/stable/reference/generated/numpy.gradient.html).

Üç tuzak:

- **Ortalanmış filtre bedava görünür.** `uniform_filter1d` ve `np.gradient` gelecekteki örnekleri kullanır: offline sıfır delay gösterirler, çalışan bir controller'da o örnekler henüz yoktur. Deney'in hızı `np.gradient` kullanır ve bu yüzden bir örnek ileriye bakar.
- **"$f_c$'de birinci derece low-pass" tek bir filtre değildir.** `scipy.signal.butter(1, 10, fs=500)` `b = [0.0592, 0.0592]`, `a = [1, -0.8816]` döndürür (bilinear transform); bu dersin $\alpha$'sı `a = [1, -0.8884]` verir. İkisine de 10 Hz'lik birinci derece low-pass denir.
- **`noise` attribute'u hiçbir şey eklemez.** MuJoCo onu metadata olarak saklar; bu dersteki noise, Adım 1'de senin eklediğin noise'dur.

## Deney

**Kurulum.** MuJoCo Menagerie'den SO-101, timestep 2 ms (500 Hz örnekleme). `shoulder_pan`, position actuator'ü üzerinden 10 s boyunca $0{,}5\sin(2\pi\,0{,}2t) + 0{,}2\sin(2\pi\,0{,}5t)$ rad izler. 12-bit bir encoder (turda 4096 sayım, basamak 1,534 mrad) artı beyaz noise $\sigma = 10$ mrad modelliyoruz. Bunlar datasheet değil öğretim değerleri. Hız, filtrelenmiş sinyalin `np.gradient` (central difference) ile türeviyle hesaplanır. Delay, filtrelenmiş sinyalle gerçek sinyali en iyi hizalayan kaydırmadır.

```python
s = spec.add_sensor(name="q_pan", type=mujoco.mjtSensor.mjSENS_JOINTPOS,
                    objname="shoulder_pan", objtype=mujoco.mjtObj.mjOBJ_JOINT)
s.noise = 0.01                          # metadata only
...
z = np.round(q / step) * step + rng.normal(0, SIGMA, q.shape)   # our encoder
y = moving_average(z, n)                # or low_pass(z, fc)
```

**Tahmin et.** Ham hız noise'u yaklaşık $\sqrt2\,\sigma/(2\Delta t) \approx 3{,}5$ rad/s olmalı, yani gerçek hızdan büyük. En iyi konum penceresi kısa, en iyi hız penceresi çok daha uzun olmalı.

**Çalıştır** `uv run python experiments/17-sensors-noise-filtering/experiment.py` (MuJoCo 3.17.0, NumPy 2.5.4, Apple M5, deterministik).

İlk kontroller: `noise=0.01` ile tanımlanan `jointpos` sensor'ü `qpos`'tan tam olarak 0,0 rad farklı. (Sensor'ler `mj_step`'in başında, integrasyondan önceki state'ten hesaplanır; o state ile karşılaştır, yoksa bir adımlık kayma görürsün.) Ham okumanın konum RMS hatası 9,95 mrad, hız RMS hatası 3,50 rad/s; gerçek hız RMS'i ise 0,63 rad/s.

![Moving average ve low-pass filtre için konum ve hız hatasının filtre delay'ine göre değişimi. Konum hatası yaklaşık 4 ms delay'de, hız hatası yaklaşık 64 ms'de en düşük.](../../experiments/17-sensors-noise-filtering/noise_vs_delay.png)

| Filtre | Delay (ms) | Konum RMS (mrad) | Hız RMS (rad/s) |
|---|---|---|---|
| ham | 0,0 | 9,95 | 3,50 |
| MA n=3 | 2,0 | 5,91 | 1,65 |
| **MA n=5** | 4,0 | **5,17** | 1,01 |
| MA n=9 | 8,0 | 6,03 | 0,55 |
| MA n=17 | 16,0 | 10,22 | 0,29 |
| MA n=33 | 32,0 | 19,76 | 0,16 |
| **MA n=65** | 64,0 | 39,01 | **0,12** |
| MA n=129 | 128,0 | 76,52 | 0,19 |
| LPF 50 Hz | 4,0 | 5,30 | 1,35 |
| LPF 20 Hz | 8,0 | 6,02 | 0,71 |
| LPF 10 Hz | 16,0 | 10,16 | 0,39 |
| LPF 2 Hz | 78,0 | 47,33 | 0,14 |
| LPF 0,5 Hz | 274,0 | 152,59 | 0,35 |

**Gözlem.**

1. Moving-average delay'i her satırda $(n-1)\Delta t/2$'ye eşit. Low-pass delay'i, $f_c$ sinyal frekanslarına yaklaşana kadar $\tau$ ile örtüşür (10 Hz'de 15,9 → 16,0 ms): 0,5 Hz'de 318,3 ms değil 274 ms'dir, çünkü $\tau$ yalnızca düşük frekans limitidir.
2. Konum hatası yaklaşık 4 ms delay'de en düşük (MA n=5, 5,17 mrad) ve sonrasında neredeyse doğrusal büyür: 17 örnekten itibaren filtre, hiç filtre olmamasından kötüdür.
3. Hız 16 kat delay ister (MA n=65, 64 ms, 0,12 rad/s). Tek bir filtre ikisine birden hizmet edemez. Ders 18'deki Kalman filter bu boşluğu kapatır.
4. Eşit delay'de iki filtre tipi neredeyse aynı (8 ms'de MA n=9: 6,03 mrad; LPF 20 Hz: 6,02 mrad). Filtrenin şeklinden çok, kabul ettiğin delay önemlidir.

IMU kontrolü (aynı script): durağan kolda accelerometer $|a| = 9{,}810$ m/s² okur. 0,02 rad/s bias ve 0,05 rad/s noise'lu bir gyro 10 s integre edilince, hiç dönmeyen bir eklem için 0,209 rad (12,0°) dönüş raporlar.

## Boz

MA n=17'yi (16 ms delay) sabit tut ve iki frekansı $k$ ile ölçekleyerek hareketi hızlandır:

| k | Gerçek hız RMS (rad/s) | Ham konum RMS (mrad) | MA17 konum RMS (mrad) |
|---|---|---|---|
| 0,5 | 0,28 | 9,94 | 5,01 |
| 1 | 0,61 | 9,95 | 10,22 |
| 2 | 1,27 | 9,95 | 20,46 |
| 4 | 2,51 | 9,95 | 40,00 |
| 8 | 3,73 | 9,94 | 59,49 |

Ham hata hıza aldırmaz. Filtrelenmiş hata *hız × delay*'i izler: 0,61 rad/s × 16 ms ≈ 9,8 mrad, ölçülen 10,22. Yavaş bir test hareketinde ayarlanmış pencere, hızlı harekette filtresizden kötü hâle gelir. Filtreyi robotun gerçekten yapacağı en hızlı hareket üzerinde ayarla ya da hareketi ortalamak yerine tahmin eden model tabanlı bir filtre kullan (ders 18).

## Gerçek robotta

- SO-101 follower, 1/345 dişli oranlı altı Feetech STS3215 servo kullanır ([LeRobot SO-101 dokümanı](https://github.com/huggingface/lerobot/blob/main/docs/source/so101.mdx)). Konum, seri bus üzerinden servonun iç encoder'ından gelir; [Menagerie modelinde](https://github.com/google-deepmind/mujoco_menagerie/tree/f3b6013/robotstudio_so101) hız sensor'ü ve IMU yoktur. STS3215 encoder çözünürlüğünü doğrulamadık; yukarıdaki 4096 sayım bizim modelleme seçimimiz.
- Gerçek donanımda okuma hızını 2 ms'lik timestep değil bus belirler; dolayısıyla $\Delta t$ ve yukarıdaki delay'ler değişir. Kendi loop hızın için yeniden hesapla.
- Kendi $\sigma$'nı ölçmek: kolu sabit bir pozda tut, her eklemi 10 s logla, eklem başına standart sapmayı al. Durağan hâlde saf bir quantizer sıfır varyans gösterebilir; bu durumda yavaş bir rampa komutla ve oturtulan doğrudan kalan farka bak. Gerçek kolda henüz ölçülmedi.

## Kendini dene

1. **Kolay.** 64 örneklik bir moving average 200 Hz'de çalışıyor. Delay'i nedir?
2. **Orta.** Tabloda en iyi hız penceresi (65) neden en iyi konum penceresinden (5) çok daha uzun? "Noise'u Δt'ye bölmek" argümanını kullan.
3. **Zor.** `np.gradient` yerine causal backward difference $(y_k - y_{k-1})/\Delta t$ kullan. Ham hız noise'unu tahmin et, sonra çalıştır. En iyi hız penceresi değişiyor mu?

### Cevaplar

1. $(64-1)/2 \times 5$ ms $= 157{,}5$ ms.
2. Türev almak kalan noise'u yaklaşık $1/\Delta t = 500$ s⁻¹ ile çarpar; gecikme terimi ise yalnızca hız × delay'den ivme × delay'e değişir. Noise terimi hızda konumdakinden çok daha büyüktür; bu yüzden noise + gecikme toplamının minimumu çok daha uzun bir delay'e kayar.
3. Tahmin edilen ham noise $\sqrt2\,\sigma/\Delta t \approx 7{,}1$ rad/s; central difference iki örnek aralığını kapsadığı için onun iki katı. Yeni optimumu görmek için çalıştır; biz çalıştırmadık.

## Akılda kalanlar ve kaynaklar

- MuJoCo sensor'leri tamdır; `noise` metadata'dır. Quantization, beyaz noise ve gyro bias'ını sen eklersin.
- Bir filtrenin delay'i tahmin edilebilir ($\frac{n-1}{2}\Delta t$ ya da $\approx\tau$) ve gecikme hatası yaklaşık hız × delay'dir. En iyi pencere eklemin ne kadar hızlı hareket ettiğine bağlıdır ve konum ile hız için farklıdır.
- Bias'lı bir gyro'yu integre etmek doğrusal drift üretir; düzeltmek için ikinci bir sensor ya da model gerekir.

Sıradaki: [ders 18](18-kalman-filter.md), Kalman filter.

Kaynaklar:
- [MuJoCo Modeling · Sensors](https://mujoco.readthedocs.io/en/stable/modeling.html#sensor-noise): sensor'ler simülasyonu etkilemez; `noise` yalnızca saklanan standart sapmadır.
- [MuJoCo XML reference · accelerometer](https://mujoco.readthedocs.io/en/stable/XMLreference.html#sensor-accelerometer): site frame'inde yerçekimi dahil doğrusal ivme verir.
- [LeRobot SO-101 dokümanı](https://github.com/huggingface/lerobot/blob/main/docs/source/so101.mdx): altı STS3215 servolu follower kol.
- [MuJoCo Menagerie · robotstudio_so101](https://github.com/google-deepmind/mujoco_menagerie/tree/f3b6013/robotstudio_so101) (Apache-2.0).
- Yukarıdaki tüm sayılar: [`experiment.py`](../../experiments/17-sensors-noise-filtering/experiment.py), çıktı [`output.txt`](../../experiments/17-sensors-noise-filtering/output.txt).
