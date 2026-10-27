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

> Her filtre noise'u delay karşılığında azaltır; doğru pencere, elindeki hareket için iki hatanın eşitlendiği penceredir.

## Problem

Bir joint controller ([ders 12](../part-4-dynamics-and-control/12-joint-control.md)) konum ve hız ister. Gerçek bir kolda ikisi de doğrudan gelmez: quantization ve elektriksel noise içeren encoder sayımları gelir, hızı da ancak bunların türevini alarak elde edersin. Türev, noise'u $1/\Delta t$ kadar büyütür; 500 Hz'de ham finite difference kullanılamaz. Akla gelen çözüm (önce yumuşat) ise delay ekler ve feedback döngüsü bunu phase lag olarak görür (latency'nin genel hâli [ders 37](../part-9-sim-to-real/37-latency.md)'de). Bu ders, simüle edilmiş bir SO-101 eklemi üzerinde bu takası ölçer ve filtre penceresi seçmek için bir kural verir.

## Ana fikir

**Sensor modelleri.** Turda $N$ sayım veren bir encoder $\tilde q = \Delta\,\mathrm{round}(q/\Delta)$, $\Delta = 2\pi/N$ raporlar. Eklem hareket ederken quantization tek başına varyansı $\Delta^2/12$ olan düzgün dağılımlı noise gibi davranır. Elektronik için üstüne beyaz noise $v\sim\mathcal N(0,\sigma^2)$ eklenir. IMU'daki gyro $\tilde\omega = \omega + b + n$ verir: yavaş değişen bir bias $b$ artı beyaz noise. Bunu integre edince açı hatası $b\,t$ gibi büyür; gyro tek başına bir yönelimi bu yüzden tutamaz. Accelerometer specific force ölçer; durağanken sıfır değil, yerçekimini okur.

**MuJoCo sensor'leri.** `jointpos`, `jointvel`, `accelerometer`, `gyro` (ve diğerleri) `mjData.sensordata` dizisine yazar. `noise` attribute'u yalnızca metadata'dır: dokümantasyon simülasyonu etkilemediğini söyler ([MuJoCo Modeling · sensors](https://mujoco.readthedocs.io/en/stable/modeling.html#sensor-noise)). Noise'u sen eklersin.

**İki filtre** (causal, örnekleme süresi $\Delta t$):

| Filtre | Güncelleme | Düşük frekansta delay |
|---|---|---|
| Moving average, pencere $N$ | $y_k = \frac1N\sum_{i=0}^{N-1} z_{k-i}$ | $\frac{N-1}{2}\Delta t$ (tam, doğrusal faz) |
| Birinci derece low-pass, cutoff $f_c$ | $y_k = y_{k-1} + \alpha(z_k - y_{k-1})$, $\alpha = \frac{\Delta t}{\tau+\Delta t}$, $\tau = \frac{1}{2\pi f_c}$ | $f \ll f_c$ için $\approx\tau$ |

İkisi de beyaz noise varyansını kabaca ekledikleri delay ile ters orantılı azaltır (moving average: $\frac{N-1}{2}\Delta t$ delay karşılığında varyans $\sigma^2/N$). Delay'in bedeli yaklaşık $|\dot q|\cdot t_d$ kadar takip hatasıdır. Toplam konum hatası şöyle davranır:

$$e_{\text{rms}}^2 \approx \underbrace{\sigma^2 \cdot g(t_d)}_{\text{noise, delay ile düşer}} + \underbrace{\overline{\dot q^2}\, t_d^2}_{\text{gecikme, delay ile büyür}}$$

ve bir minimumu vardır. Minimumun yeri eklemin ne kadar hızlı hareket ettiğine bağlıdır. Dersin tamamı bu.

## Deney

**Kurulum.** MuJoCo Menagerie'den SO-101, timestep 2 ms'ye ayarlı (500 Hz örnekleme). `shoulder_pan`, position actuator'ü üzerinden 10 s boyunca $0{,}5\sin(2\pi\,0{,}2t) + 0{,}2\sin(2\pi\,0{,}5t)$ rad izler. 12-bit bir encoder (tur başına 4096 sayım, adım 1,534 mrad) artı $\sigma = 10$ mrad beyaz noise modelliyoruz. Bunlar öğretim değerleri, datasheet değeri değil. Hız, filtrelenmiş sinyalin `np.gradient` (central difference) türeviyle hesaplanır. Delay, filtrelenmiş ve gerçek sinyali en iyi hizalayan gecikmedir.

```python
s = spec.add_sensor(name="q_pan", type=mujoco.mjtSensor.mjSENS_JOINTPOS,
                    objname="shoulder_pan", objtype=mujoco.mjtObj.mjOBJ_JOINT)
s.noise = 0.01                          # metadata only
...
z = np.round(q / step) * step + rng.normal(0, SIGMA, q.shape)   # our encoder
y = moving_average(z, n)                # or low_pass(z, fc)
```

**Tahmin.** Ham central-difference hız noise'u yaklaşık $\sqrt2\,\sigma/(2\Delta t) \approx 3{,}5$ rad/s olmalı, yani gerçek hızdan büyük. Konum için en iyi pencere kısa, hız için en iyi pencere çok daha uzun olmalı.

**Çalıştır:** `python experiments/17-sensors-noise-filtering/experiment.py` (MuJoCo 3.17.0, NumPy 2.5.4, Apple M5, deterministic).

İlk kontroller: `noise=0.01` ile tanımlanan `jointpos` sensor'ü `qpos`'tan tam olarak 0,0 rad farklı. (Sensor'ler `mj_step`'in başında, integrasyondan önceki state'ten hesaplanır; o state ile karşılaştırmazsan bir adımlık kayma görürsün.) Ham okumanın konum RMS hatası 9,95 mrad, hız RMS hatası 3,50 rad/s; gerçek hızın RMS'i ise 0,63 rad/s.

| Filtre | Delay (ms) | Konum RMS (mrad) | Hız RMS (rad/s) |
|---|---|---|---|
| ham | 0,0 | 9,95 | 3,50 |
| MA N=3 | 2,0 | 5,91 | 1,65 |
| **MA N=5** | 4,0 | **5,17** | 1,01 |
| MA N=9 | 8,0 | 6,03 | 0,55 |
| MA N=17 | 16,0 | 10,22 | 0,29 |
| MA N=33 | 32,0 | 19,76 | 0,16 |
| **MA N=65** | 64,0 | 39,01 | **0,12** |
| MA N=129 | 128,0 | 76,52 | 0,19 |
| LPF 50 Hz | 4,0 | 5,30 | 1,35 |
| LPF 20 Hz | 8,0 | 6,02 | 0,71 |
| LPF 10 Hz | 16,0 | 10,16 | 0,39 |
| LPF 2 Hz | 78,0 | 47,33 | 0,14 |
| LPF 0,5 Hz | 274,0 | 152,59 | 0,35 |

**Gözlem.**

1. Ölçülen moving-average delay'i her satırda $(N-1)\Delta t/2$'ye eşit. Low-pass delay'i, $f_c$ sinyal frekanslarına yaklaşana kadar $\tau$ ile örtüşür (10 Hz'de 15,9 → 16,0 ms): 0,5 Hz'de 318,3 ms değil 274 ms'dir, çünkü $\tau$ phase delay'in yalnızca düşük frekans limitidir.
2. Konum hatası yaklaşık 4 ms delay'de minimum (MA N=5, 5,17 mrad) ve sonrasında neredeyse doğrusal büyür: 17 örnekten itibaren filtre, hiç filtre olmamasından kötüdür.
3. Hız, 16 kat delay ister (MA N=65, 64 ms, 0,12 rad/s). Tek bir filtre ikisine birden hizmet edemez. Ders 18'deki Kalman filter bu boşluğu kapatır.
4. Eşit delay'de iki filtre tipi neredeyse aynı (8 ms'de MA N=9: 6,03 mrad; LPF 20 Hz: 6,02 mrad). Filtrenin şeklinden çok, kabul ettiğin delay önemlidir.

IMU kontrolü (aynı script): durağan kolda accelerometer $|a| = 9{,}810$ m/s² okur. 0,02 rad/s bias ve 0,05 rad/s noise'lu bir gyro 10 s integre edilince, hiç dönmeyen bir eklem için 0,209 rad (12,0°) dönüş raporlar.

## Boz

MA N=17'yi (16 ms delay) sabit tut ve iki frekansı $k$ ile ölçekleyerek hareketi hızlandır:

| k | Gerçek hız RMS (rad/s) | Ham konum RMS (mrad) | MA17 konum RMS (mrad) |
|---|---|---|---|
| 0,5 | 0,28 | 9,94 | 5,01 |
| 1 | 0,61 | 9,95 | 10,22 |
| 2 | 1,27 | 9,95 | 20,46 |
| 4 | 2,51 | 9,95 | 40,00 |
| 8 | 3,73 | 9,94 | 59,49 |

Ham hata hıza aldırmaz. Filtrelenmiş hata $|\dot q|\cdot t_d$'yi izler: 0,61 rad/s × 16 ms ≈ 9,8 mrad, ölçülen 10,22. Yavaş bir test hareketinde ayarlanmış pencere, hızlı harekette filtresizden kötü hâle gelir. Filtreyi robotun gerçekten yapacağı en hızlı hareket üzerinde ayarla ya da hareketi ortalamak yerine tahmin eden model tabanlı bir filtre kullan.

## Gerçek robotta

- SO-101 follower, 1/345 dişli oranlı altı STS3215 servo kullanır ([LeRobot SO-101 dokümanı](https://github.com/huggingface/lerobot/blob/main/docs/source/so101.mdx)). Konum, seri bus üzerinden servonun iç encoder'ından gelir; [Menagerie modelinde](https://github.com/google-deepmind/mujoco_menagerie/tree/f3b6013/robotstudio_so101) hız sensor'ü ve IMU yoktur. STS3215 encoder çözünürlüğünü doğrulamadık; yukarıdaki 4096 sayım bizim modelleme seçimimiz.
- Gerçek donanımda okuma hızını 2 ms'lik timestep değil bus belirler; dolayısıyla $\Delta t$ ve yukarıdaki delay'ler değişir. Kendi döngü hızın için yeniden hesapla.
- Kendi $\sigma$'nı ölçme prosedürü: kolu sabit bir pozda torklu tut, her eklemi 10 s logla, eklem başına standart sapmayı al. Durağan hâlde saf bir quantizer sıfır varyans gösterebilir; bu durumda yavaş bir rampa komutla ve oturtulan doğrudan kalan farka bak. Gerçek kolda henüz ölçülmedi.

## Kendini dene

1. **Kolay.** 64 örneklik bir moving average 200 Hz'de çalışıyor. Delay'i nedir?
2. **Orta.** Tabloda en iyi hız penceresi (65) neden en iyi konum penceresinden (5) çok daha uzun? $1/\Delta t$ noise büyütmesi argümanını kullan.
3. **Zor.** `np.gradient` yerine causal backward difference $(y_k - y_{k-1})/\Delta t$ kullan. Ham hız noise'unu tahmin et, sonra çalıştır. En iyi hız penceresi değişiyor mu?

### Cevaplar

1. $(64-1)/2 \times 5$ ms $= 157{,}5$ ms.
2. Türev, kalan noise'u yaklaşık $1/\Delta t = 500$ s⁻¹ ile çarpar; gecikme terimi ise yalnızca $|\dot q|\,t_d$'den $|\ddot q|\,t_d$'ye değişir. Kendi gecikme terimine göre noise terimi hızda konumdakinden çok daha büyüktür; bu yüzden noise + gecikme toplamının minimumu çok daha uzun bir delay'e kayar.
3. Tahmin edilen ham noise $\sqrt2\,\sigma/\Delta t \approx 7{,}1$ rad/s (central difference 2Δt tabanı kullandığı için onun iki katı). Yeni optimumu görmek için çalıştır; biz çalıştırmadık.

## Akılda kalanlar ve kaynaklar

- MuJoCo sensor'leri tamdır; `noise` metadata'dır. Quantization, beyaz noise ve gyro bias'ını sen eklersin.
- Filtre delay'i tahmin edilebilir ($\frac{N-1}{2}\Delta t$ ya da $\approx\tau$) ve gecikme hatası yaklaşık $|\dot q|\,t_d$'dir. En iyi pencere hareket hızına bağlıdır ve konum ile hız için farklıdır.
- Bias'lı bir gyro'yu integre etmek doğrusal drift üretir; düzeltmek için ikinci bir sensor ya da model gerekir.

Sıradaki: [ders 18](18-kalman-filter.md), Kalman filter.

Kaynaklar:
- [MuJoCo Modeling · Sensors](https://mujoco.readthedocs.io/en/stable/modeling.html#sensor-noise): sensor'ler simülasyonu etkilemez; `noise` yalnızca saklanan standart sapmadır.
- [MuJoCo XML reference · accelerometer](https://mujoco.readthedocs.io/en/stable/XMLreference.html#sensor-accelerometer): site frame'inde yerçekimi dahil doğrusal ivme verir.
- [LeRobot SO-101 dokümanı](https://github.com/huggingface/lerobot/blob/main/docs/source/so101.mdx): altı STS3215 servolu follower kol.
- [MuJoCo Menagerie · robotstudio_so101](https://github.com/google-deepmind/mujoco_menagerie/tree/f3b6013/robotstudio_so101) (Apache-2.0).
- Yukarıdaki tüm sayılar: [`experiment.py`](../../experiments/17-sensors-noise-filtering/experiment.py), çıktı [`output.txt`](../../experiments/17-sensors-noise-filtering/output.txt).
