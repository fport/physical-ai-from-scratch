# 13 · Trajectory generation

<!-- meta -->
| | |
|---|---|
| **Bölüm** | [4 · Dynamics ve control](README.md) |
| **Çalışma türü** | CPU · Mac ya da Colab CPU |
| **Önkoşul** | [10](../part-3-kinematics/10-inverse-kinematics.md), [12](12-joint-control.md) |
| **Önce öğren** | [Calculus for Machine Learning](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/04-calculus-for-ml) |
| **Süre** | ~45 dk |
| **Kod** | [`experiments/13-trajectory-generation/experiment.py`](../../experiments/13-trajectory-generation/experiment.py) |
| **Durum** | ✅ tam |
<!-- /meta -->

> Bir controller ona ne verirsen onu takip eder. Bir step verirsen sonsuz ivme üretmeye çalışır; yumuşak bir eğri verirsen torque küçük kalır.

## Problem

Ders 12'de Unitree'nin gerçek robot için yazdığı ayağa kalkma örneği duruş pozunu tek seferde göndermiyordu: 2 ms'lik 500 ila 1000 tick süren fazlarla ona doğru interpolasyon yapıyordu ([`go2_stand_example.py`](https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/example/go2/low_level/go2_stand_example.py)). Robot arayüzleri daha da ileri gider: libfranka'nın `rate_limiting.h` dosyası, 1 ms'lik kontrol periyodunda joint başına en fazla 10 rad/s² ivme, 5000 rad/s³ jerk ve 1000 N·m/s torque değişim hızı tanımlar ve komutları bu sınırlara kırpan `limitRate` fonksiyonları sunar ([`rate_limiting.h`](https://github.com/frankarobotics/libfranka/blob/main/include/franka/rate_limiting.h)). Bir planner ya da IK solver sana *nereye* gideceğini söyler; trajectory generation ise *referansın zamanda nasıl ilerleyeceğine* karar verir. Bu seçim tepe torque'u, yapıyı sarsan jerk'ü ve takip hatasını belirler. Bu ders üçünü de bir Panda üzerinde ölçüyor.

## Ana fikir

Her hareketi $q(t) = q_0 + s(t/T)\,\Delta q$ olarak yaz; burada $s(u)$ bir zaman ölçekleme fonksiyonu, $u \in [0,1]$, $s(0) = 0$, $s(1) = 1$ (Lynch & Park, bölüm 9).

| şekil | $s(u)$ | sınır koşulları | tepe $|\ddot q|$ | jerk |
|---|---|---|---|---|
| step | 1 | yok | ∞ | ∞ |
| linear | $u$ | konum | ∞ (hız sıçrar) | ∞ |
| trapezoidal hız | $T/3$ hızlan, sabit git, $T/3$ yavaşla | konum, hız | $4{,}5\,\Delta q/T^2$ | ∞ (ivme sıçrar) |
| cubic | $3u^2 - 2u^3$ | konum, hız | $6\,\Delta q/T^2$ | uçlarda ∞ |
| quintic | $10u^3 - 15u^4 + 6u^5$ | konum, hız, ivme | $5{,}77\,\Delta q/T^2$ | $60\,\Delta q/T^3$ |

İki uçta hızı ve ivmesi sıfır olan quintic, **minimum-jerk** trajectory'sidir: iki durağan durum arasındaki tüm yumuşak yollar içinde $\int \dddot q^{\,2} dt$'yi en küçük yapan odur (Flash & Hogan 1985; onu insan kol hareketini modellemek için kullandılar). Bütün ivmeler $1/T^2$ ile, jerk'ler $1/T^3$ ile ölçeklenir: süreyi yarıya indirirsen dört kat torque gerekir.

**Joint space mi, Kartezyen mi?** Joint space'te interpolasyon ucuzdur ve iki uç joint limit'lerinin içindeyse yol da içinde kalır, ama el bir eğri çizer. Kartezyen düz çizgi her ara noktada IK ister (ders 10), singularity'lerden ya da joint limit'lerinden geçebilir ve daha fazla joint hareketine mal olur.

**Feedforward.** $\tau = g(q) + K_p(q_r - q) + K_d(\dot q_r - \dot q)$ biçimindeki bir takip controller'ı, uzuvları hızlandıracak torque'u üretmek için önce hata yapmak zorundadır. Model terimi $M(q)\ddot q_r$'yi eklemek (ders 11) bu torque'u önceden sağlar; yumuşak bir referans $\ddot q_r$'yi elde edilebilir ve sınırlı kılar.

## Deney

**Kurulum.** Panda (Menagerie), `home` keyframe'inden başlıyor, joint hareketi $\Delta q$ = (0,6, 0,4, −0,4, 0,5, 0,5, −0,4, 0,6) rad. Modelin yerleşik position actuator'ları kapatıldı (`actuator_gainprm` ve `actuator_biasprm` 0 yapıldı); torque'lar `qfrc_applied`'a yazılıyor ve modelin sınırlarında kırpılıyor (joint 1–4 için 87 N·m, joint 5–7 için 12 N·m). Controller ve fizik 2 ms'de çalışıyor:

```python
s, sd, sdd = profile(kind, d.time, T)
q_ref, qd_ref = Q0 + s * DQ, sd * DQ
tau = d.qfrc_bias[:7] + KP * (q_ref - q) + KD * (qd_ref - qd)    # PD + gravity/Coriolis
tau = np.clip(tau, -lim, lim)
```

Kp = (600, 600, 600, 600, 250, 150, 50), Kd = (50, 50, 50, 50, 20, 12, 4). "Robot" sütunları simülasyonda `qacc`'den ölçülüyor; robot jerk'ü onun 2 ms'deki sonlu farkı. Tam script: [`experiments/13-trajectory-generation/experiment.py`](../../experiments/13-trajectory-generation/experiment.py). MuJoCo 3.17.0, NumPy 2.5.4, Python 3.14, Apple M5, deterministik, ~4 s.

**Tahmin et.** Hangi şeklin takip hatası en küçük? Hangisinin jerk'ü en küçük?

**A. Beş şekil, T = 1 s** (referans değerleri en büyük joint için, 0,6 rad; hata = hareket boyunca en büyük $|q_r - q|$):

| şekil | ref. tepe ivme [rad/s²] | ref. tepe jerk [rad/s³] | robot tepe ivme | robot tepe jerk | takip hatası | saturation'daki adımlar |
|---|---|---|---|---|---|---|
| step | ∞ | ∞ | 739,2 | 175025 | 34,35° | %5,5 |
| linear | ∞ | ∞ | 220,9 | 110475 | 0,78° | %0,0 |
| trapezoid | 2,70 | ∞ | 3,0 | 1003 | 1,08° | %0,0 |
| cubic | 3,59 | ∞ | 3,6 | 1337 | 1,05° | %0,0 |
| quintic | 3,46 | 35,78 | 3,7 | 35 | 1,32° | %0,0 |
| quintic + $M\ddot q_r$ − passive | 3,46 | 35,78 | 3,5 | 48 | **0,01°** | %0,0 |

- **Step**, adımların %5,5'inde torque'u saturation'a sokuyor ve kendi referansının 34,35° gerisinde kalıyor; robotun ivmesi 739,2 rad/s²'ye çıkıyor.
- **Jerk süreklilikle gelir.** İvmesi sıçrayan her şekil (linear, trapezoid, cubic) 1003–110475 rad/s³ robot jerk'ü üretiyor; quintic 35 rad/s³ üretiyor ve referansıyla (35,78) uyuşuyor. Esnek uzuvları, dişli kutularını ve gripper'daki nesneyi titreten şey jerk'tür.
- **Yumuşak şekillerin takip hatası yaklaşık 1° ve onu şekil belirlemiyor.** En kötü joint, modelin pasif damping'inin (1 N·m·s/rad) ve ataletin karşılanmadığı joint 7 (Kp 50). Model feedforward'ı $M\ddot q_r$'yi eklemek ve `qfrc_passive`'i iptal etmek hatayı 0,01°'ye düşürüyor. Jerk'ü şekil belirler; takip hatasını model belirler.

**B. Süre sweep'i** (aynı hareket):

| T [s] | şekil | ref. tepe ivme | ref. tepe jerk | robot tepe jerk | hata |
|---|---|---|---|---|---|
| 0,25 | cubic | 56,68 | ∞ | 21311 | 4,46° |
| 0,25 | quintic | 55,41 | 2249,0 | 2107 | 5,52° |
| 0,50 | cubic | 14,28 | ∞ | 5317 | 2,16° |
| 0,50 | quintic | 13,86 | 284,6 | 234 | 2,73° |
| 1,00 | cubic | 3,59 | ∞ | 1337 | 1,05° |
| 1,00 | quintic | 3,46 | 35,8 | 35 | 1,32° |
| 2,00 | cubic | 0,90 | ∞ | 335 | 0,52° |
| 2,00 | quintic | 0,87 | 4,5 | 4 | 0,65° |

İvme $1/T^2$ ile ölçekleniyor (56,68 → 14,28 → 3,59 → 0,90, her yarılanmada 4 kat), quintic'in jerk'ü $1/T^3$ ile (2249,0 → 284,6 → 35,8 → 4,5, 8 kat), yalnız PD ile takip hatası ise kabaca $1/T$ ile. Her sürede cubic'in robot jerk'ü quintic'inkinin bir mertebe ya da daha fazla üstünde (0,25 s'de 21311'e karşı 2107 rad/s³, 2 s'de 335'e karşı 4).

**C. Joint space mi, Kartezyen mi?** Tool centre point, sabit oryantasyonla $y$ boyunca 50 cm ilerliyor; uçlar A ve B IK çözümleri (IK kalıntısı 0,001 mm'nin altında):

| interpolasyon | çizgiden en büyük uzaklık | el yolu | joint yolu |
|---|---|---|---|
| joint space ($q$'da quintic) | 5,38 cm | 51,53 cm | 1,089 rad |
| Kartezyen (her ara noktada IK) | 0,00 cm | 50,00 cm | 1,256 rad |

Aynı uçlar, aynı zamanlama: joint space yolu çizgiden 5,38 cm sapıyor. Kalabalık bir sahnede bir kutunun kenarına çarpmaya yeter; boş alanda ise önemsiz, orada joint space daha ucuz (1,089'a karşı 1,256 rad joint hareketi) ve yol ortasında asla singularity'ye girmez.

## Boz

Hareketi kısalt. T = 0,25 s'de referans, 0,6 rad'lık joint hareketi için 55,41 rad/s² (quintic) istiyor; yumuşak quintic bile artık 2107 rad/s³ robot jerk'ü ve 5,52° takip hatası üretiyor; daha kısa hareketlerde referans ivmesi $1/T^2$, jerk $1/T^3$, PD takip hatası kabaca $1/T$ ile büyüyor. Uca git, T → 0, step'i elde edersin: torque saturation (adımların %5,5'i), 739,2 rad/s² ivme ve 34,35° gecikme. Mekanizma controller değil, ölçekleme yasası: hiçbir gain ayarı actuator'ların veremeyeceği ivmeyi üretemez ve torque kırpıldığında PD'nin özenle ayarlanmış damping'i ortadan kalkar.

## Gerçek robotta

- **Panda / FR3 sınıfı.** libfranka'nın rate limiter sabitleri (10 rad/s² joint ivmesi, 5000 rad/s³ jerk, 1 ms periyot; [`rate_limiting.h`](https://github.com/frankarobotics/libfranka/blob/main/include/franka/rate_limiting.h)) sıçrayan her referans tarafından ihlal edilir. Bizim sweep'imizde T = 0,5 s'de hem cubic hem quintic 10 rad/s²'yi aşıyor (14,28 ve 13,86) ve cubic'in ivme süreksizliği her sürede sınırsız jerk veriyor. 1 s'lik bir quintic iki sınırın da içinde kalıyor (3,46 rad/s², 35,78 rad/s³). Bu karşılaştırmalar simüle ettiğimiz referansları kullanıyor; burada hiçbir şey bir Panda'da koşturulmadı.
- **Go2.** SDK ayağa kalkma örneğinin parça parça doğrusal rampası (ders 12) "linear" satırıdır: konum adımları sınırlı, ama her ara noktada hız sıçrıyor.
- **SO-101.** Servolar hedef konumları kontrol döngüsünün hızında alır. Ham hedefler üreten 30 Hz'lik bir policy fiilen bir merdiven gönderiyor; hedefler arasında interpolasyonun gerçek kolda jerk'ü azaltıp azaltmadığı henüz ölçülmedi.

## Kendini dene

1. **Kolay.** Bir quintic bir joint'i 2 s'de 1 rad hareket ettiriyor. Tepe ivmesi ve tepe jerk'ü nedir?
2. **Orta.** A bölümünde linear profil, referans ivmesi sonsuz olduğu hâlde neden en küçük takip hatasına (0,78°) sahip?
3. **Zor.** `experiment.py`'de cubic ve trapezoid'e $M(q)\ddot q_r - $ `qfrc_passive` feedforward'ı ekle. Koşturmadan önce tahmin et: onların takip hatası da ~0,01°'ye düşer mi? Robot jerk'leri değişir mi?

### Cevaplar

1. Tepe ivme $5{,}77 \cdot 1 / 2^2 \approx 1{,}44$ rad/s²; tepe jerk $60 \cdot 1 / 2^3 = 7{,}5$ rad/s³ (tablodaki formüllerden).
2. Tepe hızı ($\Delta q/T$) tüm şekillerin en düşüğü ve ivmesi iki an dışında sıfır; bu, modellenmemiş damping torque'unun en küçük olmasıyla tutarlı. Sonsuz ivme yalnızca başta ve sonda iki kısa sıçrama olarak görünüyor (robot tepesi 220,9 rad/s²).
3. Burada hesaplanmadı; koştur.

## Akılda kalanlar ve kaynaklar

- Zaman ölçekleme fiziği belirler: ivmeler $1/T^2$, jerk'ler $1/T^3$ ile ölçeklenir; bir step Panda'nın torque'unu saturation'a sokar ve 34,35° geride kalır.
- Jerk'ü ivmenin sürekliliği belirler: T = 1 s'de quintic 35 rad/s³, trapezoid ve cubic 1003–1337 rad/s³. Takip hatasını model belirler: feedforward onu 1,32°'den 0,01°'ye indirir.
- Joint space interpolasyonu düz çizgiden sapar (burada 50 cm'de 5,38 cm); Kartezyen interpolasyonu yalnızca yolun önemli olduğu yerde kullan.

Sonraki: [ders 14 · Operational-space ve impedance control](14-impedance-control.md).

Kaynaklar:
- Lynch & Park, [*Modern Robotics*](https://hades.mech.northwestern.edu/index.php/Modern_Robotics), bölüm 9 (Trajectory Generation).
- T. Flash, N. Hogan, "The coordination of arm movements: an experimentally confirmed mathematical model", *Journal of Neuroscience* 5(7):1688–1703, 1985, [doi:10.1523/JNEUROSCI.05-07-01688.1985](https://doi.org/10.1523/JNEUROSCI.05-07-01688.1985).
- libfranka [`rate_limiting.h`](https://github.com/frankarobotics/libfranka/blob/main/include/franka/rate_limiting.h) (main branch, 27.10.2026'da okundu); [`go2_stand_example.py`](https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/example/go2/low_level/go2_stand_example.py).
- [MuJoCo Menagerie · `franka_emika_panda`](https://github.com/google-deepmind/mujoco_menagerie/tree/f3b6013/franka_emika_panda).
- Ölçülen bütün sayılar: [`experiments/13-trajectory-generation/output.txt`](../../experiments/13-trajectory-generation/output.txt).
