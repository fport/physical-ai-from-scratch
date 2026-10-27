# 38 · System identification

<!-- meta -->
| | |
|---|---|
| **Bölüm** | [9 · Sim-to-real](README.md) |
| **Çalışma türü** | CPU · Mac ya da Colab CPU |
| **Önkoşul** | [11](../part-4-dynamics-and-control/11-rigid-body-dynamics.md), [37](37-latency.md) |
| **Önce öğren** | [Optimization](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/08-optimization) |
| **Süre** | ~60 dk |
| **Kod** | [`experiments/38-system-identification/experiment.py`](../../experiments/38-system-identification/experiment.py) |
| **Durum** | ✅ tam |
<!-- /meta -->

> Simülatör robot hakkında bir hipotezdir; system identification bu hipotezi log'lanmış veriyle sınamanın ve sayılarını düzeltmenin yoludur.

## Problem

Menagerie SO-101 her STS3215 joint'ine aynı `damping="0.60" frictionloss="0.052" armature="0.028"` değerlerini ve `kp="998.22"` olan bir position actuator'ü veriyor; dosyanın kendi yorumu bu gain'lerin LeRobot'un kullandığı servo gain'lerinin "1-to-1" karşılığı olmadığını söylüyor ([`so101.xml`](https://github.com/google-deepmind/mujoco_menagerie/blob/f3b6013/robotstudio_so101/so101.xml)). Senin kolunun kendi baskısı, kendi yükü, kendi kablo sürtünmesi ve kendi bus gecikmesi var. Ders 37, eksik 30 ms'nin iyi bir döngüyü kötüye çevirebildiğini gösterdi. Modelin hangi sayıları senin kolun için yanlış ve bunları yalnızca log'lanmış trajectory'lerden düzeltebilir misin?

## Ana fikir

**Parametrik model.** Simülatörün yapısını koru, bir parametre vektörü $\theta$ aç ve onu, *log'lanmış komutlara* verilen simüle tepki *log'lanmış ölçümlere* uyacak şekilde fit et (output-error ya da simulation-error kriteri):

$$\hat\theta = \arg\min_\theta \sum_k \big\| y_k - \hat y_k(\theta; u_{0:k}) \big\|^2 .$$

Model bir kara kutuysa (bir MuJoCo rollout'u), bunu finite-difference Jacobian'larla bir nonlinear least-squares yöntemiyle çöz (`scipy.optimize.least_squares`; bkz. AIEFS: [Optimization](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/08-optimization)). Tork ölçümü olan torque-controlled kollarda rigid-body dynamics eylemsizlik parametrelerine göre doğrusaldır; bu yüzden $\tau = Y(q,\dot q,\ddot q)\,\pi$ regressor'ü üzerinde sıradan least squares klasik yoldur (Khalil & Dombre). Hobi sınıfı bir position servo kalibre edilmiş joint torku raporlamaz; burada model simülatörün kendisi.

**Bir kolda ne önemli** (MuJoCo adlarıyla):

| Parametre | MuJoCo alanı | Nasıl görünür |
|---|---|---|
| Link kütleleri / eylemsizlikleri | `body_mass`, `body_inertia` | yerçekimi sarkması, daha yavaş ivmelenme |
| Armature | `dof_armature` | dişli rotorun ek eylemsizliği |
| Viskoz damping | `dof_damping` | hızla orantılı sürüklenme |
| Friction loss | `dof_frictionloss` | ölü bant, stick-slip, yön değişiminde ofset |
| Actuator gain'i | `actuator_gainprm`, `actuator_biasprm` | servo'nun stiffness'ı |
| Tork limiti | `actuator_forcerange` | maksimum ivme |
| Gecikme | (modelin dışında) | komut ile hareket arasındaki faz kayması |

**Excitation.** Bir parametre ancak ölçülen sinyali gürültüden daha fazla değiştiriyorsa identify edilebilir. Giriş *persistently exciting* olmalı (Ljung): frekansça zengin ve genliği yeterince büyük, ama modelin geçerli olduğu ve robotun güvende olduğu bölgenin içinde.

## Deney

**Kurulum, sim-to-sim.** "Gerçek" bir SO-101, altı gizli parametresi değiştirilmiş Menagerie modeli (bu test için uyduruldu). Nominal bir model bunları log'lardan fit ediyor.

| | kütle ölçeği | armature | damping | frictionloss | kp ölçeği | gecikme |
|---|---|---|---|---|---|---|
| nominal | 1,000 | 0,028 | 0,600 | 0,052 | 1,000 | 0 ms |
| gizli "gerçek" | 1,200 | 0,040 | 0,900 | 0,100 | 0,700 | 15 ms |

Beş kol joint'i bir orta pozun etrafında 4 s boyunca bir multisine ile (0,5–3,5 Hz, her joint'e farklı fazlar) sürülüyor; pozisyonlar beyaz gürültüyle 100 Hz'de log'lanıyor; fizik 5 ms. Fit log üzerinde çalışıyor; farklı, ayrılmış bir trajectory (yumuşatılmış rastgele hedefler) tahmini kontrol ediyor. Hiç fit yapmadan önce, validation trajectory'sinde nominal ile gerçek arası: 9,54 mrad RMS.

```python
def resid(x):                                    # x: log-ratios to nominal, delay in 10 ms units
    return 1e3 * (simulate(to_theta(x), cmd) - q_log).ravel()     # mrad
r = least_squares(resid, x0, bounds=(lb, ub), diff_step=1e-3)
```

Tam script: [`experiments/38-system-identification/experiment.py`](../../experiments/38-system-identification/experiment.py). MuJoCo 3.17.0, NumPy 2.5.4, SciPy 1.18.1, Python 3.14, Apple Silicon Mac (Apple M5), deterministik, bir dakikadan çok daha az CPU.

**A. Excitation genliğinin sweep'i** (gürültü σ = 2 mrad). Değerler: fit sonrası parametre hatası, gerçek değerin yüzdesi olarak (gecikme: ms cinsinden hata); validation RMS mrad cinsinden; `sat` = bir servo'nun 2,94 N·m limitinde olduğu adımların payı. *Tahmin et:* hangi genlik en iyi identify eder?

| genlik | kütle | armature | damping | sürtünme | kp | gecikme [ms] | fit sonrası val. [mrad] | sat |
|---|---|---|---|---|---|---|---|---|
| 0,005 rad | −%26,7 | +%5,3 | −%61,1 | +%21,5 | −%24,6 | −4,7 | 7,33 | %0,1 |
| 0,020 rad | +%18,1 | −%2,7 | −%41,2 | +%100,5 | +%6,8 | −0,1 | 4,32 | %1,0 |
| 0,050 rad | −%5,8 | +%1,7 | −%21,9 | −%25,6 | −%6,2 | −0,1 | 2,49 | %2,4 |
| **0,100 rad** | −%4,1 | −%0,5 | +%0,1 | +%16,5 | +%1,4 | −0,0 | **0,09** | %4,6 |
| 0,200 rad | −%0,2 | +%1,5 | +%2,1 | −%48,2 | −%8,1 | −0,5 | 0,13 | %41,0 |
| 0,400 rad | −%6,9 | −%23,4 | −%35,4 | −%68,3 | +%13,3 | −3,7 | 6,67 | %100,0 |

Çok küçükse gürültü baskın: 0,005 rad'da 2 mrad'lık gürültü excitation genliğinin %40'ı; damping, sürtünme ve kütlenin etkisi bu küçük sinyalin küçük bir parçası. Çok büyükse bir servo hep tork limitinde: servo, gain'i ve damping'i pek önemli olmayan bir tork kaynağına dönüşüyor ve fit validation'da başarısız (6,67 mrad, neredeyse nominal kadar kötü). Arada (0,1–0,2 rad) validation hatası 9,54'ten 0,09–0,13 mrad'a düşüyor. 0,02 ile 0,2 rad arasında gecikme 0,5 ms içinde bulunuyor; en kolay parametre o, çünkü her örneği kaydırıyor.

**B. Gürültünün sweep'i**, 0,1 rad'da:

| σ | kütle | armature | damping | sürtünme | kp | gecikme [ms] | fit sonrası val. [mrad] |
|---|---|---|---|---|---|---|---|
| 0 | −%0,0 | +%0,0 | −%0,0 | +%0,0 | −%0,0 | −0,0 | 0,00 |
| 2 mrad | +%6,9 | −%1,3 | +%0,9 | −%12,9 | +%4,6 | +0,2 | 0,25 |
| 10 mrad | +%27,8 | −%4,3 | −%0,5 | +%120,4 | +%19,3 | +0,8 | 0,76 |
| 30 mrad | −%81,4 | −%6,1 | −%0,7 | +%131,3 | −%13,4 | +0,4 | 1,49 |

Gürültüsüz fit altı parametrenin hepsini tam olarak buluyor. Gürültüyle armature, damping ve gecikme %6,1 / 1 ms içinde kalıyor, kütle ve sürtünme ise −%81,4 ve +%131,3'e kadar savruluyor; buna rağmen validation hatası 1,49 mrad'da kalıyor, nominalden altı kat iyi. Parametre hatası ile tahmin hatası farklı şeyler: bazı parametre kombinasyonları çıktıyı neredeyse hiç değiştirmiyor ("sloppy" bir yön); veri onları sabitleyemiyor ve tahmin için bu pek önemli değil. A ve B'deki 2 mrad satırları aynı ayar, farklı gürültü çekilişleri (0,09 ve 0,25 mrad): tek bir gürültü gerçekleşmesi tek bir örnektir.

## Boz

0,1 rad'lık log'u yanlış model yapısıyla fit et: gecikme parametresi yok (gecikme 0'a sabit), gürültü yok.

| | kütle | armature | damping | sürtünme | kp | fit sonrası val. [mrad] |
|---|---|---|---|---|---|---|
| gecikme terimi yok | −%50,7 | +%35,4 | +%40,9 | +%46,0 | −%67,8 | 6,23 |

Fit değerleri: kp ölçeği 0,226 (gerçek 0,700), damping 1,268 (gerçek 0,900), kütle ölçeği 0,591 (gerçek 1,200). Optimizer 15 ms gecikmeyi temsil edemiyor, bu yüzden gecikmeyi yumuşak, aşırı sönümlü, hafif bir servo ile taklit ediyor. Fit kendi verisinde "iyi", validation'da 6,23 mrad; bu, gecikme terimiyle 30 mrad gürültüde elde edilenden kötü. Eksik bir mekanizma tek bir yanlış parametre olarak görünmez; hepsini bozar. Gecikmeyi her zaman dahil et (ders 37) ve fit etmediğin veride kontrol et.

## Gerçek robotta

Gerçek SO-101 için prosedür (gerçek kolda henüz ölçülmedi):

1. **Calibrate et ve bağlan**, ders 40'taki gibi. LeRobot joint pozisyonlarını varsayılan olarak kalibre edilmiş aralığın ortasına göre derece cinsinden raporluyor (`use_degrees = True`; [`config_so_follower.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/config_so_follower.py), [`motors_bus.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/motors/motors_bus.py)). Bu dereceler ile Menagerie joint açıları arasındaki ofset ve işaret bilinmiyor: kolu bilinen iki poza koyup önce bunları fit et ya da $\theta$'ya ekle.
2. **Bir komut akışını log'la**: Python API'si ile (`robot.send_action`, `robot.get_observation`, [LeRobot rehberi](https://huggingface.co/docs/lerobot/il_robots)) sabit hızlı bir döngüde, her komutun ve her okumanın zaman damgasını kaydederek. Bus'ın izin verdiği döngü hızı burada ölçülmedi.
3. **Önce her joint'i tek tek uyar**, sonra hepsini birlikte: büyüyen step'ler ve bir chirp ya da multisine; birkaç dereceden başla, kalibre edilmiş aralığın içinde kal, masadan uzak dur, ders 15'in safety katmanları açık olsun. Gürültüyü görmek için her koşuyu tekrarla.
4. **Fit et**: aynı altı parametreyi (artı ofsetleri), log'lanmış komutları log'lanmış zaman damgalarıyla MuJoCo'da tekrar oynatarak, `experiment.py`'deki gibi. Menagerie gain'leri servo P gain'inin 16 olduğu varsayılarak türetilmiş (`so101.xml` içindeki yorum); bu, LeRobot'un varsayılan `position_p_coefficient = 16` değeriyle uyuşuyor.
5. **Doğrula**: ayrılmış bir koşuda ve gerçek bir görev hareketinde. Fit yalnızca kendi verisinde çalışıyorsa bir mekanizma eksiktir (Menagerie dosyasının tanımladığı ama kullanmadığı backlash; kablo kuvvetleri; sıcaklığa bağlı sürtünme).

## Kendini dene

1. **Kolay.** Gecikme 0,02 rad'da neden iyi identify ediliyor da damping edilmiyor?
2. **Orta.** B bölümünde σ = 30 mrad'da kütle ölçeği −%81,4 hatalı, ama validation hatası yalnızca 1,49 mrad. Kütle bu kolun joint pozisyonları için neden bu kadar az önemli?
3. **Zor.** Yedinci parametre olarak `forcerange`'i (tork limiti) ekle ve A bölümünü tekrarla. 0,4 rad satırının iyileşip iyileşmeyeceğini tahmin et.

### Cevaplar

1. Gecikme bütün tepkiyi zamanda kaydırır; bu yüzden hızı sıfır olmayan her örnek onun hakkında bilgi taşır. Damping ise tepkiyi yalnızca hızla orantılı bir kuvvetle değiştirir; küçük genlikte bu etki gürültüden küçüktür.
2. Sert servo kolu yerçekimine karşı çok küçük bir sarkmayla tutuyor: ders 12, çıplak kol için 0,033° omuz sarkması ölçtü. Kütle o zaman yalnızca hızlı hareketlerde eylemsizlik yoluyla, fit'in iyi identify ettiği armature ile birlikte etki ediyor. Kütleyi değiştirmek pozisyonları çok az değiştiriyor, bu yüzden gürültü onu gizliyor; aynı sebeple tahmini de pek etkilemiyor.
3. Validation hatası 6,67'den 0,82 mrad'a iyileşiyor, parametreler ise iyileşmiyor: fit edilen tork limiti ölçeği 0,820 (gerçek 1,000), kp hatası −%50,0. Saturation'lı veri limiti sabitler, servo gain'ini değil; ek parametre diğerlerinin hatasını emiyor. A bölümüyle aynı gürültü çekilişi; `output.txt`'de "Exercise 3" bölümü.

## Akılda kalanlar ve kaynaklar

- Simülatörü log'lanmış komutlara ve ölçümlere fit et (output error); iyi bir excitation ile validation hatası 9,54'ten 0,09 mrad'a düşüyor.
- Excitation'ın bir penceresi var: çok küçüğü gürültüde boğulur, çok büyüğü actuator'ü saturation'a sokar ve servo parametrelerini gizler.
- Parametre hatası tahmin hatası değildir ve eksik bir mekanizma (gecikme) diğer bütün parametreleri saptırır: ayrılmış veride doğrula.

Sonraki: [ders 39 · Domain randomization](39-domain-randomization.md).

Kaynaklar:
- L. Ljung, *System Identification: Theory for the User*, 2. baskı, Prentice Hall, 1999.
- W. Khalil, E. Dombre, *Modeling, Identification and Control of Robots*, Hermes Penton, 2002.
- [SciPy `least_squares`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html); [MuJoCo XML reference · joint](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-joint) (armature, damping, frictionloss).
- [MuJoCo Menagerie · robotstudio_so101](https://github.com/google-deepmind/mujoco_menagerie/tree/f3b6013/robotstudio_so101) (Apache-2.0).
- LeRobot [`config_so_follower.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/config_so_follower.py), [`motors_bus.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/motors/motors_bus.py), [imitation learning rehberi](https://huggingface.co/docs/lerobot/il_robots) (2026-10-27'de okundu).
- J. Tan ve ark., [Sim-to-Real: Learning Agile Locomotion For Quadruped Robots](https://arxiv.org/abs/1804.10332), 2018 (aktarım için system identification, actuator modeli ve latency).
- Ölçülen bütün sayılar: [`experiments/38-system-identification/output.txt`](../../experiments/38-system-identification/output.txt).
