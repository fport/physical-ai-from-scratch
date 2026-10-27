# 16 · Grasping ve scripted pick-and-place

<!-- meta -->
| | |
|---|---|
| **Bölüm** | [4 · Dynamics ve control](README.md) |
| **Çalışma türü** | CPU · Mac ya da Colab CPU |
| **Önkoşul** | [10](../part-3-kinematics/10-inverse-kinematics.md), [13](13-trajectory-generation.md) |
| **Önce öğren** | yok |
| **Süre** | ~60 dk |
| **Kod** | [`experiments/16-grasping-and-pick-place/experiment.py`](../../experiments/16-grasping-and-pick-place/experiment.py) |
| **Durum** | ✅ tam |
<!-- /meta -->

> Bir grasp, yükü sürtünme taşıyabildiğinde tutar. Pick-and-place'teki geri kalan her şey (IK, trajectory'ler, state machine) iki çeneyi bunun doğru olduğu yere koymak için vardır.

## Problem

Bölüm 7 policy'leri demonstration'lardan eğitiyor ve en ucuz demonstration'lar zaten çalışan bir script'ten gelir. Simülasyonda scripted bir pick'i taklit etmek kolaydır: çeneler kapanınca küpü gripper'a weld et, her deneme "başarılı" olsun. Bu dersin dürüstçe cevapladığı soru fiziksel olan: gerçek bir contact modeliyle ve hiçbir bağlantı olmadan, bir SO-101 3 cm'lik bir küpü ne zaman gerçekten kaldırır, ne zaman sıkıştırıp fırlatır, düşürür ya da hiç kapanamaz? Cevap [ders 04](../part-1-simulation/04-friction.md)'ün friction cone'undan geliyor ve 240 simüle edilmiş grasp üzerinde ölçülüyor.

## Ana fikir

**Gripper geometrisi.** SO-101 gripper'ında sabit bir çene ve bir menteşe üzerinde hareketli bir çene var (`gripper` joint'i, Menagerie modelinde aralık −0,17 ile 1,75 rad), bir STS3215 position servo ile sürülüyor. 3 cm'lik bir küp üzerinde kapalı limiti (−0,17 rad) komutlamak, servo'nun torque cap'ine ulaşana kadar itmesine yol açar; sıkma kuvvetini o cap ve çene kolu belirler.

**Antipodal grasp.** Normalleri zıt olan ve contact noktalarını birleştiren doğru üzerinde yatan iki contact. Küpün iki paralel yüzüne yukarıdan yapılan bir grasp'te çeneler $N_1, N_2$ normal kuvvetleriyle yatay bastırır ve ağırlığı tamamen sürtünme taşımalıdır (Lynch & Park, bölüm 12):

$$\mu\,(N_1 + N_2) \;\ge\; m g \qquad\Longleftrightarrow\qquad \frac{\mu (N_1 + N_2)}{m g} \ge 1$$

Bu friction-cone koşuludur: her contact kuvveti yarı açısı $\arctan\mu$ olan cone'unun içinde kalmalı. MuJoCo bir contact'ta `priority`'si yüksek olan geom'un sürtünmesini kullanır (aksi halde eleman bazında maksimumu) ([Modeling](https://mujoco.readthedocs.io/en/stable/modeling.html)); SO-101'in çene pad'lerinin priority'si 1 ve sürtünmesi 1, bu yüzden küpe priority 2 veriliyor ki geçerli olan onun μ'sü olsun.

**State machine.** Her state'in bir giriş eylemi ve bir çıkış koşulu var:

| state | kol hedefi | çene | çıkış koşulu |
|---|---|---|---|
| settle | home | açık | 0,3 s |
| pre | küp merkezinin 5,5 cm üstüne min-jerk | açık | 1,2 s |
| down | grasp yüksekliğine min-jerk (merkezin 4 mm altı) | açık | 0,8 s |
| close | tut | kapalıya rampa (0,3 s) | iki çene de küpe değiyor ve çene hızı 0,1 s boyunca < 0,05 rad/s (timeout 4 s) |
| lift | merkezin 7 cm üstüne min-jerk | kapalı | 1,0 s |
| hold | tut | kapalı | 0,8 s |

Kol hedefleri 5 kol joint'i üzerinde damped-least-squares IK'dan geliyor (gripper frame hedefte, yaklaşma ekseni aşağıda, çeneler x boyunca kapanıyor; ders 10) ve ders 13'ün quintic'iyle harmanlanıyor. `close` çıkışı bir timer değil, algılanan bir koşul: çenenin ne kadar süreye ihtiyacı olduğu torque cap'ine bağlı (B bölümü).

**Başarı, fiziksel olarak.** Sonda: küpün altı yerden ≥ 4 cm yukarıda **ve** iki çeneyle de contact var.

## Deney

Pick, Bölüm 7'nin demonstration kaydetmek için import edebileceği tek bir fonksiyon:

```python
def scripted_pick(model, data, cube_pos, record=False):
    """Runs settle -> pre -> down -> close -> lift -> hold in (model, data).
    Returns success, lift_cm, squeeze_N, close_s, ... and with record=True the
    per-step (time, state, qpos, ctrl) trajectory."""
```

Kurulum: SO-101 (Menagerie `scene.xml`, timestep 2 ms), 3 cm küp, `condim` 4, küp sürtünmesi μ, küp kütlesi m, çene torque cap'i τ (varsayılan 2,94 N·m). Her sweep değeri (22, 0) cm çevresindeki 2 × 2 cm'lik bir karede düzgün dağılmış 10 küp pozisyonuyla koşuluyor, seed 0. Script'in tamamı: [`experiments/16-grasping-and-pick-place/experiment.py`](../../experiments/16-grasping-and-pick-place/experiment.py). MuJoCo 3.17.0, NumPy 2.5.4, Python 3.14, Apple M5, deterministik, ~30 s.

Nominal pick (μ 1,0, 30 g, varsayılan cap): başarılı, küpün altı 7,37 cm'de, IK hatası 0,18 mm, çene normal kuvvetleri 39,84 / 39,83 N, kaydedilen 2287 adım. Varsayılan cap'le sıkma 0,29 N'luk bir küp üzerinde yaklaşık 80 N: sürtünme asla sınır değil. Cone'u görmek için A ve C bölümleri 0,5 N·m'lik bir cap kullanıyor.

*Tahmin et:* τ = 0,5 N·m'de sıkma yaklaşık 13 N, yani 30 g'lık bir küp için cone koşulu μ ≈ 0,02'nin altında bozulur. Ölçülen sınır orada mı çıkacak?

**A. Sürtünme sweep'i** (30 g, τ = 0,5 N·m). "close" = close state'inde geçen süre, "sıkma" = state bittiğindeki $N_1 + N_2$ (çenenin kapandığı denemeler üzerinden ortalama).

| μ | başarı | Wilson %95 | close | sıkma | $\mu(N_1+N_2)/mg$ | tuttu / kapanmadı / düşürdü; fırladı |
|---|---|---|---|---|---|---|
| 0,005 | 0/10 | [0,00, 0,28] | – | – | – | 0 / 10 / 0; 10 |
| 0,01 | 0/10 | [0,00, 0,28] | – | – | – | 0 / 10 / 0; 10 |
| 0,015 | 0/10 | [0,00, 0,28] | – | – | – | 0 / 10 / 0; 10 |
| 0,02 | 0/10 | [0,00, 0,28] | 1,59 s | 14,21 N | 0,97 | 0 / 9 / 1; 8 |
| 0,03 | 2/10 | [0,06, 0,51] | 1,62 | 12,29 | 1,25 | 2 / 8 / 0; 7 |
| 0,05 | 10/10 | [0,72, 1,00] | 1,63 | 12,74 | 2,16 | 10 / 0 / 0; 0 |
| 0,1 | 10/10 | [0,72, 1,00] | 1,63 | 13,26 | 4,51 | 10 / 0 / 0; 0 |
| 0,3 | 10/10 | [0,72, 1,00] | 1,58 | 14,24 | 14,52 | 10 / 0 / 0; 0 |
| 1,0 | 10/10 | [0,72, 1,00] | 1,56 | 14,91 | 50,65 | 10 / 0 / 0; 0 |

Sınır μ = 0,03 ile 0,05 arasında, tahmin edilen değere yakın; ama altındaki başarısızlıklar kaldırmadan sonraki düşürmeler **değil**. μ ≤ 0,03'teki 48 başarısız denemenin 47'sinde çene küp üzerine hiç oturmadı: küp çeneler kapanırken onları terk etti ve 45'inde 30 cm'den daha uzakta kaldı ("fırladı"). Düşük sürtünme, ağırlık önem kazanmadan, sıkma anında başarısız oluyor. Kapanma sırasındaki contact geometrisini ölçmedik; büyük fırlama mesafeleri MuJoCo'nun çok düşük μ'deki soft contact'ının bir özelliği ve gerçek hızlar olarak okunmamalı.

**B. Çene torque cap'i** (30 g, μ = 0,05):

| τ [N·m] | başarı | close | sıkma | oran | tuttu / kapanmadı / düşürdü; fırladı |
|---|---|---|---|---|---|
| 2,94 | 7/10 | 0,96 s | 88,93 N | 15,11 | 7 / 1 / 2; 3 |
| 1,0 | 10/10 | 0,91 | 27,23 | 4,63 | 10 / 0 / 0; 0 |
| 0,5 | 10/10 | 1,63 | 12,74 | 2,16 | 10 / 0 / 0; 0 |
| 0,3 | 10/10 | 2,65 | 7,04 | 1,20 | 10 / 0 / 0; 0 |
| 0,2 | 0/10 | – | – | – | 0 / 10 / 0; 5 |
| 0,15 | 0/10 | – | – | – | 0 / 10 / 0; 0 |
| 0,1 | 0/10 | – | – | – | 0 / 10 / 0; 0 |

Sıkma cap ile kabaca orantılı artıyor (7,04 → 12,74 → 27,23 N). 0,3 N·m'nin altında çene kendi pasif damping'ine ve friction loss'una (modelde 0,60 N·m·s/rad ve 0,052 N·m) karşı o kadar yavaş kapanıyor ki 4 s'lik timeout içinde oturamıyor (cap düştükçe close süresi 0,91 → 1,63 → 2,65 s). Varsayılan 2,94 N·m'de bu kaygan yüzeyde 10 küpün 3'ü fırlatılıyor: daha fazla kuvvet her zaman daha iyi bir grasp demek değil.

**C. Kütle sweep'i** (μ = 0,3, τ = 0,5 N·m):

| m [kg] | 0,03 | 0,1 | 0,2 | 0,3 | 0,4 | 0,45 | 0,5 | 0,7 |
|---|---|---|---|---|---|---|---|---|
| başarı | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 0/10 | 0/10 | 0/10 |
| sıkma [N] | 14,24 | 14,95 | 15,52 | 15,87 | 16,25 | 16,47 | 16,61 | 16,90 |
| $\mu(N_1+N_2)/mg$ | 14,52 | 4,57 | 2,37 | 1,62 | 1,24 | 1,12 | 1,02 | 0,74 |
| sonuç | tuttu | tuttu | tuttu | tuttu | tuttu | düşürdü | düşürdü | düşürdü |

Burada hikâyenin tamamı cone koşulu: ortalama oranı ≥ 1,24 olan her kütle bütün denemelerde tutuyor, ortalama oranı ≤ 1,12 olan her kütle bütün denemelerde kaldırma sırasında düşüyor (10/10 için Wilson %95: [0,72, 1,00]; 0/10 için: [0,00, 0,28]). Sınır statik değer olan 1'in biraz üstünde; bu, kaldırma sırasında küpü yukarı ivmelendirmek için gereken ek kuvvetle tutarlı.

## Boz

Sabit bir cap'te μ'yü düşür (A bölümü) ya da kütleyi artır (C bölümü); grasp iki farklı şekilde bozulur. Ağır küp, oran 1,12 ya da altında: küp kaldırma sırasında çenelerin arasından aşağı kayıyor, tam friction cone'un öngördüğü gibi. Kaygan küp, μ ≤ 0,015: grasp hiç oluşmuyor; her değerdeki 10 küpün hepsi çene kapanırken dışarı atılıyor. Ölçülen statik oranın 1,25 olduğu μ = 0,03'te bile 10 grasp'ten 8'i hiç oluşmuyor. Son grasp üzerindeki bir friction-cone kontrolü gerekli ama yeterli değil; kapanma hareketi de contact'ları cone'larının içinde tutmalı.

## Gerçek robotta

- **SO-101 / LeRobot.** Bağlanırken LeRobot gripper servo'sunu `Max_Torque_Limit` 500 ("50% of max torque to avoid burnout"), `Protection_Current` 250 ve `Overload_Torque` 25 ile sınırlıyor ([`so_follower.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/so_follower.py)). Bu register değerlerinin çenede kaç newton-metreye karşılık geldiği orada belgelenmemiş; Menagerie'deki 2,94 N·m'lik cap bir model parametresi, ölçüm değil.
- **Gerçek kolda henüz ölçülmedi.** Prosedür: iki farklı yüzey bitişinde 3 cm'lik küpler bas ya da satın al, tart, scripted pick'i (kalibrasyondan gelen joint offset'leriyle) sabit pozisyonlarda küp başına 20 kez koş, başarıyı aynı fiziksel testle kaydet (≥ 4 cm kalktı, 1 s tutuldu) ve k/n'yi Wilson aralığıyla raporla.
- **Sim-to-real farkları.** Gerçek sürtünme statik ve kinetik arasında farklıdır, toz ve aşınmayla değişir; MuJoCo'da tek bir μ var (ders 04). Çene pad'leri deforme olur. Sınırın kaymasını bekle; yöntem (sıkmayı ölç, cone payını hesapla) aynen taşınır.

## Kendini dene

1. **Kolay.** Sıkma $N_1 + N_2$ = 12,74 N ve μ = 0,05 iken cone koşulunun izin verdiği en ağır küp kaç kg?
2. **Orta.** Script `close`'u neden sabit 0,6 s sonra değil de algılanan bir koşulla bitiriyor? B bölümünün close sürelerini kullan.
3. **Zor.** `close` state'i, çene rampası başladığından beri küpün 5 mm'den fazla hareket etmediğini de kontrol etsin; ettiyse iptal etsin (yeniden aç, yeniden yaklaş). A bölümünü yeniden koş. Koşmadan önce tahmin et: μ = 0,03'teki başarı artar mı?

### Cevaplar

1. $m \le \mu (N_1+N_2)/g = 0{,}05 \cdot 12{,}74 / 9{,}81 \approx 0{,}065$ kg.
2. Close süresi 0,91 s (τ 1,0 N·m) ile 2,65 s (τ 0,3 N·m) arasında değişiyor. Sabit 0,6 s ölçülen bütün close sürelerinden kısa; yani her cap'te çene oturmadan kaldırmaya başlar. En yavaş cap'e yetecek uzunlukta bir timer ise hızlı cap'lerde zaman kaybettirir.
3. Burada hesaplanmadı; koş ve gör.

## Akılda kalanlar ve kaynaklar

- Friction cone kaldırmayı öngörüyor: sıkma ölçüldüğünde, oranı ≥ 1,24 olanlar 50/50 denemede tuttu, oranı ≤ 1,12 olanlar 30/30 denemede düşürdü (C bölümü).
- Düşük μ'de grasp daha erken, kapanma sırasında bozuluyor (μ ≤ 0,03'te 48 başarısızlığın 47'si); sıkma çene torque cap'iyle ölçekleniyor ve fazla düşük bir cap 4 s içinde hiç kapanmıyor.
- `scripted_pick(model, data, cube_pos, record=True)` fiziksel olarak doğrulanmış bir demonstration üreticisi: state machine + IK + min-jerk + algılanan bir close koşulu, weld yok.

Sonraki: Bölüm 5 · [ders 17 · Sensor'ler, noise ve filtering](../part-5-perception-and-estimation/17-sensors-noise-filtering.md); demonstration'lar [ders 26](../part-7-learning-from-demonstration/26-teleop-replay-data.md)'da kullanılıyor.

Kaynaklar:
- Lynch & Park, [*Modern Robotics*](https://hades.mech.northwestern.edu/index.php/Modern_Robotics), bölüm 12 (Grasping and Manipulation).
- MuJoCo [Modeling · contact parametreleri (priority, friction)](https://mujoco.readthedocs.io/en/stable/modeling.html), [`mj_contactForce`](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html); [Menagerie `robotstudio_so101`](https://github.com/google-deepmind/mujoco_menagerie/tree/f3b6013/robotstudio_so101) (Apache-2.0).
- LeRobot [`so_follower.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/so_follower.py) (27.10.2026'da okundu); [LeRobot SO-101 dokümanı](https://huggingface.co/docs/lerobot/so101).
- Ölçülen tüm sayılar: [`experiments/16-grasping-and-pick-place/output.txt`](../../experiments/16-grasping-and-pick-place/output.txt).
