# 14 · Operational-space ve impedance control

<!-- meta -->
| | |
|---|---|
| **Bölüm** | [4 · Dynamics ve control](README.md) |
| **Çalışma türü** | CPU · Mac ya da Colab CPU |
| **Önkoşul** | [11](11-rigid-body-dynamics.md), [12](12-joint-control.md) |
| **Önce öğren** | [Linear Systems](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/17-linear-systems) |
| **Süre** | ~60 dk |
| **Kod** | [`experiments/14-impedance-control/experiment.py`](../../experiments/14-impedance-control/experiment.py) |
| **Durum** | ✅ tam |
<!-- /meta -->

> Position control "el nerede?" diye sorar. Impedance control "el orada değilken ne kadar sert geri itmeli?" diye sorar.

## Problem

Beklenmedik bir masaya çarpan sert bir position controller bir şey pes edene kadar itmeye devam eder: motorun torque limiti, parça ya da masa. Silme, bir pimi deliğe sokma ya da bir düğmeye basma gibi işler tersini ister: seçilmiş bir stiffness'a sahip yay gibi davranan bir el. libfranka'nın Kartezyen impedance örneği bir Panda'da tam bunu yapıyor: 150 N/m doğrusal stiffness ve 10 N·m/rad dönme stiffness'ı, joint torque'ları her döngüde Jacobian'dan hesaplanıyor ([`cartesian_impedance_control.cpp`](https://github.com/frankarobotics/libfranka/blob/main/examples/cartesian_impedance_control.cpp)). Bu ders o controller'ı ve onun operational-space akrabasını MuJoCo'da torque ile kontrol edilen bir Panda üzerinde kuruyor, sonra stiffness'ın gerçekte ne kazandırdığını görmek için eli sert bir yüzeye bastırıyor.

## Ana fikir

$x$ tool centre point'in (TCP) pozu, $\dot x = J(q)\dot q$ onun 6 boyutlu hızı (ders 09), $e$ poz hatası olsun (konum ve oryantasyon olarak $\tfrac12\sum_i R_i \times R_{d,i}$).

**Task-space PD / impedance.** Sanal bir yay–sönümleyiciden istenen wrench'i seç ve onu $J^\top$ ile joint'lere aktar (sanal iş ilkesi):

$$\tau = J^\top \big(K e - D\dot x\big) + g(q) \;(+\, C\dot q)$$

Bir engele dayanmış dururken el $F = K e$ ile iter: kuvveti $K$'yı ve hedefin yüzeyin ne kadar ötesinde durduğunu seçerek belirlersin (Hogan 1985).

**Operational space.** Kolun elde hissedilen ataleti

$$\Lambda(q) = \big(J M^{-1} J^\top\big)^{-1}$$

ve bu, yöne ve poza göre değişir. $F = \Lambda\,(k_p e - k_d \dot x)$ komutu eli her yönde birim kütle gibi davrandırır; böylece tek bir gain çifti x, y ve z boyunca aynı tepkiyi verir (Khatib 1987; Lynch & Park bölüm 11). Tam bir uygulama task-space Coriolis terimini de iptal eder; burada $C\dot q + g$ joint space'te `qfrc_bias` ile iptal ediliyor.

**Null space.** 6 boyutlu bir pozu kontrol eden 7 joint'li bir kolda, eli hareket ettirmeyen bir joint-space yönü vardır. Şununla izdüşürülen torque'lar

$$N = I - J^\top \bar J^\top, \qquad \bar J = M^{-1}J^\top\Lambda,$$

yalnızca o yönde etki eder; bir duruş terimi $N\,(k_0 (q_0 - q) - d_0\dot q)$ dirseği eli bozmadan istediğin yerde tutar.

**Modelde neyi değiştirdik?** Menagerie'nin Panda'sında position-servo actuator'ları var. Onların `gainprm` ve `biasprm` değerlerini sıfırladık (kuvvet üretmiyorlar), kendi torque'larımızı `qfrc_applied`'a yazdık ve modelin `forcerange` değerlerinde kırptık (joint 1–4 için 87 N·m, joint 5–7 için 12 N·m). `MjSpec` ile flanşın 10,34 cm önüne bir `tcp` site'ı ve hemen önüne 1 cm yarıçaplı bir probe küresi ekledik; B bölümü için elin altına sert bir blok ekledik ve onunla yalnızca probe çarpışabiliyor (`contype = conaffinity = 2`).

## Deney

Tam script: [`experiments/14-impedance-control/experiment.py`](../../experiments/14-impedance-control/experiment.py). MuJoCo 3.17.0, NumPy 2.5.4, Python 3.14, Apple M5, deterministik, ~3 s. Başlangıç pozu: `home`'dan TCP (50, 0, 30) cm'ye IK, el aşağı bakıyor (q = 0,0, −0,11, −0,0, −2,273, −0,0, 2,163, −0,785).

```python
Lam = np.linalg.inv(J @ Minv @ J.T)
if mode == "impedance":
    F = np.concatenate([K * e[:3] - D * v[:3], Kr * e[3:] - Dr * v[3:]])
else:                                            # operational space
    F = Lam @ (kp * e - kd * v)
tau = J.T @ F + d.qfrc_bias[:7]
tau = np.clip(tau, -lim, lim)
```

**A. x, y, z boyunca 5 cm'lik step'ler** (her biri 2 s). Impedance: K = 600 N/m, D = 50 N·s/m. Operational space: $k_p$ = 100 s⁻², $k_d$ = 14 s⁻¹. *Tahmin et:* hangi controller üç yönde de aynı tepkiyi verir?

| controller | eksen | $\Lambda_{ii}$ | overshoot | ±2 mm'ye giriş süresi | eksen dışı tepe |
|---|---|---|---|---|---|
| impedance | x | 11,87 kg | %13,0 | 0,854 s | 5,43 mm |
| impedance | y | 5,10 kg | %9,8 | 0,532 s | 1,02 mm |
| impedance | z | 6,88 kg | %17,0 | 0,572 s | 5,61 mm |
| operational space | x | 11,87 kg | %3,2 | 0,322 s | 1,70 mm |
| operational space | y | 5,10 kg | %3,8 | 0,304 s | 0,68 mm |
| operational space | z | 6,88 kg | %4,6 | 0,500 s | 2,97 mm |

El x boyunca 11,87 kg, y boyunca ise 5,10 kg hissediyor. Tek bir K ve tek bir D ile impedance control her yönde farklı bir yay–kütle sistemi oluyor (oturma 0,532–0,854 s) ve eksenleri birbirine bağlıyor (5,61 mm eksen dışı). Λ ile ölçeklemek tepkileri birbirine yaklaştırıyor (%3,2–4,6 overshoot) ve eksen dışı hareketi küçültüyor. Kalan farklar bu controller'ın dışarıda bıraktıklarından geliyor (task-space Coriolis, joint damping, torque kırpma).

**A2. Null space.** Operational-space control TCP'yi sabit tutuyor; +y yönünde 20 N'luk bir itme 0,3 s boyunca dirseğe (`link4`) uygulanıyor.

| controller | TCP hatası (en büyük) | sondaki dirsek yer değiştirmesi | sondaki $\|q - q_0\|$ |
|---|---|---|---|
| null-space terimi yok | 14,13 mm | 0,30 cm | 0,124 rad |
| + null-space duruş terimi | 14,14 mm | 0,00 cm | 0,000 rad |

Duruş terimi olmadan fazla serbestlik yönünü hiçbir şey geri getirmiyor: kol, el aynı yerdeyken farklı bir konfigürasyonda kalıyor. İzdüşürülmüş terim el hatasını değiştirmeden onu geri getiriyor.

**B. Bir yüzeye bastır.** Hedef, yüzeyin 3 cm üstünden 5 cm altına rampalanıyor (quintic, 1 s) ve 1,5 s tutuluyor. Impedance control, $D = 2\sqrt{K\Lambda_{zz}}$, $K_r$ = 30 N·m/rad. *Tahmin et:* tutma kuvveti her K için $K \times 5$ cm mi?

| K [N/m] | K × 5 cm | tutma kuvveti | tepe kuvvet | penetration | saturation'daki adımlar | contact kaybı | en büyük eğilme |
|---|---|---|---|---|---|---|---|
| 100 | 5,0 N | 5,00 N | 54,7 N | 0,545 mm | %0,0 | 0 | 0,46° |
| 300 | 15,0 | 14,98 | 74,7 | 0,708 | %0,0 | 0 | 0,56° |
| 1000 | 50,0 | 49,77 | 89,5 | 0,823 | %0,0 | 0 | 0,59° |
| 3000 | 150,0 | 148,45 | 148,5 | 0,888 | %0,0 | 0 | 0,60° |
| 10000 | 500,0 | 483,60 | 1028,1 | 1,823 | %77,9 | 0 | 56,91° |
| 30000 | 1500,0 | 395,95 | 1132,7 | 19,180 | %80,8 | 6 | 96,27° |
| 100000 | 5000,0 | 205,03 | 1245,0 | 21,679 | %81,8 | 10 | 111,22° |

K = 3000'e kadar tutma kuvveti K × 5 cm ile örtüşüyor (K = 3000'de 150 N'a karşı 148,45 N): **kuvvet ölçülmüyor, tasarlanıyor.** Penetration 1 mm'nin altında kalıyor ve onu seçtiğin stiffness değil MuJoCo'nun soft-contact parametreleri belirliyor: kuvvet yaklaşık 30 kat artarken penetration yalnızca 0,545'ten 0,888 mm'ye çıkıyor. Düşük K'daki tepe kuvvet (K = 100'de 5,00 N'luk tutma kuvvetine karşı 54,7 N) çarpmadır: $\Lambda_{zz}$ = 6,88 kg'lık bir el hızla gelir ve arkasında hangi yayı tasarlamış olursan ol, sert contact onu durdurur.

## Boz

K'yı joint'lerin verebileceğinin ötesine çıkar. K = 10000'de komut edilen wrench, adımların %77,9'unda bir joint'in verebileceğinden fazla torque istiyor. Joint başına kırpma her torque'u yasal tutuyor ama gerçekte uygulanan wrench'in yönünü değiştiriyor ve oryantasyon döngüsü kaybediyor: el 56,91° eğiliyor. K = 30000 ve 100000'de eğilme 90°'yi geçiyor, probe yüzeyde zıplıyor (tutma sırasında 6 ve 10 contact kaybı), tutma kuvveti 395,95 ve 205,03 N'a düşüyor ve ölçülen penetration (artık eğilmiş probe boyunca 19,180 ve 21,679 mm) hiç de bir contact özelliği değil. Ders şu: bir stiffness yalnızca torque limitlerine ulaşılmadığı sürece gerçektir. Ulaşıldığında joint'leri tek tek kırpmak yerine komutun tamamını ölçekle ya da wrench'i task space'te sınırla; ders 15 torque limitlerine bir safety aracı olarak geri dönüyor.

## Gerçek robotta

- **Panda.** libfranka'nın örneği $K$ = 150 N/m ve 10 N·m/rad ile $2\sqrt{K}$ damping (birim kütle) kullanıyor ve $\tau_d = J^\top(-K e - D J\dot q) + $ `coriolis` gönderiyor; komuta gravity terimi eklemiyor ([`cartesian_impedance_control.cpp`](https://github.com/frankarobotics/libfranka/blob/main/examples/cartesian_impedance_control.cpp)). Başlamadan önce robotun çarpışma eşiklerini de ayarlıyor (`setCollisionBehavior`). Bizim K = 100–300 satırlarımız bu aralıkta; yukarıdaki sayılar Panda ölçümleri değil, simülasyon sonuçları.
- **SO-101.** LeRobot STS3215 servolarını position modunda çalıştırıyor ([`so_follower.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/so_follower.py)); joint torque komutu yok, dolayısıyla bu dersin controller'ı onda yazıldığı hâliyle çalışamaz. Oradaki esneklik servonun kendi gain'lerinden ve torque limitinden gelir (ders 12, C bölümü).
- **Contact modelleri farklıdır.** Gerçek contact stiffness'ı, sürtünme ve robotun kendi joint esnekliği çarpma tepesini şekillendirir; 54,7 N'luk sıçrama MuJoCo'nun 2 ms adımdaki soft contact'ının bir özelliğidir ve gerçek bir kuvvet olarak okunmamalıdır.

## Kendini dene

1. **Kolay.** K = 1000 N/m ile 20 N'luk bir bastırma için hedef yüzeyin ne kadar arkasında durmalı?
2. **Orta.** A2 bölümünde TCP hatası null-space terimi olmadan 14,13 mm, onunla 14,14 mm. Null-space terimi neden onu azaltamıyor?
3. **Zor.** B bölümünde joint başına kırpmayı tek tip bir ölçeklemeyle değiştir, $\tau \leftarrow \tau \cdot \min(1, \min_i \tau_{\max,i}/|\tau_i|)$, ve K = 10000'i tekrar koştur. Koşturmadan önce tahmin et: eğilme kaybolur mu? Kuvvete ne olur?

### Cevaplar

1. $e = F/K$ = 20 / 1000 = 2 cm.
2. Dinamik olarak tutarlı izdüşüm $N = I - J^\top \bar J^\top$ ile null-space torque'ları elde hiçbir ivme üretmez; dolayısıyla bir el hatasına ne yol açabilir ne de onu düzeltebilir. 14 mm dirseğe yapılan itmenin kendisinden geliyor ve ona yalnızca controller'ın task-space kısmı direniyor.
3. Burada hesaplanmadı; koştur.

## Akılda kalanlar ve kaynaklar

- $\tau = J^\top F + g$ tasarlanmış bir task-space yayını joint torque'larına çevirir; bir yüzeye karşı kuvvet $K e$'dir (5 cm'de K = 100–3000 için 5,00 / 14,98 / 49,77 / 148,45 N).
- Elin ataleti $\Lambda$ yöne bağlıdır (burada 11,87'ye karşı 5,10 kg); $\Lambda$ ile ölçeklemek (operational space) tekdüze tepkiler verir; null-space izdüşümü fazladan serbestliği olan dirseği eli oynatmadan kontrol eder.
- Stiffness torque limitlerinde biter: onların ötesinde joint başına kırpma eli eğer (K = 10000'de 56,91°) ve kuvvet artık K'yı izlemez.

Sonraki: [ders 15 · Safety: limit'ler, torque cap, e-stop ve watchdog](15-safety.md).

Kaynaklar:
- O. Khatib, "A unified approach for motion and force control of robot manipulators: The operational space formulation", *IEEE Journal on Robotics and Automation* 3(1):43–53, 1987, [doi:10.1109/JRA.1987.1087068](https://doi.org/10.1109/JRA.1987.1087068).
- N. Hogan, "Impedance Control: An Approach to Manipulation: Part I—Theory", *Journal of Dynamic Systems, Measurement, and Control* 107(1):1–7, 1985, [doi:10.1115/1.3140702](https://doi.org/10.1115/1.3140702).
- Lynch & Park, [*Modern Robotics*](https://hades.mech.northwestern.edu/index.php/Modern_Robotics), bölüm 11 (Robot Control: task-space hareket, kuvvet ve impedance control).
- libfranka [`cartesian_impedance_control.cpp`](https://github.com/frankarobotics/libfranka/blob/main/examples/cartesian_impedance_control.cpp); LeRobot [`so_follower.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/so_follower.py) (27.10.2026'da okundu).
- MuJoCo: [`mj_jacSite`, `mj_fullM`, `mj_contactForce`](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html); [Menagerie `franka_emika_panda`](https://github.com/google-deepmind/mujoco_menagerie/tree/f3b6013/franka_emika_panda).
- Ölçülen bütün sayılar: [`experiments/14-impedance-control/output.txt`](../../experiments/14-impedance-control/output.txt).
