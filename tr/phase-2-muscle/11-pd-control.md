# 11 · 12 eklem, iki sayı: 20 ve 0,5

> Yapay zekâ "bacak şu açıya" der. Oraya iten, iki sayılık bir kuraldır.

> **Durum: taslak.** Deney koşturuldu, notebook hazır ve sayılar doğrulandı. Metin henüz kitap formatında tamamlanmadı; I (integral) terimi bu sayfada eksik.

| | |
|---|---|
| **Faz** | [2 · Kas](README.md) |
| **Tür** | Deney |
| **Önkoşul** | [10 · Konum servosu](10-position-servo.md) |
| **Süre** | ~45 dk |
| **Araç** | MuJoCo 3.17 · Menagerie `unitree_go2` · Colab ya da Mac |
| **Kod** | [notebook](../../notebooks/11-go2-two-numbers.ipynb) |
| **Durum** | 📝 taslak · notebook hazır |

## Neden önemli

Unitree Go2'nin 12 eklemi var (bacak başına hip, thigh, knee: kalça, uyluk, diz). Unitree'nin açık training kodunda, training simülasyonundaki her eklem aynı iki sayıyla sürülüyor: `stiffness = 20` (Kp, N·m/rad) ve `damping = 0.5` (Kd, N·m·s/rad), `control_type = 'P'` ([`go2_config.py`](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/go2/go2_config.py)). Tork şu satırla hesaplanıyor ([`legged_robot.py` · `_compute_torques`](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/base/legged_robot.py)):

```python
torques = p_gains * (actions_scaled + default_dof_pos - dof_pos) - d_gains * dof_vel
```

Policy 50 Hz'de hedef açıyı söylüyor (`sim.dt = 0.005` × `decimation = 4` = 20 ms), PD control kuralı onu her 5 ms'de torka çeviriyor.

**Dikkat:** bu sayılar **training simülasyonu** için. Unitree'nin reposunda Go2'yi gerçek robota deploy etmek için config yok (yalnız G1, H1, H1_2). Unitree'nin gerçek Go2 için low-level kalkma (stand) örneği ise **Kp = 60, Kd = 5** kullanıyor ([`go2_stand_example.py`](https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/example/go2/low_level/go2_stand_example.py)).

## Kavram

```python
tau = Kp * (q_hedef - q) - Kd * q_hiz
```

- **Kp** (stiffness) farkı torka çevirir: yay gibi. Büyüdükçe eklem hedefe daha sert çekilir.
- **Kd** (damping) hızı frenler: amortisör gibi. Salınımı ve overshoot'u söndürür.
- **I terimi** (birikmiş hata) burada yok. Yük altındaki kalıcı sarkmayı kapatmanın klasik yolu I'dır; Go2 training'inde bu farkı policy, hedefi kaydırarak kapatır (bkz. Sayılarla sonuç).

## Deney (notebook'ta)

Go2 yerde, bacakları katlı yatıyor. 12 ekleme 3 sn boyunca aynı PD kuralı veriliyor; tork motorun kendi sınırında kırpılıyor (clip). Hedef açılar Unitree config'indeki `default_joint_angles`. Ölçüm gövde yüksekliği üzerinden. **Overshoot**, hedefe göre değil, robotun 3. saniyedeki kendi oturduğu yüksekliğe göre: (tepe − son) / (son − başlangıç).

## Sayılarla sonuç

MuJoCo 3.17.0, 2 ms timestep, Menagerie `unitree_go2`. Sayılar ham loglardan yeniden hesaplandı ve aynı protokolle yeniden koşturuldu (birebir aynı).

| Kp / Kd | Başlangıç | Tepe | Son (2,5–3 sn ort.) | Overshoot | Not |
|---|---|---|---|---|---|
| **20 / 0,5** (Unitree training) | 10,0 cm | 25,6 | **25,2** | %3,0 | Dört ayak hep yerde |
| 5 / 0,5 | 9,9 | 11,1 | **11,1** | — | Kalkamadı |
| 80 / 0,5 | 10,0 | **50,8** | 29,9 | %104,5 | Tepe anında (t = 0,33 sn) dört ayak da havada; tork doymuş (saturation) |
| 20 / 0 | 10,0 | 26,3 | 24,7 | **%11,0** | Damping yokken daha çok sekiyor |
| 20 / 2 | 10,0 | 25,3 | 25,2 | %1,2 | |
| 60 / 5 (gerçek Go2 örneği) | | | 29,9 | %0,2 | |

**7 santimlik açık.** Hedef açılar tam tutulsaydı gövde yaklaşık **32 cm**'de olurdu (ön bacak 32,35, arka 31,39 cm; yalnız geometri). 20 / 0,5 ile gövde 25,2 cm'de kalıyor: **~7 cm alçakta**. Tek başına PD bu farkı kapatmıyor. Training'de policy hedefi kaydırarak (`0.25 · action`) kapatır.

**Sağlam olanlar ve olmayanlar.** "Kp 5 kalkamaz", "Kp 80 fırlar" ve "20 / 0,5'te ~7 cm sarkma" başlangıç pozundan, ölçüm penceresinden ve modelin pasif damping'inden bağımsız çıktı. Kd 0 ile 0,5 arasındaki overshoot **oranı** ise sağlam değil (pencereye ve modele göre 2,2× ile 3,7× arası). Yalnız "Kd'yi silince daha çok seker" niteliği sağlam.

## Sınırlar

- Menagerie Go2 modeli eklemlere pasif damping (`damping="2"`) ve friction loss (`frictionloss="0.2"`) veriyor; training URDF'inde bunlar yok. Bu yüzden "Kd = 0" koşusunda bile eklemde 2'lik damping var ve bütün salınımlar training simülasyonundakinden yumuşak.
- Timestep 2 ms (training'de 5 ms); diz tork sınırı 45,43 N·m (training URDF'inde 35,55).
- Bu bir fizik modeli deneyi: yapay zekâ yok, sabit hedef açılara PD var.
- Go2'nin `base_height_target = 0.25` reward'ı Go2 training'inde kullanılmıyor (ağırlığı 0). 25 cm ile ölçülen 25,2 cm'nin çakışması tesadüf.

## Alıştırmalar (notebook'un "BURAYI DEĞİŞTİR" hücresiyle)

1. **Kolay.** `KP = 60, KD = 5` ile koştur. Gövde nerede duruyor?
2. **Orta.** Sırta 5 kg yük koy (`LOAD_KG = 5`). 20 / 0,5 ile sarkma kaç cm artıyor?
3. **Orta.** [`go2_stand_example.cpp`](https://github.com/unitreerobotics/unitree_sdk2/blob/main/example/go2/go2_stand_example.cpp)'yi oku: hedef açılar bir anda mı veriliyor, yoksa adım adım mı kaydırılıyor? Kaydırılırsa büyük Kp neden daha az fırlatır? Deneyde bunu uygula.
4. **Zor.** Kurala bir I terimi ekle (`Ki * ∫(q_hedef − q) dt`). 7 cm'lik açık kapanıyor mu? Ne kadar sürede? Overshoot ne oluyor?

## Kaynaklar

- [`unitree_rl_gym` · `go2_config.py`](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/go2/go2_config.py), [`legged_robot.py`](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/base/legged_robot.py), [`legged_robot_config.py`](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/base/legged_robot_config.py) (BSD-3-Clause, © Unitree Robotics).
- [`unitree_rl_gym/deploy`](https://github.com/unitreerobotics/unitree_rl_gym/tree/main/deploy): Go2 deploy config'i yok.
- [`unitree_sdk2` · `go2_stand_example.cpp`](https://github.com/unitreerobotics/unitree_sdk2/blob/main/example/go2/go2_stand_example.cpp): Kp 60, Kd 5.
- [MuJoCo Menagerie · `unitree_go2`](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_go2): Unitree URDF'inden türetilmiş, klasör lisansı BSD-3-Clause (© Unitree Robotics).
- Lynch & Park, [*Modern Robotics*](https://hades.mech.northwestern.edu/index.php/Modern_Robotics), Bölüm 11 (robot control).
- Sayılar Koddan Kasa 11 v3 deneyinin loglarından ve iç doğrulama notundan (27.10.2026).

> Go2 modeli: MuJoCo Menagerie (Google DeepMind), Unitree Robotics URDF'inden, BSD-3-Clause. Unitree ve Go2 adları olgusal olarak kullanılmıştır; Unitree bu çalışmayı desteklemez ya da onaylamaz.
