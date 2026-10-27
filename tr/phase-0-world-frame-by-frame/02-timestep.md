# 02 · Timestep: küp masanın içinden neden geçti?

> Simülasyon zamanı akıtmaz, adım adım atlar. Collision'a da yalnız bu anlarda bakar.

| | |
|---|---|
| **Faz** | [0 · Dünya kare kare](README.md) |
| **Tür** | Deney |
| **Önkoşul** | [01 · Dünya bir metin dosyası](01-world-as-text.md) (şart değil, bu ders tek başına da okunur) |
| **Süre** | ~30 dk |
| **Araç** | MuJoCo 3.17 · Python · Mac ya da Colab CPU |
| **Kod** | [`experiments/02-timestep/experiment.py`](../../experiments/02-timestep/experiment.py) · [notebook](../../notebooks/02-timestep.ipynb) |
| **Durum** | ✅ tam |

## Neden önemli

Robot köpekler yürümeyi gerçek dünyada değil, simülasyonda öğreniyor. Unitree'nin açık eğitim kodu [`unitree_rl_gym`](https://github.com/unitreerobotics/unitree_rl_gym) Go2 için simülasyonu **5 ms**'lik adımlarla koşturuyor (`sim.dt = 0.005`). Yapay zekâ her 4 adımda bir karar veriyor (`decimation = 4`): saniyede 50 karar. Aradaki her adımda ise joint'lere tork uygulanıyor ([`legged_robot_config.py`](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/base/legged_robot_config.py), [`go2_config.py`](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/go2/go2_config.py)).

Bu sayı rastgele seçilmiyor. MuJoCo belgesi timestep'i, kullanıcının ayarlayabileceği belki de en önemli parametre olarak niteliyor ([Computation](https://mujoco.readthedocs.io/en/stable/computation/index.html)). Adım büyüdükçe simülasyon hızlanır, çünkü aynı saniyeyi daha az hesapla geçersin. Ama bir noktadan sonra fizik yanlış sonuç vermeye başlar. Bu derste o noktayı kendi gözünle bulacaksın: 1 metreden düşen bir küp, 1 cm kalınlığındaki bir tablanın içinden geçecek.

## Kavram

### Zaman sürekli değil, kare kare

Bir physics engine bir film makinesi gibi çalışır. Dünyanın durumunu (konumlar, hızlar) bir anda hesaplar, sonra zamanı `dt` kadar ileri alır ve yeniden hesaplar. MuJoCo'da bu sayının adı `timestep`, default değeri **0,002 sn** (2 ms), yani saniyede 500 adım ([XML referansı · option](https://mujoco.readthedocs.io/en/stable/XMLreference.html#option)).

Her adımda MuJoCo'nun default integrator'ı (semi-implicit Euler) iki iş yapar ([Computation](https://mujoco.readthedocs.io/en/stable/computation/index.html)):

```
v ← v + a · dt      # first the velocity is updated with the acceleration
x ← x + v · dt      # then the position is updated with the NEW velocity
```

Yani küp bir adımda **hızı × dt** kadar sıçrar. Arada nerede olduğu hesaplanmaz.

### Collision'a yalnız adım anlarında bakılır

Asıl mesele burada. MuJoCo contact'ları her adımda o anki konumlardan bulur. İki adım arasında küpün süpürdüğü yolu kontrol etmez. Küp tablaya yaklaşırken bir adımda tablanın üstündeyse ve sonraki adımda yarısından fazlası tablanın içindeyse, engine contact bulur, ama küpü dışarı çıkarmak için en kısa yolu seçer. O en kısa yol aşağı doğruysa, fizik küpü tablanın **altına** iter.

Daha büyük adımlarda küp tek adımda tablanın tamamen altına atlar ve contact hiç görülmez. Buna oyun ve simülasyon dünyasında **tunneling** denir.

## Kendin kur

### Adım 1 · Kurulum

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # mujoco==3.17.0, numpy
```

### Adım 2 · Sahne: zemin, 1 cm tabla, 5 cm küp

```python
XML = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <geom name="table" type="box" size="0.12 0.12 0.005" pos="0 0 0.295"/>
    <body name="cube" pos="0 0 1.30">
      <freejoint/>
      <geom type="box" size="0.025 0.025 0.025" mass="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""
```

MJCF'te `size` yarı uzunluk demek. Tabla 24 × 24 cm, 1 cm kalın, üst yüzü 30 cm'de. Küp 5 cm, merkezi 1,30 m'de, yani alt yüzü tablanın 97,5 cm üstünde.

### Adım 3 · Tek satırı değiştir: `timestep`

```python
import mujoco, numpy as np

def drop(dt_ms, sim_sec=1.5):
    m = mujoco.MjModel.from_xml_string(XML)
    m.opt.timestep = dt_ms / 1000          # <- the only number we change
    d = mujoco.MjData(m)
    z = []
    while d.time < sim_sec - 1e-12:
        mujoco.mj_step(m, d)
        z.append(d.body("cube").xpos[2])
    return np.array(z)

for dt in [2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50]:
    z = drop(dt)
    print(f"dt {dt:2d} ms · final height {z[-1]*100:5.1f} cm ·", "WENT THROUGH" if z[-1] < 0.25 else "stayed")
```

Aynı 1,5 saniyeyi kapsamak için adım sayısı da değişir (2 ms'de 750, 10 ms'de 150). Bu, `dt`'nin doğrudan sonucu, ayrı bir değişken değil.

### Adım 4 · Neden? İlk overlap anına bak

Tam script ([`experiment.py`](../../experiments/02-timestep/experiment.py)) her koşu için şunları da yazar: küpün tablayla ilk overlap ettiği adımda merkezi tablanın mid-plane'inin (29,5 cm) üstünde mi altında mı, ve contact'ın hiç görülüp görülmediği.

```bash
python experiments/02-timestep/experiment.py
```

## Sayılarla sonuç

MuJoCo 3.17.0, Apple M5, deterministik. Aynı script iki kez koşturuldu, aynı çıktıyı verdi. Sayılar [doğrulama notundaki](#kaynaklar) bağımsız koşuyla da birebir aynı.

Küp tablaya **4,37 m/s** ile çarpıyor (√(2 · 9,81 · 0,975)). 2 ms'de bir adım **0,87 cm**, 10 ms'de **4,37 cm**.

| dt (ms) | adım/sn | bir adımda yol (cm) | ilk overlap'te küp merkezi (cm) | contact görüldü mü | sonuç |
|---|---|---|---|---|---|
| 2 | 500 | 0,87 | 31,99 | evet | kaldı |
| 4 | 250 | 1,75 | 32,43 | evet | kaldı |
| 6 | 166 | 2,62 | 32,00 | evet | kaldı |
| 8 | 125 | 3,50 | 29,80 | evet | kaldı (kıl payı) |
| **10** | 100 | 4,37 | **28,47** | evet | **geçti** |
| 12 | 83 | 5,25 | 30,69 | evet | kaldı |
| **15** | 66 | 6,56 | **27,36** | evet | **geçti** |
| 20 | 50 | 8,75 | 30,72 | evet | kaldı |
| 25 | 40 | 10,93 | — | **hayır** | geçti |
| 30 | 33 | 13,12 | — | hayır | geçti |
| 40 | 25 | 17,49 | — | hayır | geçti |
| 50 | 20 | 21,87 | — | hayır | geçti |

Tablanın mid-plane'i 29,5 cm. Kural tablonun tamamında tutuyor: **küp tablayla ilk overlap ettiği adımda merkezi mid-plane'in altındaysa geçiyor, üstündeyse kalıyor.**

Üç gözlem:

1. **"Adım tabladan inceyse güvenli" kuralı yanlış.** 4, 6 ve 8 ms'de bir adım (1,75 / 2,6 / 3,5 cm) tablanın 1 cm kalınlığından büyük, ama küp kaldı. 12 ve 20 ms'de de kaldı.
2. **10 ve 15 ms'de contact yakalanıyor, ama ters yöne çözülüyor.** 10 ms koşusunda t = 0,46 sn'de küp tablaya 1,97 cm gömülü ve contact normal'in z bileşeni −1,00: MuJoCo küpü aşağı itiyor. Contact force doğru hesaplanıyor, sadece yanlış tarafa.
3. **25 ms ve üstü gerçek tunneling.** Küp tek adımda tablanın tamamen altına atlıyor, engine contact'ı hiç görmüyor.

Sonuç `dt` ile düzgün artmıyor (12 ve 20 kalıyor), çünkü hangi tarafa düşüleceği son contact'sız adımın tablaya ne kadar yakın düştüğüne bağlı. 2 ms bile "rahatça" yakalamıyor: soft contact küpü derine gömüyor, merkez en fazla 29,60 cm'ye, mid-plane'in yalnız 0,1 cm üstüne iniyor ve geri çıkıyor. (Soft contact [ders 03](03-soft-contact.md)'ün konusu.)

## Sınırlar

- Sahne sadeleştirilmiş: robot kol yok, yalnız zemin, tabla ve küp. Videodaki sahnede bir kol da duruyordu ama kontrolü sıfırdı; sonuçlar aynı.
- Her `dt` için tek koşu. MuJoCo deterministik olduğu için tekrar koşmak aynı sonucu verir. Bu "her seferinde" demek değil, "denediğim her değerde" demek.
- 50 ms'nin üstü denenmedi.
- Kural bu geometri (1 cm tabla, 5 cm küp, 97,5 cm düşüş) için ölçüldü. Başka kalınlık ya da hızda eşikler değişir.

## Alıştırmalar

1. **Kolay.** Tablayı 4 cm kalın yap (`size="0.12 0.12 0.02"`, `pos` değerini üst yüz 30 cm'de kalacak şekilde ayarla). Hangi `dt`'lerde geçiyor? Kuralı (ilk overlap'te merkez mid-plane'in altında mı) yeni mid-plane ile kontrol et.
2. **Orta.** Küpü 3 m yükseğe koy. Çarpma hızı kaç m/s olur? Eşik `dt` değişiyor mu? Önce tahmin et, sonra koştur.
3. **Orta.** `m.opt.integrator` değerini `mujoco.mjtIntegrator.mjINT_RK4` yap. Tablo değişiyor mu? Değişmiyorsa neden (ipucu: sorun integrator'da mı, collision check'in zamanlamasında mı)?
4. **Zor.** Verilen bir düşme yüksekliği için küpün geçtiği **en küçük** `dt`'yi 0,1 ms çözünürlükle bulan bir fonksiyon yaz. Yükseklik 0,25 m'den 3 m'ye çıkarken bu eşiği çiz. Eşik neden düzgün bir eğri değil?

## Terimler

| Terim | Sık duyulan | Aslında |
|---|---|---|
| `timestep` (dt) | "Simülasyonun hızı" | Physics engine'in bir adımda zamanı ne kadar ileri aldığı. Küçük dt daha doğru, ama aynı saniye için daha çok hesap demek |
| Semi-implicit Euler | "Basit integrator" | Önce hızı, sonra konumu yeni hızla güncelleyen yöntem. MuJoCo'nun default'u |
| Tunneling | "Fizik hatası" | Hızlı bir cisim tek adımda ince bir engelin öbür tarafına geçtiği için contact'ın hiç görülmemesi |
| Contact normal | "Çarpışma yönü" | Engine'in iki cismi ayırmak için seçtiği yön. Cisim yeterince içeri girmişse ters yönü gösterebilir |
| Decimation | — | RL eğitiminde policy'nin kaç physics step'te bir karar verdiği. `unitree_rl_gym`'de 4 |

## Kaynaklar

- [MuJoCo XML referansı · `option/timestep`](https://mujoco.readthedocs.io/en/stable/XMLreference.html#option): default 0,002 sn.
- [MuJoCo · Computation](https://mujoco.readthedocs.io/en/stable/computation/index.html): step döngüsü ve semi-implicit Euler; timestep'in önemi.
- [`unitree_rl_gym` · `legged_robot_config.py`](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/base/legged_robot_config.py): `sim.dt = 0.005`, `decimation = 4`. Go2 config'i `dt`'yi değiştirmiyor, `decimation`'ı yine 4 yazıyor (27.10.2026'da okundu).
- Bu sayfadaki her sayı `experiment.py`'nin koşusundan (MuJoCo 3.17.0, 27.10.2026). Penetration (1,97 cm), contact normal (−1,00) ve 2 ms'deki en düşük merkez (29,60 cm) aynı sahnede ayrı bir probe script'iyle okundu. Sayılar Koddan Kasa video ekibinin 25.10.2026 tarihli iç doğrulama notuyla da birebir aynı.
