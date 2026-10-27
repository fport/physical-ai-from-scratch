# 02 · Time integration ve timestep

<!-- meta -->
| | |
|---|---|
| **Bölüm** | [1 · Simulation](README.md) |
| **Çalışma türü** | CPU · Mac ya da Colab CPU |
| **Önkoşul** | [01](01-world-as-text.md) |
| **Önce öğren** | [Calculus for Machine Learning](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/04-calculus-for-ml) · [Numerical Stability](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/13-numerical-stability) |
| **Süre** | ~30 dk |
| **Kod** | [`experiments/02-timestep/experiment.py`](../../experiments/02-timestep/experiment.py) · [notebook](../../notebooks/02-timestep.ipynb) |
| **Durum** | ✅ tam |
<!-- /meta -->

> Simülatör zamanı akıtmaz; `dt` kadar sıçrar ve contact'ı yalnızca düştüğü yerde arar.

## Problem

Unitree'nin bacaklı robotları için açık eğitim kodu (Isaac Gym) fiziği `sim.dt = 0.005` s ile adımlıyor ve policy'nin her `decimation = 4` adımda bir karar vermesine izin veriyor, yani saniyede 50 karar ([`legged_robot_config.py`](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/base/legged_robot_config.py)). Neden 5 ms de 20 değil? MuJoCo timestep'i hız–doğruluk dengesini etkileyen "en önemli tek parametre" olarak tanımlıyor ([XML reference · option/timestep](https://mujoco.readthedocs.io/en/stable/XMLreference.html#option)). Somut bir hata: 1 m yükseklikten 1 cm'lik bir masa tablasına bırakılan 5 cm'lik bir küp yerde bitiyor. Soru, hangi `dt` değerlerinin buna yol açtığı ve "küçük her zaman güvenlidir" sözünün doğru olup olmadığı.

## Ana fikir

### Semi-implicit Euler

MuJoCo ivmeleri sürekli zamanda hesaplar, sonra integre eder. Default `Euler` integrator'ı semi-implicit'tir: konum güncellemesi *yeni* hızı kullanır ([Computation · Numerical integration](https://mujoco.readthedocs.io/en/stable/computation/index.html#geintegration)):

$$v_{t+h} = v_t + h\,a_t, \qquad q_{t+h} = q_t + h\,v_{t+h}$$

Bir adımda cisim $v h$ kadar sıçrar. $t$ ile $t+h$ arasında olan hiçbir şey hesaplanmaz. (Bir ODE'yi ayrıklaştırmak: bkz. AIEFS: [Calculus for Machine Learning](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/04-calculus-for-ml).)

### Contact yalnızca adımlarda kontrol edilir

Collision detection her adımdaki konumlar üzerinde çalışır. Hızlı bir cisim ve ince bir engel için üç şey ters gidebilir:

| Hata | Koşul | Sonuç |
|---|---|---|
| kaçırılan contact | hiçbir adım çakışmaya denk gelmez | engine masayı hiç görmez |
| yanlış taraftan çakışma | ilk çakışma zaten tablanın orta düzleminin ötesinde | box–box contact normal'i *aşağı* bakar; solver küpü içinden iter |
| fazla yumuşak contact | contact doğru tarafta görülür ama soft constraint küpün orta düzlemin ötesine batmasına izin verir | normal çarpma sırasında ters döner |

Üçüncüsü aslında bir timestep problemi değil; onu `solref` belirler ([ders 03](03-contact.md)).

### Integrator'lar

| `integrator` | Neyi implicit çözer | Neyi iyileştirir |
|---|---|---|
| `Euler` (default) | yalnızca joint damping | en ucuzu; doküman onu uyumluluk için tutuyor ve `implicitfast` öneriyor |
| `implicitfast` | Coriolis/merkezkaç dışındaki bütün hıza bağlı kuvvetler | sert damping, actuator `kv`, drag |
| `implicit` | aynısı artı Coriolis/merkezkaç (LU çözümü) | birbirine bağlı dönen zincirler (çok uzuvlu sarkaçlar) |
| `RK4` | hiçbiri; 4. mertebe, çok aşamalı | düzgün hareketin doğruluğu, adım başına 4 kuvvet hesabı |
| `discrete` | hız *ve* konum stiffness'ı, constraint'lerle birlikte | explicit limitin çok ötesinde sert yaylar ve position servo'lar |

Kaynak: [Computation · Integrators](https://mujoco.readthedocs.io/en/stable/computation/index.html#geintegrators), [option/integrator](https://mujoco.readthedocs.io/en/stable/XMLreference.html#option-integrator). Hiçbiri continuous collision detection eklemez: contact'lar yalnızca örneklenen state'lerde bulunur, aradaki yolda asla. Seçimin burada önemli olup olmadığı aşağıda ölçülüyor.

## Deney

Sahne: zemin, üst yüzü 30 cm'de olan 24 × 24 × 1 cm'lik bir tabla (orta düzlem 29,5 cm) ve alt yüzü tablanın 97,5 cm üstünden başlayan 5 cm, 100 g'lık bir küp.

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
</mujoco>"""

m = mujoco.MjModel.from_xml_string(XML)
m.opt.timestep = dt_ms / 1000          # <- the swept number
```

Script her `dt` için şunları kaydediyor: tablayla ilk çakışan adımda küp merkezinin yüksekliği, masa contact'ı bulunup bulunmadığı, çarpma sırasındaki en alçak merkez ve son sonuç. Sweep'i daha sert bir contact'la, iki geom'a da `solref="0.005 1"` vererek tekrarlıyor (contact iki geom'un değerlerinin ortalamasını kullanır; `dt` 2,5 ms'nin üstündeyken solver 0,005'i 2·`dt`'ye yükseltir, bkz. [ders 03](03-contact.md)).

**Tahmin et.** Çarpma hızı $\sqrt{2 \cdot 9{,}81 \cdot 0{,}975}$ = 4,37 m/s, yani bir adım 2 ms'de 0,87 cm, 10 ms'de 4,37 cm. Küp nerede içinden geçer?

**Çalıştır.** [`experiment.py`](../../experiments/02-timestep/experiment.py), MuJoCo 3.17.0, NumPy 2.5.4, Apple M5, deterministic; çıktı [`output.txt`](../../experiments/02-timestep/output.txt)'de.

| dt (ms) | adım (cm) | ilk çakışmada merkez (cm) | contact görüldü | en alçak merkez (cm) | default `solref` | `solref 0.005 1` |
|---|---|---|---|---|---|---|
| 0,5 | 0,22 | 32,32 | evet | — | **geçti** | kaldı |
| 1 | 0,44 | 32,21 | evet | — | **geçti** | kaldı |
| 2 | 0,87 | 31,99 | evet | 29,60 | kaldı | kaldı |
| 3 | 1,31 | 31,34 | evet | — | **geçti** | kaldı |
| 4 | 1,75 | 32,43 | evet | 30,54 | kaldı | kaldı |
| 6 | 2,62 | 32,00 | evet | 30,84 | kaldı | kaldı |
| 8 | 3,50 | 29,80 | evet | 29,65 | kaldı | kaldı |
| 10 | 4,37 | **28,47** | evet | — | **geçti** | **geçti** |
| 12 | 5,25 | 30,69 | evet | 30,69 | kaldı | kaldı |
| 15 | 6,56 | **27,36** | evet | — | **geçti** | **geçti** |
| 20 | 8,75 | 30,72 | evet | 30,72 | kaldı | kaldı |
| 25, 30, 40, 50 | 10,93–21,87 | — | **hayır** | — | **geçti** | **geçti** |

**Gözlemle.** Üç hata türünün üçü de görünüyor:

1. **Kaçırılan contact, ≥ 25 ms.** Hiçbir adım çakışmaya denk gelmiyor; hiç contact üretilmiyor.
2. **Yanlış taraftan çakışma, 10 ve 15 ms.** İlk çakışan adımda merkez zaten 29,5 cm'nin altında. 10 ms'de ilk masa contact'ı (t = 0,460 s) 1,97 cm derinde ve küpe etki eden normal'in z bileşeni −1,00: kuvvetin büyüklüğü doğru, yönü yanlış. Daha sert bir contact burada işe yaramaz.
3. **Fazla yumuşak contact, 0,5, 1 ve 3 ms.** İlk çakışma doğru tarafta; 1 ms'de ilk contact (t = 0,447 s) 0,29 cm derinde ve normal z = +1,00. Default soft contact yine de 4,37 m/s'lik küpün batmasına izin veriyor: t = 0,463 s'de merkez 29,49 cm'ye, orta düzlemin hemen altına iniyor (1 cm'lik tablaya yaklaşık 3 cm penetration), normal aşağı dönüyor ve küp düşüyor. Ayakta kalan 2 ms koşusu bile 29,60 cm'ye, dönmeye 0,1 cm kala iniyor. `solref="0.005 1"` ile bunların hepsi kalıyor.

Sonuç `dt`'ye göre monoton değil: "küçük daha güvenli" 0,5 ve 1 ms'de, "tabladan ince adım güvenli" 3 ms'de çöküyor; 12 ve 20 ms ise son contact'sız adımın nereye düştüğü şansıyla ayakta kalıyor.

## Boz

`dt`'yi sabit tut, yalnızca integrator'ı değiştir (default contact):

| dt | Euler | implicitfast | RK4 | discrete |
|---|---|---|---|---|
| 1 ms | geçti | geçti | geçti | geçti |
| 2 ms | kaldı (29,60) | kaldı (29,60) | **geçti** | **geçti** |
| 8 ms | kaldı (29,65) | kaldı (29,65) | **geçti** | **geçti** |
| 10 ms | geçti | geçti | geçti | geçti |
| 25 ms | geçti | geçti | geçti | geçti |

Hiçbir integrator başarısız bir durumu kurtarmıyor; RK4 ve `discrete`, Euler'in atlattığı iki durumda başarısız oluyor. Her yöntem çarpmayı farklı çözüyor; Euler koşusu dönmeye yalnızca 0,1 cm uzakken bu fark sonucu belirliyor. Integrator, detection'lar arasında kuvvetlerin harekete nasıl çevrileceğini kontrol eder; örneklenen state'ler arasındaki yolu taramaz ve yumuşak bir contact'ı sertleştirmez. Aday çözümler daha kalın collision geom'ları ya da daha sert bir `solref`'tir (yukarıdaki yumuşak-contact durumlarını düzelten), her biri bunun gibi bir sweep'le kontrol edilir. Integrator'ların *gerçekten* düzelttiği şeyler (sert damping ve sert position servo'lar) [ders 05](05-joints-and-actuators.md)'te ölçülüyor.

## Gerçek robotta

- SO-101'de Menagerie modeli zaten `timestep="0.005"` ve `integrator="implicitfast"` ayarlıyor ([so101.xml](https://github.com/google-deepmind/mujoco_menagerie/blob/f3b6013/robotstudio_so101/so101.xml)). Orta hızla ince bir nesneye (bir kart, bir kablo) kapanan bir gripper parmağı, bu küp ve tablayla aynı geometridir.
- Prosedür: simülasyondaki bir grasp'e güvenmeden önce grasp'i 5, 2 ve 1 ms'de çalıştır ve nesnenin son pose'unu karşılaştır. `dt` ile değişen bir sonuç grasp'in özelliği değil, simülasyon artefaktıdır. **Gerçek kolda henüz ölçülmedi**; gerçek dünyadaki karşılığıyla karşılaştırılmadı.
- RL'de `decimation`, control hızını fizik hızından ayırır: policy 50 Hz'de çalışırken fizik 200 Hz'de çalışabilir.

## Kendini dene

1. **Kolay.** Tablayı 4 cm kalın yap (`size="0.12 0.12 0.02"`, üst yüz 30 cm'de kalsın). Hangi hata türü önce kaybolmalı, neden?
2. **Orta.** Küpü 3 m'den bırak. Çarpma hızını hesapla, sonra sweep'i yeniden çalıştır. Başarısız `dt` kümesi büyüyor mu küçülüyor mu?
3. **Zor.** 0,5–20 ms aralığındaki her `dt` için küpün kaldığı en büyük `solref` timeconst'u (dampratio 1'de) bul. Çiz. Bunun neden 10 ve 15 ms'yi düzeltemeyeceğini açıkla.

### Cevaplar

1. Fazla yumuşak contact türü: normal'in dönebilmesi için merkezin artık üst yüzün 0,5 cm değil 2 cm altındaki yeni orta düzlemin altına inmesi gerekir. Geri kalanını ölç, varsayma.
2. Çarpma hızı yeni yükseklikle $\sqrt{2 g h}$; daha hızlı çarpmalar daha derin batar ve daha uzun adım atar, yani daha çok başarısızlık bekle, sonra kontrol et.
3. 10 ve 15 ms'de ilk çakışan adım zaten orta düzlemin altında, yani stiffness ne olursa olsun ilk contact aşağı bakar.

## Akılda kalanlar ve kaynaklar

- Semi-implicit Euler $v\,h$ kadar sıçrar; contact yalnızca adım anlarında tespit edilir.
- Bu sahnede üç hata türü bir arada: kaçırılan contact (büyük `dt`), yanlış taraftan çakışma (orta `dt`), fazla yumuşak contact (her `dt`, `solref` ile düzelir). Sonuç `dt`'ye göre monoton değil.
- Integrator değiştirmek hiçbirini düzeltmedi; contact'a kritik davranışı `dt` sweep'iyle doğrula.

Sonraki: [ders 03 · Contact: soft constraint, solref ve solimp](03-contact.md).

**Kaynaklar**
- [MuJoCo XML reference · option](https://mujoco.readthedocs.io/en/stable/XMLreference.html#option): `timestep` (default 0,002), `integrator`.
- [MuJoCo Computation · Numerical integration ve Integrators](https://mujoco.readthedocs.io/en/stable/computation/index.html#geintegration).
- [`unitree_rl_gym` · `legged_robot_config.py`](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/base/legged_robot_config.py): `dt = 0.005`, `decimation = 4` (2026-10-27'de okundu).
- [Menagerie · robotstudio_so101](https://github.com/google-deepmind/mujoco_menagerie/tree/f3b6013/robotstudio_so101).
- Sayılarımız: [`experiments/02-timestep/output.txt`](../../experiments/02-timestep/output.txt).
