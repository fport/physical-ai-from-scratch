# 01 · Dünya bir metin dosyası: MJCF ve URDF

<!-- meta -->
| | |
|---|---|
| **Bölüm** | [1 · Simulation](README.md) |
| **Çalışma türü** | CPU · Mac ya da Colab CPU |
| **Önkoşul** | yok |
| **Önce öğren** | [Python for AI Engineering](https://aiengineeringfromscratch.com/lesson?path=phases/00-setup-and-tooling/13-python-for-ai-engineering) · [Python Environments](https://aiengineeringfromscratch.com/lesson?path=phases/00-setup-and-tooling/06-python-environments) |
| **Süre** | ~30 dk |
| **Kod** | [`experiments/01-world-as-text/experiment.py`](../../experiments/01-world-as-text/experiment.py) · [notebook](../../notebooks/01-world-as-text.ipynb) |
| **Durum** | ✅ tam |
<!-- /meta -->

> Bir robot modeli bir metin dosyasıdır: engine'in kullandığı her sayı ya dosyada yazılıdır ya da belgelenmiş bir default değerle doldurulur.

## Problem

Her simülasyon bir model dosyası yükleyerek başlar ve yüklenen bir dosya fiziksel olarak doğru olmak zorunda değildir. MuJoCo'nun kendi referansı, herkese açık bazı URDF modellerinde inertia değerlerinin keyfi göründüğünü ve fazla büyük olduğunu söyler ([XML reference · compiler/inertiafromgeom](https://mujoco.readthedocs.io/en/stable/XMLreference.html#compiler-inertiafromgeom)). Dosyayı okuyamazsan hangi kütleleri, inertia'ları, limit'leri ve gain'leri simüle ettiğini bilemezsin. Bu derste MJCF'i eleman eleman okuyacak, 15 satırlık bir dünya yazacak, default değerlerin ona ne yaptığını görecek, sonra gerçek SO-101 modelini açıp içinde ne olduğunu sayacaksın.

## Ana fikir

### Tek tabloda MJCF

MJCF, MuJoCo'nun XML formatıdır. Compiler onu bir `mjModel`'e (bütün sabit sayılar) çevirir, simülatör de bir `mjData`'yı (değişen state) adım adım ilerletir.

| Eleman | Ne olduğu | Dokunacağın attribute'lar |
|---|---|---|
| `option` | global fizik ayarları | `timestep` (default 0,002 s), `integrator` (default `Euler`), `gravity`, `density`/`viscosity` (default 0: hava yok), `cone` |
| `worldbody` | ağacın sabit kökü | statik geom'lar (zemin, masa) |
| `body` | kinematic tree'de rigid bir frame | parent'a göre `pos`, `quat` |
| `geom` | collision için ve default olarak kütle için bir şekil | `type`, `size` (box için yarı uzunluklar), `mass` ya da `density` (default 1000 kg/m³), `friction`, `solref`/`solimp` |
| `joint` | bir body ile parent'ı arasındaki serbestlik derecesi | `type` (`hinge`, `slide`, `ball`, `free`), `axis`, `range`, `damping`, `armature` |
| `inertial` | bir body'nin açık yazılmış kütle özellikleri | `mass`, `pos` (kütle merkezi), `diaginertia` ya da `fullinertia` |
| `default` / `class` / `childclass` | attribute değerlerinin isimli şablonları | ör. her servo joint'i aynı `damping` ve `armature`'ı alır |
| `actuator` | kuvvet üreten şey | `motor`, `position`, `velocity`, `general`; `joint`, `gear`, `ctrlrange`, `forcerange` |
| `sensor` | `mjData.sensordata`'ya yazılan büyüklükler | `jointpos`, `framequat`, `touch`, … |

Kaynaklar: [XML reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html), [Modeling · Default settings](https://mujoco.readthedocs.io/en/stable/modeling.html#cdefault).

### Kütle ve inertia: yazılmış ya da çıkarılmış

Çalışma anında yalnızca *body*'nin inertial özellikleri vardır; geom kütleleri sadece compile sırasında kullanılır ([geom/mass](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-geom-mass)). Hangi kuralın onları dolduracağını `compiler/inertiafromgeom` belirler:

| Değer | Davranış |
|---|---|
| `auto` (default) | body'nin `<inertial>`'ı yoksa geom'larından çıkar |
| `true` | `<inertial>`'ı ezerek her zaman geom'lardan çıkar |
| `false` | her body'de `<inertial>` olmak zorunda |

Çıkarım düzgün yoğunluk varsayar. Tam kenar uzunlukları $a, b, c$ olan bir box için:

$$I_{xx} = \tfrac{m}{12}(b^2 + c^2),\quad I_{yy} = \tfrac{m}{12}(a^2 + c^2),\quad I_{zz} = \tfrac{m}{12}(a^2 + b^2)$$

### URDF ve MJCF

| | URDF | MJCF |
|---|---|---|
| Ağaç | `link` + `joint` | iç içe `body` + `joint` |
| Kütle | her link'te `inertial` | `inertial` ya da geom'lardan çıkarım |
| Actuator, sensor, solver, contact parametreleri | formatın parçası değil | birinci sınıf elemanlar |
| Şablonlar | yok | `default` class'ları |

MuJoCo URDF'i doğrudan yükler, ama modelin yalnızca bir alt kümesini, compiler seçenekleri için de özel bir `<mujoco>` bloğuyla ([Modeling · URDF extensions](https://mujoco.readthedocs.io/en/stable/modeling.html#curdf)). Menagerie her dönüşümü modelin README'sinde belgeler. Go2 için adımlar şunlar: mesh'leri dönüştür, URDF'e `<mujoco><compiler discardvisual="false" …/></mujoco>` ekle, MuJoCo'da yükleyip MJCF olarak kaydet, bir `freejoint` ekle, ortak değerleri `<default>`'a taşı, sonra ayak contact'larını ayarla ([Menagerie · unitree_go2](https://github.com/google-deepmind/mujoco_menagerie/tree/f3b6013/unitree_go2)). SO-101 modeli ise üreticinin açık MJCF'inden başlar, sonra `implicitfast`'e geçer ve collision primitive'leri ekler ([Menagerie · robotstudio_so101](https://github.com/google-deepmind/mujoco_menagerie/tree/f3b6013/robotstudio_so101)).

### Orientation, w-önce bir quaternion'dır

`quat="1 0 0 0"` sıfır rotation'dır. Birim eksen $(x, y, z)$ etrafında $a$ açılık bir rotation $(\cos\frac{a}{2},\ \sin\frac{a}{2}\,(x, y, z))$'dir: skaler önce ([Modeling · Frame orientations](https://mujoco.readthedocs.io/en/stable/modeling.html#corientation)). Başka kütüphaneler başka sıralar kullanır; bunu [ders 06](../part-2-spatial-math/06-rotations.md) ele alıyor.

## Deney

### A · 15 satır, iki küp

```xml
<mujoco>
 <option gravity="0 0 -9.81"/>
 <worldbody>
  <geom type="plane" size="2 2 .1"/>
  <geom name="table" type="box" size=".35 .25 .4"/>
  <body name="light" pos="-.13 0 .9">
   <freejoint/>
   <geom type="box" size=".1 .1 .1" mass=".1"/>
  </body>
  <body name="heavy" pos=".13 0 .9">
   <freejoint/>
   <geom type="box" size=".1 .1 .1" mass=".1"/>
  </body>
 </worldbody>
</mujoco>
```

İki adet 20 cm'lik küp; alt yüzleri masa yüzeyinin 40 cm üstünde. `inertial` yok, yani kütle özellikleri geom'lardan geliyor. Script 8. ve 12. satırları (kütleler), hava için de 2. satırı değiştiriyor:

```python
def make_xml(light_kg=0.1, heavy_kg=1.0, air=False):
    lines = list(LINES)
    lines[7] = lines[7].replace('mass=".1"', f'mass="{light_kg:g}"')
    lines[11] = lines[11].replace('mass=".1"', f'mass="{heavy_kg:g}"')
    if air:
        lines[1] = ' <option gravity="0 0 -9.81" density="1.2"/>'   # air density, kg/m³
    return "\n".join(lines)
```

**Tahmin et.** Boşlukta 1 kg'lık küp 100 g'lık küpten önce mi iner? `density="1.2"` ile hangisi gecikir ve gecikme kütleyle nasıl ölçeklenir?

**Çalıştır.** [`experiment.py`](../../experiments/01-world-as-text/experiment.py), MuJoCo 3.17.0, NumPy 2.5.4, Apple M5, deterministic (iki çalıştırma aynı çıktıyı veriyor, [`output.txt`](../../experiments/01-world-as-text/output.txt)'de kayıtlı).

**Gözlemle.** Compile edilmiş default'lar: timestep 0,002 s, density 0, viscosity 0, integrator Euler. 100 g'lık küpün çıkarılan inertia'sı her eksende 6,6667·10⁻⁴ kg·m², yani $m(a^2+a^2)/12$. Body quaternion'ı `[1 0 0 0]`.

Boşlukta iki küp de masaya **143. adımda** ulaşıyor (analitik $\sqrt{2 \cdot 0{,}40 / 9{,}81}$ = 0,2856 s). Havayla ağır küp 143. adımda, hafif olan 145. adımda iniyor; ağır olan indiğinde hafif olan **1,10 cm** yukarıda, interpolasyonla gecikme **4,08 ms**.

| hafif küp (ağır = 1 kg) | boşlukta gecikme | `density="1.2"` ile gecikme |
|---|---|---|
| 5 g | 0,00 ms | 93,44 ms |
| 10 g | 0,00 ms | 46,25 ms |
| 20 g | 0,00 ms | 22,59 ms |
| 50 g | 0,00 ms | 8,65 ms |
| 100 g | 0,00 ms | 4,08 ms |
| 200 g | 0,00 ms | 1,81 ms |
| 500 g | 0,00 ms | 0,45 ms |
| 1 kg | 0,00 ms | 0,00 ms |

Boşlukta kütle sadeleşiyor ($a = F/m = g$). Havada gecikme kabaca $1/m$ ile ölçekleniyor, çünkü $k = \frac12 \rho A$ olmak üzere drag yavaşlaması $k v^2 / m$ kütleye bölünüyor ([Fluid forces](https://mujoco.readthedocs.io/en/stable/computation/fluid.html)). Script'in yazdırdığı elle kontrol: kaybedilen mesafe $\approx (k/m)\, g^2 T^4/12$ ile 1,28 cm − 0,13 cm = **1,15 cm**; simülasyonda 1,10 cm. Bu, MuJoCo'nun drag modelinin çıktısı, gerçek bir küpün ölçümü değil.

### B · Gerçek SO-101 dosyası

```python
m = mujoco.MjModel.from_xml_path(path("robotstudio_so101", "so101.xml"))
print(m.nbody, m.njnt, m.nu, sum(m.body_mass))
```

| Büyüklük (compile edilmiş model) | Değer |
|---|---|
| body'ler (world dahil) | 9 |
| joint'ler | 6, hepsi hinge (6 dof) |
| actuator'ler | 6; gain `fixed` + bias `affine` = `<position>` servo, kp 998,22, kv 2,731 |
| geom / sensor | 50 / 0 |
| açık `<inertial>`'ı olan body'ler | 8'de 7; `camera_mount` geom'larından çıkarılıyor |
| toplam kütle | **0,6454 kg** |
| `option` | timestep 0,005 s, `implicitfast`, `elliptic` cone |

Gerçek dosya birkaç `option` default'unu eziyor (ayrıca `iterations`, `ls_iterations` ve `impratio` da ayarlı). Actuator sayıları [ders 05](05-joints-and-actuators.md)'in konusu.

## Boz

Aynı SO-101 dosyasını `inertiafromgeom="true"` ile compile et: her body'nin `<inertial>`'ı, 1000 kg/m³ (default, su) yoğunlukta geom çıkarımıyla değiştirilir:

| | dosyadaki hâli | `inertiafromgeom="true"` |
|---|---|---|
| toplam kütle | 0,6454 kg | **2,3565 kg** (×3,65) |
| shoulder kütlesi | 0,1000 kg | 0,4018 kg |
| upper_arm kütlesi | 0,1030 kg | 0,4551 kg |
| upper_arm en büyük asal inertia | 1,509·10⁻⁴ kg·m² | 9,178·10⁻⁴ kg·m² |

Hiçbir hata çıkmıyor ve model simüle olmaya devam ediyor. Mekanizma: çıkarım, default `inertiagrouprange` (0–5) içindeki her geom'u, görsel mesh'ler dahil, 1000 kg/m³ düzgün yoğunlukta dolu bir cisim sayıyor ([compiler/inertiagrouprange](https://mujoco.readthedocs.io/en/stable/XMLreference.html#compiler-inertiagrouprange), [geom/density](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-geom-density)). Böyle bir modelde yapılan gravity compensation ya da system identification aynı oranda yanlış çıkar. Yükledikten sonra her zaman `sum(m.body_mass)`'i yazdır.

## Gerçek robotta

- Menagerie SO-101 kütleleri üreticinin MJCF'inden geliyor (yukarıdaki README). Bu kitapta bir teraziyle karşılaştırılmadı; kontrol, gerçek kolun her link'ini tartmak. **Gerçek kolda henüz ölçülmedi.**
- Prosedür: monte edilmiş follower kolu sabitleme kıskacı ve kablolar olmadan tart, 0,6454 kg ile karşılaştır; simüle edilen torklara güvenmeden önce her fark (vidalar, kablolar, baskı dolgusu) açıklanmalı.
- Gerçek kolun motorları [LeRobot SO-101 rehberinde](https://huggingface.co/docs/lerobot/so101) listeleniyor: follower'da 1/345 dişli oranlı altı STS3215 servo.

## Kendini dene

1. **Kolay.** `gravity="0 0 -1.62"` yap. Boşlukta iki küp hâlâ aynı adımda mı iniyor? Önce analitik süreyi tahmin et.
2. **Orta.** Üçüncü bir küp ekle: 100 g ama 10 cm. Havada, 20 cm'lik 100 g'lık küpten hızlı mı yavaş mı? $k/m = \frac12 \rho A / m$'den akıl yürüt, sonra çalıştır.
3. **Zor.** 30 cm'lik bir capsule geom (`mass="0.2"`) ve capsule'ün inertia'sının on katı olan açık bir `<inertial>` içeren bir MJCF body yaz. `auto`, `true` ve `false` ile compile edip her biri için `body_mass` ve `body_inertia`'yı yazdır. Hangi ayar compile olmaz ve neden?

### Cevaplar

1. Evet: boşlukta kütle hiç devreye girmiyor. Analitik süre $\sqrt{2 \cdot 0{,}40 / 1{,}62}$; adım sayısını bununla kontrol et.
2. Daha hızlı: aynı kütlede ön yüz alanı dörtte bir, dolayısıyla $k/m$ dört kat küçük. Ne kadar hızlı olduğunu görmek için çalıştır.
3. `auto` açık `<inertial>`'ını korur, `true` onu capsule'den çıkarılan inertia ile değiştirir, `false` burada compile olur çünkü body'de `<inertial>` var; yalnızca `<inertial>`'ı olmayan body'ler için hata verir.

## Akılda kalanlar ve kaynaklar

- MJCF = `option` + `body` / `joint` / `geom` ağacı + `actuator` / `sensor`; yazılmayan her şey belgelenmiş bir default alır.
- Kütle özellikleri ya açık yazılır (`<inertial>`) ya da geom'lardan düzgün yoğunlukla çıkarılır; kararı `inertiafromgeom` verir ve yanlış seçim SO-101 kütlesini hiçbir hata vermeden 3,65 kat değiştirdi.
- MuJoCo quaternion'ları skaler-önce: `1 0 0 0` = birim.

Sonraki: [ders 02 · Time integration ve timestep](02-timestep.md).

**Kaynaklar**
- [MuJoCo XML reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html): `option`, `geom`, `joint`, `inertial`, `compiler/inertiafromgeom`, `inertiagrouprange`.
- [MuJoCo Modeling](https://mujoco.readthedocs.io/en/stable/modeling.html): default class'ları, frame orientation'ları, URDF extensions.
- [MuJoCo Fluid forces](https://mujoco.readthedocs.io/en/stable/computation/fluid.html).
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie): `robotstudio_so101` (Apache-2.0) ve `unitree_go2` README'leri (sabitlenmiş commit `experiments/common/menagerie.py` içinde).
- [LeRobot SO-101 rehberi](https://huggingface.co/docs/lerobot/so101).
- Sayılarımız: [`experiments/01-world-as-text/output.txt`](../../experiments/01-world-as-text/output.txt).
