# 04 · Friction: ağır küp daha geç kayar mı?

> Ağır olan daha sıkı tutar, ama o kadar da çekilir. Kütle sadeleşir: tan θ = μ.

| | |
|---|---|
| **Faz** | [0 · Dünya kare kare](README.md) |
| **Tür** | Deney |
| **Önkoşul** | [01](01-world-as-text.md), [02](02-timestep.md); [03](03-soft-contact.md) faydalı ama şart değil |
| **Süre** | ~40 dk |
| **Araç** | MuJoCo 3.17 · Python · Mac ya da Colab CPU |
| **Kod** | [`experiments/04-friction/experiment.py`](../../experiments/04-friction/experiment.py) · [notebook](../../notebooks/04-friction.ipynb) |
| **Durum** | ✅ tam |

## Neden önemli

Bir robot köpek yürümeyi öğrenirken kaç farklı zemin görür? Unitree'nin açık eğitim kodunda (`unitree_rl_gym`, ETH Zürih'in `legged_gym`'i üzerine kurulu) cevap açıkça yazıyor: **4.096** paralel environment aynı anda eğitiliyor (`num_envs = 4096`) ve her environment'ın zemin sürtünmesi **0,5 ile 1,25** arasında rastgele seçiliyor (`randomize_friction = True`, `friction_range = [0.5, 1.25]`). Go2 config'i bu değerleri değiştirmiyor ([`legged_robot_config.py`](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/base/legged_robot_config.py), [legged_gym'deki aynı satırlar](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot_config.py); 27.10.2026'da okundu).

Neden ağırlık değil de friction coefficient rastgele? Çünkü bir ayağın kayıp kaymayacağını belirleyen asıl sayı μ. Bir robot elinin bir kutuyu düşürmeden tutması da aynı sınıra dayanır: contact force **friction cone**'un içinde kalmalı (Lynch & Park, [*Modern Robotics*](https://hades.mech.northwestern.edu/index.php/Modern_Robotics), Bölüm 12). Bu derste o sınırı bir rampada ölçeceksin ve sezginin seni yanıltacağı bir yeri göreceksin: 100 kat ağır küp, hafif olanla aynı açıda kayıyor.

## Kavram

### Sürtünmenin bir tavanı var

Coulomb modeline göre iki yüzey arasındaki statik sürtünme kuvveti, bastırma kuvveti N ile orantılı bir tavanı aşamaz: **f ≤ μ·N** ([OpenStax *University Physics* 1, §6.2](https://openstax.org/books/university-physics-volume-1/pages/6-2-friction)). Çekiş bu tavanı aşınca cisim kayar.

### Rampada iki kuvvet

θ açılı bir rampada m kütleli küp için:

| | formül |
|---|---|
| Küpü yokuş aşağı çeken (ağırlığın rampa boyunca bileşeni) | m · g · sin θ |
| Sürtünmenin tavanı (μ × bastırma kuvveti) | μ · m · g · cos θ |

Kayma eşiğinde ikisi eşit:

```
m · g · sin θ = μ · m · g · cos θ   ⇒   tan θ = μ
```

İki tarafta da m var. **Kütle sadeleşir.** Ağır küp daha sıkı tutulur, ama tam o oranda daha güçlü çekilir. Kayma açısı yalnız μ'ye bağlıdır.

### MuJoCo sürtünmeyi nasıl hesaplıyor

- Her `geom`'un `friction` attribute'u üç sayıdır: sliding, torsional, rolling. İlki μ ([XML referansı · geom](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-geom)).
- İki geom temas edince MuJoCo, `priority` verilmemişse, iki μ'den **büyüğünü** kullanır. Biz ikisine aynı μ'yü verdik.
- Contact force bir friction cone ile sınırlanır. Default `cone="pyramidal"` koniyi bir piramitle yaklaşık alır; `elliptic` gerçek koniye daha yakındır ([XML referansı · option](https://mujoco.readthedocs.io/en/stable/XMLreference.html#option), [Computation · Contact](https://mujoco.readthedocs.io/en/stable/computation/index.html)).
- MuJoCo'da tek bir μ var. Gerçek dünyadaki "statik sürtünme kinetikten büyüktür" ayrımı bu modelde yok.

## Kendin kur

### Adım 1 · Sahne: eğik rampa + 8 cm küp

```python
HALF = 0.04                    # cube half-size -> 8 cm cube
RAMP = (0.40, 0.15, 0.01)      # ramp half-sizes -> 80 × 30 × 2 cm

def xml(theta_deg, mu, mass, cone="pyramidal"):
    return f"""
<mujoco>
  <option cone="{cone}"/>
  <worldbody>
    <body name="ramp" euler="0 {-theta_deg} 0">
      <geom type="box" size="{RAMP[0]} {RAMP[1]} {RAMP[2]}" friction="{mu} 0.005 0.0001"/>
    </body>
    <body name="cube">
      <freejoint/>
      <geom type="box" size="{HALF} {HALF} {HALF}" mass="{mass}" friction="{mu} 0.005 0.0001"/>
    </body>
  </worldbody>
</mujoco>"""
```

Küp, rampanın yüzeyine değecek şekilde ve rampayla aynı yönelimde, **duran hâlde** konur. Sonra 2 sn simülasyon koşar ve küpün rampa boyunca aşağı hızı kaydedilir.

### Adım 2 · "Kayıyor" ne demek?

Burada ilk tuzak var. MuJoCo'nun contact'ı soft (bkz. [ders 03](03-soft-contact.md)): eşiğin altında bile küp çok yavaş, **sabit hızla** kayar; buna creep denir. Bu deneyde eşiğin en az 1° altında ölçülen en büyük creep hızları: μ 0,3 → 0,3 cm/s, μ 0,5 → 0,6 cm/s, μ 0,8 → 1,42 cm/s.

Gerçek kayma ise **hızlanır**: a = g·(sin θ − μ·cos θ). O yüzden iki şart birden aranır:

```python
V_TH, A_TH = 2.0, 0.5          # cm/s: slip criterion

def slips(theta_deg, mu, mass, cone="pyramidal"):
    """Slipping <=> both fast (v2 > 2 cm/s) and accelerating (v2 - v1 > 0.5 cm/s)."""
    v1, v2 = speeds(theta_deg, mu, mass, cone)   # mean downhill speed over 0.5–1.0 s and 1.5–2.0 s
    return v2 > V_TH and (v2 - v1) > A_TH
```

Yalnız hız eşiği (v2 > 2 cm/s) kullanılsaydı kayma açısı pyramidal cone'da μ 0,3'te 16,35°, μ 0,5'te 26,25° çıkardı: teorinin ~0,3° altında, çünkü arctan μ'nün hemen altında küp hızlanmadan sabit 2–3 cm/s ile süzülüyor.

### Adım 3 · Sweep

**Kayma açısı:** ölçütü geçen ve üstündeki bütün açıların da geçtiği en küçük açı. Grid 0,05°, her μ için teorik değerin 3° altından 1° üstüne (~80 açı). 3 μ × 3 kütle = 9 sweep.

```bash
python experiments/04-friction/experiment.py      # ~10 s on Apple M5
python experiments/04-friction/experiment.py --quick   # μ = 0.5 only
```

## Sayılarla sonuç

MuJoCo 3.17.0, 2 ms timestep, default pyramidal cone, deterministik.

| μ | teori arctan μ | 0,1 kg | 1 kg | 10 kg |
|---|---|---|---|---|
| 0,3 | 16,70° | 16,75° | 16,75° | 16,75° |
| 0,5 | 26,57° | 26,60° | 26,60° | 26,60° |
| 0,8 | 38,66° | 38,70° | 38,70° | 38,70° |

- Her satırda üç kütle **aynı** açıyı veriyor (yayılım 0,000°). 100 kat ağır küp (0,1 → 10 kg) aynı açıda kayıyor.
- Ölçülen değer teoriden en fazla **+0,05°** sapıyor, yani grid adımı kadar.
- Açıyı yalnız μ değiştiriyor.

Statik sweep'te 0,1 kg ve 10 kg küplerin hız dizileri arasındaki fark 1,2·10⁻¹⁵ cm/s mertebesinde: sayısal yuvarlama düzeyi.

## Tuzak: friction cone yöne bağlı olabilir

Videodaki ve yukarıdaki bütün koşularda küp rampayla hizalı, döndürülmemiş. Küp rampa normali etrafında **45° döndürülünce** pyramidal cone'da kayma erken ve düzensiz geliyor: μ 0,5 → 26,25° (−0,32°), μ 0,8 → 37,70° (−0,96°), eşiğin altında 4–16 cm/s'lik sabit kaymalar görülüyor ve ölçüt monoton değil. Elliptic cone'da döndürülmemiş küp aynı açıları veriyor.

Bu farkın mekanizması (piramidin yön bağımlılığı mı, box-box contact noktaları mı) incelenmedi. **Doğrulanmadı**; burada yalnız gözlem olarak yazılıyor. Notebook'un 5. bölümünde pyramidal ve elliptic cone'u yan yana koşturabilirsin.

## Sınırlar

- Tek μ modeli: gerçek yüzeylerde statik sürtünme kinetikten büyüktür ve "birden kayma" hissinin bir kısmı oradan gelir. Bu simülasyonda o etki yok.
- Gerçek malzemelerin μ değerleri (tahta-tahta, kauçuk-beton vb.) bu derste bilerek verilmedi; kaynaklar arasında geniş farklar var.
- 8 cm küpte devrilme sınırı da tan θ = 1 (45°). μ ≥ 1'de küpün kayacağı mı devrileceği mi ayrıca ölçülmedi.

## Alıştırmalar

1. **Kolay.** μ = 1 olsa küp kaç derecede kayar? Önce tahmin et, sonra notebook'un 6. bölümünde `MU = 1.0` ile koştur. (Cevap sayfanın sonunda.)
2. **Kolay.** `MASS = 1000` yap. Tablo değişiyor mu?
3. **Orta.** Ölçütü yalnız hız eşiğine (v2 > 2 cm/s) indir ve tabloyu yeniden üret. Hangi satırlar kayıyor ve neden?
4. **Orta.** Rampa ile küpe farklı μ ver (ör. rampa 0,3, küp 0,8). Hangi açıda kayıyor? MuJoCo'nun "büyüğü seç" kuralıyla açıkla. Sonra bir geom'a `priority="1"` ekle ve tekrar ölç.
5. **Zor.** Unitree'nin eğitim aralığı μ ∈ [0,5 · 1,25]. Bu aralıktaki en kaygan ve en tutucu zemin için kayma açılarını hesapla. Aralığın altındaki bir zemin (ör. μ = 0,2) eğitimde hiç görülmediyse, böyle bir zeminde policy'den ne beklersin? Bu soruyu [ders 37 · domain randomization](../phase-7-sim-to-real/37-domain-randomization.md) için not al.

## Terimler

| Terim | Sık duyulan | Aslında |
|---|---|---|
| μ (friction coefficient) | "Yüzeyin pürüzlülüğü" | Sürtünme tavanının bastırma kuvvetine oranı. Kayma açısının tanjantı |
| Friction cone | — | Contact force'un sürtünmeyle karşılanabilecek yönlerinin kümesi. Kuvvet koninin dışına çıkınca kayma başlar |
| Pyramidal / elliptic cone | — | MuJoCo'nun koniyi yaklaşık almasının iki yolu. Default pyramidal |
| Creep | "Simülasyon hatası" | Soft contact yüzünden eşiğin altında görülen sabit, yavaş kayma |
| `friction_range` | — | RL eğitiminde her environment'ın sürtünmesinin seçildiği rastgele aralık (domain randomization) |

## Kaynaklar

- [OpenStax, *University Physics* Cilt 1, §6.2 Friction](https://openstax.org/books/university-physics-volume-1/pages/6-2-friction): Coulomb sürtünmesi ve eğik düzlem.
- [MuJoCo XML referansı · `geom/friction`, `option/cone`](https://mujoco.readthedocs.io/en/stable/XMLreference.html) ve [Computation · Contact](https://mujoco.readthedocs.io/en/stable/computation/index.html).
- [`unitree_rl_gym` · `legged_robot_config.py`](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/base/legged_robot_config.py) ve [`legged_gym`](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot_config.py): `num_envs = 4096`, `randomize_friction = True`, `friction_range = [0.5, 1.25]`.
- Lynch & Park, [*Modern Robotics*](https://hades.mech.northwestern.edu/index.php/Modern_Robotics), Bölüm 12 (Grasping and Manipulation): friction cone ve grasping.
- Bu sayfadaki her sayı `experiment.py` ve notebook'un koşusundan (MuJoCo 3.17.0, 27.10.2026). Koddan Kasa video ekibinin iç doğrulama notuyla birebir aynı.

## Alıştırma cevapları

1. Teori: arctan 1 = **45°**. Notebook **44,90°** ölçüyor (−0,10°). Bu derste küp kayıyor, devrilmiyor; ama 8 cm küpte devrilme sınırı da 45° olduğu için bu ayrı ölçülmedi.
