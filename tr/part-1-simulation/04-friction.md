# 04 · Friction ve friction cone

<!-- meta -->
| | |
|---|---|
| **Bölüm** | [1 · Simulation](README.md) |
| **Çalışma türü** | CPU · Mac ya da Colab CPU |
| **Önkoşul** | [03](03-contact.md) |
| **Önce öğren** | [Vectors, Matrices & Operations](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/02-vectors-matrices-operations) |
| **Süre** | ~40 dk |
| **Kod** | [`experiments/04-friction/experiment.py`](../../experiments/04-friction/experiment.py) · [notebook](../../notebooks/04-friction.ipynb) |
| **Durum** | ✅ tam |
<!-- /meta -->

> Kaymaya ağırlık değil bir oran karar verir: tan θ = μ, simülasyondaki cone da buna yaklaşmak zorunda.

## Problem

Unitree'nin açık temel eğitim config'i (Go2 config'inin miras aldığı) zemin friction'ını rastgeleleştirir ama kütleyi değil: `num_envs = 4096` paralel kopyanın her biri friction'ını `friction_range = [0.5, 1.25]` aralığından çekiyor, `randomize_base_mass = False` ([`legged_robot_config.py`](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/base/legged_robot_config.py)). Kayma için doğru düğme friction'dır: bir ayağın ya da tutulan bir nesnenin kayıp kaymayacağı, nesnenin ne kadar ağır olduğuna değil, friction katsayısına ve contact kuvvetinin yönüne bağlı. Bu ders MuJoCo'da bir rampa üstündeki küpün kayma açısını ölçüyor, Coulomb yasasıyla karşılaştırıyor ve simülasyonun ondan saptığı iki yeri buluyor: eşiğin altında creep ve yöne bağlı pyramidal friction cone.

## Ana fikir

### Sade anlatımla

Friction bir bütçedir. Bir yüzey bir cismi yana doğru itebilir, ama yalnızca bir sınıra kadar; bu sınır da cismin yüzeye ne kadar bastırdığıyla orantılıdır. Orantı katsayısı friction katsayısı μ'dür.

Bir kitabın üstüne telefon koy ve kitabı yavaşça eğ. Eğmek aynı anda iki şey yapar: telefonun ağırlığının daha azı kitaba bastırır (bütçe küçülür), daha fazlası telefonu aşağı çeker (talep büyür). Bir açıda talep bütçeyi aşar ve telefon kayar. Ağır bir telefon daha sert bastırır, ama tam aynı oranda daha sert de çekilir; ağırlık sadeleşir: kayma açısı yalnızca μ'ye bağlıdır.

Bir simülatör bu bütçeyi temsil etmek zorundadır. Tam kural, yan kuvvetin aynı sınıra kadar her yöne bakabileceğini söyler; yani izin verilen kuvvetler yuvarlak bir koni oluşturur. MuJoCo'nun default'u bu yuvarlak koniyi bir piramitle yaklaşıklar; piramidin düz yüzleri sınırı yöne bağımlı yapar. Ayrıca MuJoCo'nun contact'ı yumuşak olduğu için ([ders 03](03-contact.md)) friction da yumuşaktır: sınırın altında bile bir cisim yavaşça creep yapabilir.

```mermaid
flowchart TD
    W["ağırlık m·g"] --> N["rampaya bastıran<br/>m·g·cos θ"]
    W --> P["aşağı çeken<br/>m·g·sin θ"]
    N --> B["friction bütçesi<br/>μ · m·g·cos θ"]
    B --> C{"çekiş bütçeden<br/>büyük mü?"}
    P --> C
    C -- "hayır" --> H["küp durur"]
    C -- "evet" --> S["küp kayar"]
```

### Kayma testi, kod olarak

```python
pull = m * g * sin(theta)                  # gravity component along the ramp (N)
budget = mu * m * g * cos(theta)           # largest friction force the surface can supply (N)
slips = pull > budget                      # m and g cancel: tan(theta) > mu (Build it, Step 1)

def in_elliptic_cone(f_n, f_t1, f_t2, mu):  # MuJoCo's round cone, condim 3
    return f_n >= 0 and f_t1**2 + f_t2**2 <= (mu * f_n)**2
```

| Büyüklük | Formül |
|---|---|
| küpü aşağı çeken kuvvet | $m g \sin\theta$ |
| friction limiti | $\mu\, m g \cos\theta$ |
| kayma koşulu | $\tan\theta > \mu$ |
| kayarken ivme | $a = g(\sin\theta - \mu\cos\theta)$ |
| elliptic cone | $f_1 \ge 0,\ f_1^2 \ge \sum_i f_i^2/\mu_{i-1}^2$ |

$m$ küpün kütlesi, $g$ yerçekimi, $\theta$ rampa açısı, $\mu$ kayma friction katsayısıdır; $f_1$ yüzey normal'i boyunca contact kuvveti, $f_2, f_3$ (daha yüksek `condim` ile daha fazlası) ise her biri kendi katsayısı $\mu_{i-1}$ ile sınırlanan teğetsel bileşenlerdir.

<details>
<summary>Matematik ayrıntısı: rampada Coulomb yasası ve simüle edilen cone neden sapar</summary>

**Kayma koşulu.** Statik friction en fazla $\mu N$ sağlayabilir; rampada normal kuvvet $N = m g\cos\theta$'dır. Küp $m g\sin\theta \le \mu\, m g\cos\theta$ olduğu sürece durur. $m g\cos\theta$'ya bölmek $\tan\theta \le \mu$ verir: kütle ve yerçekimi sadeleşir ([OpenStax University Physics 1, §6.2](https://openstax.org/books/university-physics-volume-1/pages/6-2-friction)). Eşiğin üstünde net aşağı yönlü kuvvet $m g(\sin\theta - \mu\cos\theta)$'dır; tablodaki ivme buradan gelir.

**Cone'lar.** Bir contact kuvveti $f$ friction cone'un içindeyse kabul edilebilirdir ([Computation · Contact](https://mujoco.readthedocs.io/en/stable/computation/index.html#cocontact)). Elliptic cone tablodaki eşitsizliktir. Pyramidal cone, $n$ = `condim` olmak üzere $2(n-1)$ negatif olmayan kenar kuvvetinden kurulur; `condim=3` için 4 kenar. Piramidin yüzleri düzdür; kesiti çember değil çokgendir ve izin verdiği friction limiti, kaymanın contact frame'ine göre yönüne bağlıdır. Doküman elliptic cone'u fiziksel gerçekliğin daha iyi modeli olarak tanımlıyor; pyramidal cone ise bazen daha hızlı ve sağlam ([option/cone](https://mujoco.readthedocs.io/en/stable/XMLreference.html#option)).

**Creep.** Friction constraint'i de normal constraint gibi yumuşak çözülür; bu yüzden eşiğin altında küçük, sabit bir kayma hızına izin verilir. Doküman buna slow slippage diyor ve iki önlem sayıyor: `impratio` > 1 ile elliptic cone'lar ya da `noslip` solver'ı ([Modeling · Slow slippage](https://mujoco.readthedocs.io/en/stable/modeling.html#cslowslippage)).

**Tek μ.** `geom/friction` üç sayı tutar (kayma, burulma, yuvarlanma; default `1 0.005 0.0001`). Tek bir kayma katsayısı var, yani ayrı statik ve kinetik friction yok. Bir contact çiftinde, bir geom'un `priority`'si yüksek değilse eleman bazında maksimum kullanılır ([geom/friction](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-geom-friction), [Contact parameters](https://mujoco.readthedocs.io/en/stable/modeling.html#ccontact)).

Daha derine: [Vectors, Matrices & Operations](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/02-vectors-matrices-operations) (AIEFS: bir kuvveti bileşenlerine ayırmak); [OpenStax University Physics 1, §6.2](https://openstax.org/books/university-physics-volume-1/pages/6-2-friction); [MuJoCo Computation · Contact](https://mujoco.readthedocs.io/en/stable/computation/index.html#cocontact).

</details>

### MuJoCo'da friction cone'lar

| `option/cone` | Tanım | `condim=3` contact başına satır |
|---|---|---|
| `pyramidal` (default) | bir piramidin $2(n-1)$ negatif olmayan kenar kuvveti | 4 |
| `elliptic` | $f_1 \ge 0,\ f_1^2 \ge \sum f_i^2/\mu_{i-1}^2$ | 3 |

*Satır*, constraint solver'ın o contact için bulması gereken tek bir skaler bilinmeyendir.

<!-- figure: 04-friction-slope -->

## Sıfırdan yaz

Şimdi Coulomb kayma testini, kayma ivmesini ve MuJoCo'nun pyramidal cone'unu yaklaşık 40 satır NumPy ile yazacak; MuJoCo'nun hareketiyle, constraint satırlarıyla ve bir SciPy lineer programıyla karşılaştıracaksın. Adımları kendi dosyana yaz ve her adımdan sonra çalıştır. Bitmiş fonksiyonlar [`physai/sim.py`](../../physai/sim.py) içinde; `uv run python experiments/04-friction/build.py` dört adımı kontrolleriyle birlikte koşturur (çıktı [`build_output.txt`](../../experiments/04-friction/build_output.txt) dosyasında).

### Adım 1 · Kayma testi

```python
def ramp_slip(theta, mu, g=9.81):
    """Does a block on a ramp at angle theta slip (tan theta > mu), and its downhill acceleration if so."""
    slips = np.tan(theta) > mu
    return slips, g * (np.sin(theta) - mu * np.cos(theta)) if slips else 0.0
```

Bunu Deney'in 0,05°'lik ızgarasında uygula, sonra Deney'in kendi kriterini (2 s sonunda hızlı ve ivmelenen) bu ideal bloğa uygula; blok durgunluktan sabit ivmeyle başladığı için 0,5–1,0 s ve 1,5–2,0 s'deki ortalama hızları $0{,}75a$ ve $1{,}75a$:

```text
== Step 1. Slip angle: tan(theta) > mu, then the Experiment's criterion on an ideal Coulomb block ==
  mu  arctan mu  first grid angle that slips  ideal block + criterion
 0.3     16.70°                       16.70°                   16.80°
 0.5     26.57°                       26.60°                   26.65°
 0.8     38.66°                       38.70°                   38.75°
 1.0     45.00°                       45.05°                   45.05°
```

Deney 16,75°, 26,60°, 38,70° ve 44,90° ölçtü: ideal bloğun kriteri geçtiği açıdan bir ızgara adımı *önce* (μ = 1'de üç). $\arctan\mu$'nün hemen üstünde ideal blok o kadar yavaş ivmeleniyor ki 2 s sonunda hâlâ 2 cm/s'nin altında; MuJoCo küpü buna creep hızını (Gözlemle'de 1,42 cm/s'ye kadar) ekliyor.

### Adım 2 · Kayma ivmesi

Eşiğin üstünde blok $g(\sin\theta - \mu\cos\theta)$ ile ivmelenir. Deney'in 80 cm'lik rampası bunu uzun süre ölçmek için kısa; onun yerine yerçekimini eğ: yerçekimi $(g\sin\theta, 0, -g\cos\theta)$ olan düz zemindeki bir küp, kayıp düşeceği bir kenar olmadan aynı fizik. $x(t)$'ye bir parabol oturt:

```text
== Step 2. Sliding acceleration (mu = 0.5): g (sin - mu cos) vs MuJoCo, fit over 0.2-1.0 s ==
 theta  ours (m/s^2)  elliptic  pyramidal
 28.0°        0.2747    0.2744     0.2745
 30.0°        0.6571    0.6571     0.6567
 35.0°        1.6088    1.6207     1.6117
 45.0°        3.4684    3.4640     3.5092
```

Her açıda, iki cone için de %1,2'nin içinde. Kalan fark formülden gelmiyor: kayan bir box ön kenarına devrilip zıplıyor (zaman zaman hiç contact'ı yok), fit de bu tümsekli kaymanın ortalamasını alıyor.

### Adım 3 · Constraint satırları olarak pyramid

`condim=3` için pyramidal cone her contact'ı $n \pm \mu t_1$ ve $n \pm \mu t_2$ boyunca dört kenar kuvvetiyle $f_k \ge 0$ değiştirir. Constraint biçiminde her kenar bir $J_n \pm \mu J_{t_i}$ satırıdır, contact kuvveti de onların toplamı:

```python
def pyramid_rows(j_normal, j_tangent, mu):
    """Constraint rows of MuJoCo's pyramidal cone: J_n + mu J_t and J_n - mu J_t for each tangent direction."""
    return np.array([j_normal + s * m * jt for m, jt in zip(mu, j_tangent) for s in (1, -1)])
```

```python
def pyramid_to_force(edge_forces, mu):
    """Contact-frame force (normal, tangential...) from the non-negative pyramid edge forces."""
    f = np.asarray(edge_forces, float).reshape(-1, 2)
    return np.r_[f.sum(), [m * (fp - fm) for m, (fp, fm) in zip(mu, f)]]
```

$J_n$, $J_t$'yi contact frame'inden (`d.contact[i].frame`, önce normal) ve küpün contact noktasındaki nokta Jacobian'ından (`mj_jac`) kur, sonra MuJoCo'nun `efc_J`'si ve `mj_contactForce` ile karşılaştır:

```text
== Step 3. Pyramid rows and edge forces vs MuJoCo (cube holding at 20°, mu = 0.5, pyramidal) ==
4 contacts x 4 rows = 16 rows · max |our rows - efc_J| = 0.0e+00
max |force from our edge sum - mj_contactForce| = 0.0e+00 N
```

Birebir: MuJoCo pyramidal bir contact'ı böyle kuruyor.

### Adım 4 · Ne kadar sürtünme, hangi yönde

Birim normal kuvvetle negatif olmayan kenar kuvvetleri $|f_1| + |f_2| \le \mu$ olan her teğet kuvvete ulaşır: çember değil, bir eşkenar dörtgen. Birinci teğetle $\phi$ açısı yapan bir yönde limit bu yüzden $\mu / (|\cos\phi| + |\sin\phi|)$:

```python
def tangential_limit(phi, mu, cone):
    """Largest friction force per unit normal force when sliding at angle phi to the first tangent axis."""
    if cone == "elliptic":
        return mu
    return mu / (abs(np.cos(phi)) + abs(np.sin(phi)))  # pyramid: |f1| + |f2| <= mu f_n, a diamond
```

```text
== Step 4. Largest friction force per unit normal force, by sliding direction (mu = 0.5) ==
  phi  elliptic  pyramid (ours)  pyramid (linprog)
   0°    0.5000          0.5000             0.5000
  15°    0.5000          0.4082             0.4082
  30°    0.5000          0.3660             0.3660
  45°    0.5000          0.3536             0.3536
MuJoCo: friction impulse / normal impulse on the cube over 0.2-1.0 s (world frame)
  holding at 20° (tan 20° = 0.3640)  elliptic  0.3640
  holding at 20° (tan 20° = 0.3640)  pyramidal 0.3640
  sliding at 35° (mu = 0.5)          elliptic  0.5000
  sliding at 35° (mu = 0.5)          pyramidal 0.5000
```

Dört kenar kuvveti üzerindeki SciPy lineer programı her açıda aynı sonucu veriyor. Köşegen boyunca pyramid yalnızca $\mu/\sqrt2$'ye izin veriyor: 0,5 yerine 0,3536. MuJoCo'nun kendi kuvvetleri yasaya uyuyor: duran küp tam ihtiyacı kadar sürtünme kullanıyor, $\tan 20°$ = 0,3640; kayan küp cone'un sınırında, oran μ. Buradaki iki küp de bir teğet ekseni boyunca kayıyor ($\phi = 0$), iki cone'un uyuştuğu yerde. Boz'daki döndürülmüş küpün farklı bir $\phi$'de kayıp kaymadığı ayrıca incelenmedi.

### Elinde kalan

`physai.sim` artık `ramp_slip`, `pyramid_rows`, `pyramid_to_force` ve `tangential_limit` içeriyor. Ders 16, gripper bir nesneyi sıkarken sürtünme limitine ihtiyaç duyuyor. [`tests/test_04_friction.py`](../../tests/test_04_friction.py) bunları MuJoCo'ya sabitliyor; bir şey değiştirdikten sonra `uv run pytest tests/test_04_friction.py` çalıştır.

## Kütüphaneyle kullan

MuJoCo'da sürtünme geom başına üç sayı artı iki global seçenektir:

```python
<option cone="elliptic" impratio="10"/>          # round cone; impratio > 1 reduces creep (Math details)
<geom ... friction="1 0.005 0.0001" condim="3"/>  # sliding, torsional, rolling; condim picks which act

f = np.zeros(6)
mujoco.mj_contactForce(m, d, i, f)               # contact i: f[0] normal, f[1:3] friction, in its frame
f_world = d.contact[i].frame.reshape(3, 3).T @ f[:3]
```

Tuzaklar:

- **Dünya değil, contact frame'i.** `mj_contactForce` kuvveti contact frame'inde, önce normal olarak döndürür; contact'ları toplamadan ya da yerçekimiyle karşılaştırmadan önce `frame.T` ile döndür (Adım 4).
- **İki cisim için tek μ.** `priority` karar vermiyorsa MuJoCo iki geom'un `friction`'ının eleman bazında maksimumunu alır (Matematik ayrıntısı): μ = 1'lik bir gripper pad'indeki kaygan bir nesne μ = 1 alır.
- **Contact değil, satır.** `d.nefc` satırları sayar: pyramidal `condim=3` contact başına dört, elliptic başına üç (Adım 3 4 contact için 16 satır buldu).

## Deney

80 × 30 × 2 cm'lik bir rampaya hizalı ve durgun yerleştirilmiş 8 cm'lik bir küp; iki geom da aynı μ'yü alıyor.

```python
def xml(theta_deg, mu, mass, cone="pyramidal"):
    return f"""
<mujoco>
  <option cone="{cone}"/>
  <worldbody>
    <body name="ramp" euler="0 {-theta_deg} 0">
      <geom type="box" size=".40 .15 .01" friction="{mu} 0.005 0.0001"/>
    </body>
    <body name="cube">
      <freejoint/>
      <geom type="box" size=".04 .04 .04" mass="{mass}" friction="{mu} 0.005 0.0001"/>
    </body>
  </worldbody>
</mujoco>"""
```

2 s simüle ediliyor; $v_1$ ve $v_2$, 0,5–1,0 s ve 1,5–2,0 s aralıklarındaki ortalama aşağı yönlü hızlar. Creep (eşiğin altında yavaş kayma, bkz. Matematik ayrıntısı) yüzünden "kayma" iki koşul istiyor:

```python
def slips(v1, v2):
    return v2 > 2.0 and (v2 - v1) > 0.5      # cm/s: fast AND accelerating
```

Kayma açısı, kendisinden büyük her açıda da kaymanın olduğu en küçük açı; grid 0,05°, $\arctan\mu$'nün 3° altından 1° üstüne kadar.

**Tahmin et.** 10 kg'lık bir küp 0,1 kg'lıktan daha büyük bir açıda mı kayar?

**Çalıştır.** [`experiment.py`](../../experiments/04-friction/experiment.py), MuJoCo 3.17.0, Apple M5, `dt` = 2 ms, deterministic; çıktı [`output.txt`](../../experiments/04-friction/output.txt)'de.

| μ | $\arctan\mu$ | 0,1 kg | 1 kg | 10 kg |
|---|---|---|---|---|
| 0,3 | 16,70° | 16,75° | 16,75° | 16,75° |
| 0,5 | 26,57° | 26,60° | 26,60° | 26,60° |
| 0,8 | 38,66° | 38,70° | 38,70° | 38,70° |

**Gözlemle.**
- Kütlenin etkisi yok: bütün sweep boyunca 0,1 kg ve 10 kg'lık küplerin hız dizileri en fazla 2,1·10⁻¹² cm/s farklı (yuvarlama).
- Ölçülen açı teoriden en fazla bir grid adımı uzakta (en büyük sapma +0,05°). μ = 1 için script 45,00°'ye karşı 44,90° veriyor.
- **Creep.** Eşiğin en az 1° altında küp hâlâ sabit hızla hareket ediyor: μ = 0,3, 0,5, 0,8 için 0,30, 0,60 ve 1,42 cm/s'ye kadar; $v_2 - v_1$ ise 0,05 cm/s'nin altında kalıyor. Yalnızca hıza bakan bir kriterle ($v_2$ > 2 cm/s) "kayma açısı" μ = 0,3 ve 0,5 için 16,35°'ye (−0,35°) ve 26,25°'ye (−0,32°) düşüyor. O kriterin yakaladığı şey ivmelenme değil, creep.
- Elliptic cone hizalı küp için aynı açıları veriyor (16,75°, 26,60°, 38,70°).

## Boz

Küpü rampa normali etrafında 45° döndür. Coulomb friction izotroptur, hiçbir şey değişmemeli:

| μ | cone | kayma açısı | teoriye göre | eşiğin ≥ 0,5° altında en yüksek sabit hız |
|---|---|---|---|---|
| 0,5 | pyramidal | 26,30° | −0,27° | 7,40 cm/s |
| 0,5 | elliptic | 26,60° | +0,03° | 0,18 cm/s |
| 0,8 | pyramidal | 37,70° | −0,96° | 27,24 cm/s |
| 0,8 | elliptic | 38,60° | −0,06° | 0,34 cm/s |

Default pyramidal cone'da döndürülmüş küp μ = 0,8'de neredeyse 1° erken kayıyor ve durması gereken yerde 27,24 cm/s'ye kadar kayıyor. Elliptic cone bir iki grid adımı içinde kalıyor. Makul bir mekanizma, piramidin kenarlarının contact frame'ine bağlı olması ve böylece izin verilen teğetsel kuvvetin yöne bağlı hâle gelmesi; ama bu burada ayrıştırılmadı (box–box contact noktalarının yerleşimi de rotation'la değişiyor). Bunu ölçülmüş bir uyarı olarak al: **friction limitleri önemliyse (grasping, ayak kayması) `cone="elliptic"` kullan ya da bir sweep'le doğrula.**

Bu sayıların arkasındaki hız eğrileri, μ = 0,8 için, iki sapmayı birlikte gösteriyor ([`plot.py`](../../experiments/04-friction/plot.py) aynı `sweep()`'i yeniden çalıştırıyor; açı başına hızlar `output.txt`'de yok):

![μ = 0,8 için kararlı aşağı yönlü hızın rampa açısına göre log eksenli grafiği. Hizalı küp pyramidal cone'la eşiğin altında sabit bir hızla creep yapıyor (1,42 cm/s'ye kadar) ve 38,70°'den itibaren ivmeleniyor; 45° döndürülmüş küp pyramidal cone'la eşiğin çok altında bile birkaç cm/s ile kayıyor; döndürülmüş küp elliptic cone'la en yavaş kalıyor ve hemen altındaki tek bir açı dışında 38,60°'den itibaren kayıyor.](../../experiments/04-friction/speed_vs_angle.png)

## Gerçek robotta

- Menagerie SO-101 zaten `cone="elliptic"` ve `impratio="10"` kullanıyor, yani yukarıdaki slow slippage önlemini. Gripper collision geom'ları `friction="1 5e-3 5e-4"`, `condim="6"` ve `priority="1"` kullanıyor, yani onların μ = 1 değeri nesneninkini eziyor ([so101.xml](https://github.com/google-deepmind/mujoco_menagerie/blob/f3b6013/robotstudio_so101/so101.xml)). μ = 1'in baskı pad'lerle belirli bir nesne arasında tutup tutmadığı **gerçek kolda henüz ölçülmedi**.
- Prosedür (bu deneyle aynı): nesneyi pad malzemesinden bir rampaya koy, yavaşça eğ, kayma açısını 5 kez kaydet, μ = tan θ. Gerçek yüzeylerde statik katsayı kinetikten büyüktür; MuJoCo'nun tek μ'sü bu farkı temsil edemez.

## Kendini dene

1. **Kolay.** `mass`'ı 1000 kg yap. μ = 0,5'teki kayma açısını tahmin et, sonra düzenleyip `--quick` ile çalıştır.
2. **Orta.** Rampaya μ = 0,3, küpe μ = 0,8 ver. Hangi açıda kayar? Sonra rampa geom'una `priority="1"` ekle.
3. **Zor.** μ ∈ [0,5, 1,25] ile eğitilmiş policy için rampadaki kayma açısı aralığını hesapla. Sonra döndürülmüş küp ve iki cone ile pyramidal sonucun μ = 1,25'te ne kadar sapabileceğini ölç. [Ders 39 · Domain randomization](../part-9-sim-to-real/39-domain-randomization.md) için not al.

### Cevaplar

1. Değişmez, bu düzenekte 26,60°: kütle sadeleşir.
2. Priority olmadan MuJoCo büyük μ'yü, 0,8'i kullanır, yani yaklaşık 38,7° bekle; rampada priority varken μ = 0,3, yaklaşık 16,75°. Doğrulamak için çalıştır.
3. Teoride $\arctan 0{,}5$'ten (26,57°, script yazdırıyor) $\arctan 1{,}25$'e; döndürülmüş küpün μ = 1,25'teki sapması tahminle değil ölçümle bulunmalı.

## Akılda kalanlar ve kaynaklar

- tan θ = μ, kütleden bağımsız; kayma kriteri ivmelenme istediğinde MuJoCo bunu μ = 0,3–0,8 için 0,05° içinde üretiyor (μ = 1'de 45,00°'ye karşı 44,90°).
- Soft contact eşiğin altında creep yapıyor (burada 1,42 cm/s'ye kadar); yalnızca hıza bakan kriter creep'i kayma sanıyor.
- Default pyramidal cone bu testte yöne bağlı (45°'de −0,96°); elliptic değil.

Sonraki: [ders 05 · Joint'ler ve actuator'ler](05-joints-and-actuators.md).

**Kaynaklar**
- [OpenStax, University Physics Vol. 1, §6.2 Friction](https://openstax.org/books/university-physics-volume-1/pages/6-2-friction).
- [MuJoCo XML reference · option/cone, geom/friction](https://mujoco.readthedocs.io/en/stable/XMLreference.html); [Computation · Contact](https://mujoco.readthedocs.io/en/stable/computation/index.html#cocontact); [Modeling · Contact parameters](https://mujoco.readthedocs.io/en/stable/modeling.html#ccontact).
- [`unitree_rl_gym` · `legged_robot_config.py`](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/base/legged_robot_config.py): `num_envs = 4096`, `randomize_friction = True`, `friction_range = [0.5, 1.25]` (2026-10-27'de okundu).
- [Menagerie · robotstudio_so101](https://github.com/google-deepmind/mujoco_menagerie/tree/f3b6013/robotstudio_so101).
- Sayılarımız: [`experiments/04-friction/output.txt`](../../experiments/04-friction/output.txt).
