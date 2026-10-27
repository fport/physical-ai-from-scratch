# 01 · Dünya bir metin dosyası

> Masa, iki küp ve yerçekimi: 15 satır yazı. Gerisini physics engine adım adım çözer.

| | |
|---|---|
| **Faz** | [0 · Dünya kare kare](README.md) |
| **Tür** | Deney |
| **Önkoşul** | Yok. Python okuyabilmek yeter |
| **Süre** | ~30 dk |
| **Araç** | MuJoCo 3.17 · Python · Mac ya da Colab CPU |
| **Kod** | [`experiments/01-world-as-text/experiment.py`](../../experiments/01-world-as-text/experiment.py) · [notebook](../../notebooks/01-world-as-text.ipynb) |
| **Durum** | ✅ tam |

## Neden önemli

Boston Dynamics Spot ve Unitree Go2 gibi robot köpeklerin simülasyon modelleri herkese açık birer metin dosyası. Google DeepMind'ın [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) koleksiyonunda `boston_dynamics_spot` ve `unitree_go2` klasörleri var. İçlerinde gövdeyi, bacakları, joint'leri ve motorları tarif eden XML dosyaları duruyor. Örneğin Go2 modeli, Unitree'nin açık URDF'inden türetilmiş ve toplam kütlesi 15,21 kg ([Menagerie · unitree_go2](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_go2)).

Bu dosyalar oyuncak değil. [MuJoCo Playground](https://playground.mujoco.org/) makalesine göre bu tür modellerde eğitilen policy'ler tek GPU'da dakikalar içinde öğreniyor ve bazıları gerçek robota zero-shot, yani ek ayar yapmadan aktarılıyor ([Zakka vd. 2025, arXiv 2502.08844](https://arxiv.org/abs/2502.08844)).

Yani fiziksel AI'nin ilk katmanı bir metin dosyası. Bu derste 15 satırlık kendi dünyanı yazacak, onu bozacak ve physics engine'in bu yazıyı nasıl "oynattığını" ölçeceksin.

## Kavram

### Tarif ve uygulayan

Bir yemek tarifi yemek değildir. Satırları uygulayan biri gerekir. Simülasyonda tarif **MJCF** (MuJoCo'nun XML formatı), uygulayan **physics engine**. Engine tarifi yorumlamaz. Her adımda aynı hareket denklemlerini çözer.

### 15 satırın anatomisi

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

| Satır | Ne tarif ediyor |
|---|---|
| 2 | `option`: yerçekimi −9,81 m/s² (default değerle aynı, açıkça yazıldı) |
| 4 | `plane`: sonsuz görünen zemin |
| 5 | `table` (masa): 70 × 50 cm kutu, üstü 40 cm'de (`size` yarı uzunluk; alt yarısı zeminin altında) |
| 6–9, 10–13 | İki `body` (`light` ve `heavy`), her birinde 20 cm'lik bir küp. `freejoint` cismi serbest bırakır: 3 öteleme + 3 dönme |
| 8, 12 | `mass`: kütle (kg). Deneyde değiştireceğimiz tek şey |

### Yazmadıkların da bir şey söyler

Dosyada olmayan her ayar default değerini alır ([XML referansı · option](https://mujoco.readthedocs.io/en/stable/XMLreference.html#option)):

- `timestep` yok → **0,002 sn**, saniyede 500 adım. Her adımda serbest düşen küpün hızı 9,81 × 0,002 = **1,96 cm/s** artar (semi-implicit Euler, [Computation](https://mujoco.readthedocs.io/en/stable/computation/index.html)).
- `density` ve `viscosity` yok → ikisi de **0**. MuJoCo hiçbir fluid force hesaplamaz: bu dünyada **hava yok** ([Fluid forces](https://mujoco.readthedocs.io/en/stable/computation/fluid.html)).

### Neden ağır olan daha hızlı düşmez?

Yerçekimi ağır küpü 10 kat güçlü çeker (F = m·g), ama onu hızlandırmak da 10 kat zordur (a = F/m = g). Kütle sadeleşir. Galileo *İki Yeni Bilim*'de (*Discorsi*, 1638) dirençsiz bir ortamda farklı ağırlıktaki cisimlerin aynı hızla düşeceğini savundu ([Stanford Encyclopedia of Philosophy · Galileo](https://plato.stanford.edu/entries/galileo/)). Bizim dosyamız da dirençsiz, yani karşılaştırma doğru koşulda.

> Pisa Kulesi'nden top atma hikâyesi ilk kez Galileo'nun öğrencisi Viviani'nin ölümünden sonra basılan biyografisinde geçer. Galileo'nun kendi yazılarında yoktur ve tarihçiler doğrulanmamış sayar. Belgelenmiş erken bir düşürme deneyi, Stevin ile de Groot'un Delft'te 1586'da yaptığı deneydir ([Britannica · Galileo](https://www.britannica.com/biography/Galileo-Galilei), [Wikipedia · Galileo's Leaning Tower of Pisa experiment](https://en.wikipedia.org/wiki/Galileo%27s_Leaning_Tower_of_Pisa_experiment)).

## Kendin kur

### Adım 1 · Kurulum

```bash
pip install -r requirements.txt     # mujoco==3.17.0, numpy
```

### Adım 2 · Dosyayı oku, satırı say

```python
import mujoco
LINES = XML.split("\n")
print(len(LINES), "lines")                    # 15
m = mujoco.MjModel.from_xml_string(XML)
print(m.opt.timestep, m.opt.density, m.opt.viscosity)   # 0.002 0.0 0.0
```

### Adım 3 · Kodu canlı değiştir

Deney, dosyayı satır satır düzenler. 8. ve 12. satırdaki kütleler değişir; havayı eklemek için 2. satıra tek bir attribute gelir:

```python
def make_xml(light_kg=0.1, heavy_kg=1.0, air=False):
    lines = list(LINES)
    lines[7] = lines[7].replace('mass=".1"', f'mass="{light_kg:g}"')
    lines[11] = lines[11].replace('mass=".1"', f'mass="{heavy_kg:g}"')
    if air:
        lines[1] = ' <option gravity="0 0 -9.81" density="1.2"/>'   # air density, kg/m³
    return "\n".join(lines)
```

1,2 kg/m³, 20 °C'de deniz seviyesindeki kuru havanın yoğunluğuna (1,204) yakın ([Engineering ToolBox · air density](https://www.engineeringtoolbox.com/air-density-specific-weight-d_600.html)). Satır sayısı değişmez.

### Adım 4 · Bırak ve ölç

`drop()` iki küpü aynı anda bırakır ve her birinin alt yüzünün masa üstüne (40 cm) indiği **physics step**'i bulur. Tam kod: [`experiment.py`](../../experiments/01-world-as-text/experiment.py).

```bash
python experiments/01-world-as-text/experiment.py
```

## Sayılarla sonuç

MuJoCo 3.17.0, Apple M5, deterministik (rastgelelik yok, iki koşu aynı çıktı).

**Vakum (dosyadaki hâli), hafif 100 g, ağır 1 kg:** ikisi de **143. adımda** (0,286 sn) masaya iniyor. Contact'tan önceki yükseklik dizileri bit düzeyinde aynı. Analitik düşme süresi √(2 · 0,40 / 9,81) = 0,2856 sn.

**Havayı ekleyince (`density="1.2"`):** ağır küp yine 143. adımda, hafif küp **145. adımda** iniyor. Ağır küp masaya değdiğinde hafif küp **1,10 cm** yukarıda. Interpolation ile zaman farkı **4,08 ms**.

**Sweep: hafif küpün kütlesi** (ağır hep 1 kg, 20 cm küpler, 40 cm düşüş):

| hafif küp | vakumda fark | havada fark |
|---|---|---|
| 5 g | 0,00 ms | 93,44 ms |
| 10 g | 0,00 ms | 46,25 ms |
| 20 g | 0,00 ms | 22,59 ms |
| 50 g | 0,00 ms | 8,65 ms |
| 100 g | 0,00 ms | 4,08 ms |
| 200 g | 0,00 ms | 1,81 ms |
| 500 g | 0,00 ms | 0,45 ms |
| 1 kg | 0,00 ms | 0,00 ms |

Vakumda sekiz kütlenin hepsinde fark sıfır. Havada fark kabaca **1/kütle** ile ölçekleniyor: kütle yarıya inince gecikme yaklaşık iki katına çıkıyor. Beklenen de bu, çünkü hava direncinin (drag) yarattığı yavaşlama (k·v²/m) kütleye bölünüyor.

**Elle sağlama.** MuJoCo'nun default fluid modeli kutu için quadratic drag kullanır, ½·ρ·A·v² (etkin katsayı ≈ 1). 100 g'lık 20 cm küp için k/m = 0,5 · 1,2 · 0,04 / 0,1 = 0,24 1/m. Düşüşteki yol kaybı yaklaşık (k/m)·g²·T⁴/12 ≈ 1,28 cm (hafif) − 0,13 cm (ağır) ≈ **1,15 cm**. Simülasyonun 1,10 cm'si ile uyumlu.

## Sınırlar

- **Hava sonucu bir modelin sonucu, gerçek ölçüm değil.** Yüzü öne bakan bir küpün gerçek drag coefficient'ı ≈ 1,05 (Hoerner, *Fluid-Dynamic Drag*, 1965); MuJoCo'nun etkin değeri 1'e yakın. Küp dönmezse gerçek fark da bu mertebede olur.
- Gerçek bir kâğıt küp yalpalar ve döner. Simülasyondaki küp rigid body'dir ve sabit yönelimle düşer.
- 20 cm'lik 100 g küpün yoğunluğu boş bir karton kutununki kadar (0,0125 g/cm³). Hafif küp bilerek hafif seçildi ki hava etkisi görünsün.
- Küp masaya değdikten sonra soft contact onu birkaç mm gömer. Bunu [ders 03](03-soft-contact.md) ölçecek.

## Alıştırmalar

1. **Kolay.** Küpleri 2 m yükseğe koy (`pos="... 2.4"`). Vakumda yine aynı adımda iniyorlar mı? Havada fark kaç cm oluyor?
2. **Kolay.** `gravity="0 0 -1.62"` (Ay) yap. Vakumda iki küp kaçıncı adımda iniyor? Analitik süreyle karşılaştır.
3. **Orta.** `density="1000"` (su). Küpler ne yapıyor? 100 g'lık 20 cm küpün yoğunluğu suyunkinden büyük mü küçük mü?
4. **Orta.** 20 gramlık kâğıt küp kaç ms geç kalır? Önce 1/kütle kuralıyla tablodan tahmin et, sonra koştur. (20 cm kâğıt küp: 6 yüz × 0,04 m² × 80 g/m² ≈ 19 g.)
5. **Zor.** Dosyaya üçüncü bir küp ekle: aynı kütle, ama 10 cm. Havada hangisi önce iner? Cevabı k/m = ½·ρ·A/m ifadesinden çıkar, sonra ölç.

## Terimler

| Terim | Sık duyulan | Aslında |
|---|---|---|
| MJCF | "MuJoCo'nun 3D model dosyası" | Dünyayı (body'ler, joint'ler, motorlar, fizik ayarları) tarif eden XML formatı. Görünüm değil, fizik tarifi |
| URDF | "Robot dosyası" | ROS dünyasının robot tarif formatı. Menagerie modellerinin çoğu üreticinin URDF'inden MJCF'e çevrilmiş |
| `body` / `geom` | "Nesne" | `body` hareket eden frame, `geom` ona bağlı collision/kütle şekli |
| `freejoint` | — | Cismi 6 degree of freedom (DoF) ile (3 öteleme, 3 dönme) serbest bırakan joint |
| Default değer | "Ayar yok" | Yazılmayan her attribute bir değer alır. `density = 0` "hava yok" demektir |

## Kaynaklar

- [MuJoCo XML referansı · `option`](https://mujoco.readthedocs.io/en/stable/XMLreference.html#option): `timestep` 0,002, `density` 0, `viscosity` 0 default'ları.
- [MuJoCo · Computation](https://mujoco.readthedocs.io/en/stable/computation/index.html) ve [Fluid forces](https://mujoco.readthedocs.io/en/stable/computation/fluid.html): integrator ve fluid modeli.
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie): `boston_dynamics_spot`, `unitree_go2` ve diğer modeller.
- [Zakka vd., *MuJoCo Playground*, arXiv 2502.08844](https://arxiv.org/abs/2502.08844).
- [Stanford Encyclopedia of Philosophy · Galileo](https://plato.stanford.edu/entries/galileo/) · [Britannica · Galileo](https://www.britannica.com/biography/Galileo-Galilei).
- Bu sayfadaki her sayı `experiment.py`'nin koşusundan (MuJoCo 3.17.0, 27.10.2026). Koddan Kasa video ekibinin iç doğrulama notuyla birebir aynı.
