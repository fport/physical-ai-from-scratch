# 20 · Pixel'den masaya: homography ve depth back-projection

<!-- meta -->
| | |
|---|---|
| **Bölüm** | [5 · Perception ve estimation](README.md) |
| **Çalışma türü** | CPU · Mac ya da Colab CPU |
| **Önkoşul** | [19](19-camera-calibration.md) |
| **Önce öğren** | [Linear Systems](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/17-linear-systems) · [Monocular Depth & Geometry Estimation](https://aiengineeringfromscratch.com/lesson?path=phases/04-computer-vision/26-monocular-depth) |
| **Süre** | ~45 dk |
| **Kod** | [`experiments/20-pixels-to-table/experiment.py`](../../experiments/20-pixels-to-table/experiment.py) |
| **Durum** | ✅ tam |
<!-- /meta -->

> Bir pixel bir ışındır; onu noktaya çevirmek için bir bilgi daha gerekir: bilinen bir düzlem (homography) ya da ölçülmüş bir depth.

## Problem

Ders 21'deki detector "küp (u, v) pixel'inde" diyecek. Kolun ise base frame'inde (x, y, z) bilgisine ihtiyacı var. Bir pixel yalnızca kamera merkezinden çıkan bir ışını belirler; o ışın üzerindeki her nokta aynı pixel'e düşer. Noktayı seçmenin iki standart yolu var: nesnenin masa düzleminde olduğunu varsayıp homography kullanmak ya da depth görüntüsünü okuyup back-projection yapmak. İkisinde de ince hata yapmak kolaydır: homography düzlemin üstündeki her şey için sessizce yanlıştır, depth görüntüsü de değerlerinin ne anlama geldiğini yanlış okursan sessizce yanlıştır.

## Ana fikir

**Back-projection** ($K$, [ders 19](19-camera-calibration.md)'dan). **Optik eksen boyunca** ölçülen $Z$ depth'li $(u,v)$ pixel'i için:

$$X_c = Z\,K^{-1}\begin{bmatrix}u\\v\\1\end{bmatrix},\qquad X_w = R_{wc}X_c + c$$

Burada $R_{wc}$'nin sütunları OpenCV kamera eksenleri, $c$ kamera merkezidir. $Z$ bunun yerine ışın boyunca Öklid mesafesi olsaydı, önce $\|K^{-1}[u,v,1]^\top\|$'ye bölmen gerekirdi.

**MuJoCo'dan depth.** GPU metre değil, doğrusal olmayan bir z-buffer değeri $d\in[0;\,1]$ saklar. Near/far kırpma mesafeleri $n = \texttt{znear}\cdot\text{extent}$ ve $f = \texttt{zfar}\cdot\text{extent}$ ile (varsayılanlar 0,01 ve 50; [MuJoCo XML · visual/map](https://mujoco.readthedocs.io/en/stable/XMLreference.html#visual-map-znear)) klasik bir OpenGL buffer'ı $Z = \dfrac{nf}{f - d\,(f-n)}$ verir. MuJoCo 3.17'nin Python `Renderer`'ı ters çevrilmiş bir buffer okur (near'da 1, far'da 0), eşdeğer $Z = \dfrac{nf}{n + d\,(f-n)}$ dönüşümünü float64'te uygular, sonra metreyi float32 olarak döndürür ([`rendering/classic/renderer.py`, tag 3.17.0](https://github.com/google-deepmind/mujoco/blob/3.17.0/python/mujoco/rendering/classic/renderer.py)). Yani `Renderer.enable_depth_rendering()` zaten optik eksen boyunca metrik depth döndürür; ham buffer'ı kendin okursan (C API `mjr_readPixels`) doğrusallaştırman gerekir. Dokümantasyon, near düzlemini fazla yakın koymanın depth çözünürlüğüne mal olduğu konusunda uyarır.

**Homography.** Bir düzlemdeki noktalar görüntüye 3×3'lük bir projektif dönüşümle gider. Masa $z=0$ için: $s\,[u,v,1]^\top = K[r_1\ r_2\ t]\,[x,y,1]^\top$, yani $[x,y,1]^\top \sim H\,[u,v,1]^\top$. $H$'nin 8 serbestlik derecesi var; 4 nokta çifti (hiçbir üçü aynı doğru üzerinde olmayan) onu belirler, daha fazlası least-squares uyumu verir (`cv2.findHomography`, method 0; outlier'lar için RANSAC/LMEDS/RHO, [OpenCV calib3d](https://github.com/opencv/opencv/blob/4.x/modules/calib3d/include/opencv2/calib3d.hpp)). $K$ ya da depth gerekmez, yalnızca bilinen masa noktaları. Bedeli: formül yalnızca $z = 0$ için doğrudur.

| Yöntem | Gerekenler | Geçerli olduğu yer | Bozulduğu durum |
|---|---|---|---|
| Homography | 4+ bilinen masa noktası | düzlemdeki noktalar | nesnenin yüksekliği var, masa düz değil |
| Depth back-projection | $K$, kamera pozu, depth görüntüsü | görünen her yüzey | depth yanlış okunmuş, depth noise'u, eksik depth |

## Deney

**Kurulum.** MuJoCo sahnesi: gri masa düzlemi, 10/20/50/100 mm yüksekliğinde dört kutu, dikeyden 30° eğik, (0,25, 0, 0)'a bakan 640×480'lik bir kamera (`fovy = 50`). Bir pixel'in ground truth'u, ışınının gördüğü yüzeyle tam kesişimidir. Homography marker'ları: 20 × 20 cm'lik bir bölgedeki 4 ya da 12 bilinen masa noktası, pixel'leri $\sigma$ Gauss noise'uyla bozulmuş; test pixel'leri tüm görüş boyunca rastgele masa pixel'leri, onlar da bozulmuş. Depth back-projection, render edilen depth'i bilinear okumayla kullanır.

```python
r.enable_depth_rendering(); depth = r.render()            # metres along the optical axis
P = c + depth_at(uv)[:, None] * (np.c_[uv, ones] @ np.linalg.inv(K).T @ R_wc.T)
H, _ = cv2.findHomography(marker_px, marker_xy, 0)       # pixel -> table (x, y)
```

**Tahmin.** Simülasyonda depth tamdır, dolayısıyla hatası pixel noise'u çarpı pixel izdüşümü olmalı. Homography kendi uydurma hatasını eklemeli ve düzlemin üstünde bozulmalı.

**Çalıştır:** `python experiments/20-pixels-to-table/experiment.py` (MuJoCo 3.17.0, OpenCV 5.0.0, NumPy 2.5.4, Apple M5, deterministic). Bu Mac'te MuJoCo, stderr'e `ARB_clip_control`'ün olmadığını ve depth doğruluğunun sınırlı olacağını yazan bir uyarı basar; aşağıdaki ölçümler buna rağmen elde edilenlerdir.

**Depth görüntüsünde ne var** (kamera 0,6 m'de; extent 0,497 m, yani near 5,0 mm, far 24,9 m; 300 masa pixel'i):

| Karşılaştırma | Medyan | En büyük |
|---|---|---|
| render edilen depth - optik eksen boyunca z-depth | 0,001 mm | 0,008 mm |
| render edilen depth - Öklid ışın uzunluğu | 45,8 mm | 138,0 mm |
| back-projection, depth $Z$ olarak | 0,001 mm | 0,009 mm |
| back-projection, depth yanlışlıkla ışın uzunluğu olarak | 45,8 mm | 138,0 mm |

Dönen değerler z-depth'tir. Onları ışın uzunluğu saymak görüntü köşelerinde 138,0 mm'ye kadar hataya mal olur.

**Pixel noise'u** (kamera 0,6 m; 200 test pixel'i × 50 deneme, mm cinsinden medyan hata):

| Noise (px) | 4 noktadan H | 12 noktadan H | Depth back-projection |
|---|---|---|---|
| 0 | 0,00 | 0,00 | 0,00 |
| 0,5 | 2,15 | 1,81 | 0,81 |
| 1 | 4,33 | 3,27 | 1,60 |
| 2 | 8,90 | 7,67 | 3,21 |
| 4 | 17,08 | 14,17 | 6,37 |

**Kamera yüksekliği** (1 px noise, 12 noktalı H):

| Yükseklik (m) | Hedefte pixel başına mm | Homography (mm) | Depth back-projection (mm) |
|---|---|---|---|
| 0,3 | 0,67 | 1,13 | 0,88 |
| 0,5 | 1,12 | 2,76 | 1,43 |
| 0,8 | 1,79 | 5,13 | 2,18 |
| 1,2 | 2,69 | 9,56 | 3,28 |
| 2,0 | 4,49 | 13,92 | 5,72 |

**Gözlem.**

1. Tüm hatalar pixel noise'unda doğrusal. Yükseklik sweep'inde depth back-projection hatası, hedefteki pixel izdüşümünün 1,2–1,3 katı: depth kusursuzken hata bütçesinin tamamı pixel noise'udur.
2. Homography eşit noise'ta yaklaşık 3 kata kadar daha kötü: 8 parametresi yalnızca 20 × 20 cm'lik bir bölgeyi kaplayan noisy marker'lardan uyduruluyor ve test pixel'leri bunun dışına extrapolate ediliyor. 12 marker 4'ü yener (1 px'te 3,27'ye karşı 4,33 mm).
3. Yükseklikle depth hatası izdüşüm gibi büyür (0,88 → 5,72 mm, izdüşüm 0,67 → 4,49 mm), homography hatası daha hızlı (1,13 → 13,92 mm): kamera yükseldikçe sabit marker bölgesi daha az pixel kaplar.
4. Simüle edilen depth noise'suzdur; gerçek depth sensor'leri kendi noise'unu ekler, dolayısıyla depth sütunu bir alt sınırdır.

## Boz

Her kutunun üst yüzünün merkezini gören pixel'i sorgula (pixel noise'u yok):

| Kutu yüksekliği (mm) | Homography xy hatası (mm) | Depth back-projection hatası (mm) |
|---|---|---|
| 10 | 4,7 | 0,00 |
| 20 | 10,8 | 0,00 |
| 50 | 36,7 | 0,00 |
| 100 | 88,5 | 0,01 |

Homography her pixel'i, ışınının $z = 0$'a çarptığı yere götürür; $h$ yüksekliğindeki bir nokta ışın boyunca kameradan uzağa kayar: hata $h$ ile ve ışının ne kadar eğik olduğuyla büyür. 10 mm'lik bir nesne bile 4,7 mm'ye mal olur. Çözüm: nesne yüksekliği $h$ biliniyorsa ışını bunun yerine $z = h$ ile kesiştir (bu yalnızca $H$'yi değil, $K$'yı ve pozu gerektirir) ya da depth kullan.

## Gerçek robotta

- Masa üstü bir SO-101 kurulumunda, ölçülmüş konumlara yapıştırılan 4–12 marker'dan bir homography hızlı bir calibration'dır ve düz, ince nesneler için yeterlidir. Marker konumlarını göz kararı değil, robot base frame'inde ölç. Gerçek kolda henüz ölçülmedi.
- Gerçek depth kameraları geçerli depth'i olmayan pixel'ler döndürür; driver'ının bunları nasıl işaretlediğine bak ve ortalamadan önce maskele. Sensor'ünün z-depth mi range mi raporladığını dokümanından kontrol et; hiçbir gerçek depth kamerasını test etmedik.
- Kamera pozu $(R_{wc}, c)$, extrinsic calibration'dan (base'te bilinen bir konumdaki board) ya da bilek kamerası için hand-eye calibration'dan gelir ([ders 19](19-camera-calibration.md)); hatası back-project edilen her noktaya doğrudan eklenir.

## Kendini dene

1. **Kolay.** $H$ neden en az 4 nokta çifti ister?
2. **Orta.** 0,6 m'de ve 1 px noise'ta depth back-projection 1,60 mm veriyor. Gerçek robotta bu simülasyonun dışarıda bıraktığı iki hata kaynağını say.
3. **Zor.** "Bilinen yükseklik" back-projection'ını uygula: pixel ışınını $z = h$ ile kesiştir ve Boz tablosunu tekrar çalıştır. 50 mm'lik kutu için $h = 30$ mm varsayarsan ne kadar hata alırsın?

### Cevaplar

1. $H$ ölçek dışında tanımlıdır: 9 eleman eksi 1 = 8 bilinmeyen; her nokta çifti 2 denklem verir.
2. Depth sensor noise'u ve $K$ ile kamera pozundaki calibration hatası (ayrıca hareket eden robotta rolling shutter ve zamanlama kaymaları).
3. Doğru $h$ ile ışın gerçek noktaya çarpar. Sabit bir ışın boyunca yatay kayma yükseklikte doğrusaldır ve homography satırı $h = 0$ özel durumudur; dolayısıyla 50 mm'lik kutu için 30 mm varsaymak $36{,}7 \times 20/50 \approx 14{,}7$ mm verir (tablodan türetildi, çalıştırılmadı).

## Akılda kalanlar ve kaynaklar

- `mujoco.Renderer` depth'i optik eksen boyunca metrik z-depth'tir (burada en fazla 0,008 mm sapma); onu ışın uzunluğu saymak 138,0 mm'ye kadar hataya mal olur.
- Homography: intrinsics gerekmez ama yalnızca düzlemde geçerlidir; 10 mm nesne yüksekliği bile 4,7 mm'ye mal olur.
- İki yöntem de pixel noise'u ve pixel izdüşümüyle doğrusal ölçeklenir; kameraları yakına yerleştir ve 4'ten fazla marker kullan.

Sıradaki: [ders 21](21-object-perception.md), manipulation için object perception.

Kaynaklar:
- [MuJoCo XML reference · visual/map znear, zfar](https://mujoco.readthedocs.io/en/stable/XMLreference.html#visual-map-znear): model extent'inin kesri olarak kırpma düzlemleri; yakın near düzlemi için çözünürlük uyarısı.
- [MuJoCo `rendering/classic/renderer.py`, 3.17.0](https://github.com/google-deepmind/mujoco/blob/3.17.0/python/mujoco/rendering/classic/renderer.py): ters Z okuma ve metreye dönüşüm.
- [OpenCV calib3d](https://github.com/opencv/opencv/blob/4.x/modules/calib3d/include/opencv2/calib3d.hpp): `findHomography` yöntemleri.
- R. Hartley, A. Zisserman, *Multiple View Geometry in Computer Vision*, 2. baskı, Cambridge University Press, 2004 ([kitap sayfası](https://www.robots.ox.ac.uk/~vgg/hzbook/)): düzlemin indüklediği homography'ler ve back-projection.
- Yukarıdaki tüm sayılar: [`experiment.py`](../../experiments/20-pixels-to-table/experiment.py), çıktı [`output.txt`](../../experiments/20-pixels-to-table/output.txt).
