# 22 · Configuration space ve sampling-based planning

<!-- meta -->
| | |
|---|---|
| **Bölüm** | [6 · Planning ve optimal control](README.md) |
| **Çalışma türü** | CPU · Mac ya da Colab CPU |
| **Önkoşul** | [10](../part-3-kinematics/10-inverse-kinematics.md) |
| **Önce öğren** | [Probability and Distributions](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/06-probability-and-distributions) |
| **Süre** | ~60 dk |
| **Kod** | [`experiments/22-motion-planning/experiment.py`](../../experiments/22-motion-planning/experiment.py) |
| **Durum** | ✅ tam |
<!-- /meta -->

> Sampling-based bir planner engeli hiç görmez. Yalnızca sorduğu configuration'larda collision checker'ın verdiği cevapları görür.

## Problem

Inverse kinematics ([ders 10](../part-3-kinematics/10-inverse-kinematics.md)) sana hedef configuration'ı verir, oraya nasıl gideceğini vermez. Bir Franka Panda'nın önüne 20 × 20 × 50 cm'lik bir kutu koy; el kutunun solunda başlasın, sağında bitsin. İki configuration da çarpışmasız. Joint space'te aralarındaki düz çizgi ise değil: her 0,005 rad'da yeniden kontrol edildiğinde kol kutuya 139,0 mm giriyor (aşağıdaki koşumumuz). 7 boyutlu joint space'in boş kısmından geçen bir path'e ve "bulunan path"in gerçekten boş olup olmadığını bilmeye ihtiyacımız var.

## Ana fikir

**Configuration space.** Bir configuration $q \in \mathcal{C} \subset \mathbb{R}^7$ tek bir joint vektörüdür; joint limitleri $\mathcal{C}$'yi bir kutu yapar. Engel bölgesi ve boş uzay:

$$\mathcal{C}_{obs} = \{\, q : \text{robot}(q) \cap \text{engeller} \neq \emptyset \,\}, \qquad \mathcal{C}_{free} = \mathcal{C} \setminus \mathcal{C}_{obs}.$$

Workspace'teki bir kutu, $\mathcal{C}$'de kimsenin açıkça hesaplamadığı eğri bir 7 boyutlu kümeye dönüşür ([Lynch & Park böl. 10](https://hades.mech.northwestern.edu/index.php/Modern_Robotics); [LaValle böl. 4](http://lavalle.pl/planning/)). Sampling-based planner'lar yalnızca bir üyelik sorusu sorar: $q$ boş mu?

**MuJoCo'da collision checking.** `mj_kinematics` verilen `qpos` için her link'i yerine koyar, `mj_collision` broad ve narrow phase'i çalıştırıp `d.contact`'ı doldurur. Varsayılan sıfır contact margin'inde MuJoCo yalnızca mesafesi $\le 0$ olan contact'ları raporlar; yani `d.ncon > 0` ise $q \in \mathcal{C}_{obs}$'tir ve `-contact.dist` penetrasyon derinliğidir. Bu, dinamiği hesaplamadan `mj_forward`'ın üreteceği contact'ların aynısını üretir.

**Edge kontrolü.** $q_a$'dan $q_b$'ye bir tree edge'i $q(s) = q_a + s\,(q_b - q_a)$ doğru parçasıdır. $n = \lceil \lVert q_b - q_a \rVert / \text{res} \rceil$ noktada kontrol edilir. Kontrol edilen iki nokta arasında olan her şeyin boş olduğu varsayılır. Bir joint axis'ten $r$ uzaklıktaki nokta, o joint $\Delta q$ döndüğünde $r\,\Delta q$ kadar yer değiştirir; yani iki kontrol arasındaki en kötü hareket $\sum_i r_i\,\lvert \Delta q_i \rvert$ ile sınırlıdır.

**Planner'lar.**

| Yöntem | Bir iterasyon |
|---|---|
| RRT (goal bias'lı) | $q_{rand}$ örnekle (%5 olasılıkla hedef), en yakın tree node'unu bul, örneğe doğru en fazla $\varepsilon$ adım at, edge boşsa yeni node'u tut; hedef $\varepsilon$ içinde ve ulaşılabilir olunca dur |
| RRT-Connect | A tree'sini $q_{rand}$'a doğru bir adım büyüt, sonra B tree'si yeni node'a ulaşana (çözüldü) ya da engellenene kadar ona doğru tekrar tekrar adım atsın; tree'leri değiştir ([Kuffner & LaValle 2000](http://lavalle.pl/papers/KufLav00.pdf)) |
| Shortcutting | iki waypoint seç; aradaki alt-path'i, düz doğru parçası edge kontrolünden geçiyorsa onunla değiştir; tekrarla |

İki RRT türü de olasılıksal olarak tamdır, optimal değildir ([LaValle böl. 5](http://lavalle.pl/planning/)): ham path zikzak çizer, shortcutting bu yüzden gelir.

## Deney

**Tahmin.** Daha kaba bir edge-check çözünürlüğü daha az collision check ister ve daha hızlı plan yapar. Ama bazı edge'lerin kutunun köşesinin üstünden atlamasına da izin vermeli.

Tam script: [`experiments/22-motion-planning/experiment.py`](../../experiments/22-motion-planning/experiment.py). Özü:

```python
def depth(self, q):                       # penetration (m) at q, 0 if free
    d.qpos[:7] = q; d.qpos[7:] = 0.04     # fingers fixed open
    mujoco.mj_kinematics(m, d)
    mujoco.mj_collision(m, d)
    return 0.0 if d.ncon == 0 else -min(d.contact[i].dist for i in range(d.ncon))

def edge_free(self, a, b, res):
    n = max(1, int(np.ceil(np.linalg.norm(b - a) / res)))
    return all(self.free(a + (b - a) * k / n) for k in range(1, n + 1))
```

RRT-Connect, $\varepsilon = 0{,}5$ rad'lık extend adımı (7 joint üzerinde L2 norm), 2000 iterasyon bütçesi ve 200 shortcut denemesi kullanır. Dönen her path sonra 0,005 rad'da yeniden kontrol edilir: "çarpışan path" için referans budur. MuJoCo 3.17.0, NumPy 2.5.4, Apple Silicon Mac (Apple M5), sabit seed'ler, satır başına 100 seed.

| res (rad) | başarı | medyan iterasyon | medyan collision check | medyan uzunluk ham → shortcut (rad) | çarpışan path | en kötü penetrasyon (mm) |
|---|---|---|---|---|---|---|
| 0,01 | 100/100 | 33 | 2070 | 5,62 → 3,61 | 0/100 | 0,0 |
| 0,05 | 100/100 | 33 | 431 | 5,57 → 3,59 | 7/100 | 4,5 |
| 0,10 | 100/100 | 33 | 225 | 5,55 → 3,58 | 17/100 | 8,0 |
| 0,25 | 100/100 | 29 | 96 | 5,37 → 3,44 | 42/100 | 26,5 |
| 0,50 | 100/100 | 26 | 52 | 5,01 → 3,12 | 66/100 | 72,0 |
| 1,00 | 100/100 | 25 | 48 | 4,98 → 3,00 | 80/100 | 156,4 |

Verinin söyledikleri:

1. **Başarı oranı burada kalite ölçüsü olarak işe yaramaz.** Her satır 100/100 diyor. Sadece yoğun yeniden kontrol, 1,0 rad'da 100 "çözüm"ün 80'inin kutunun içinden geçtiğini gösteriyor; en kötüsü 156,4 mm derinde, saf düz çizgiden (139,0 mm) bile derin.
2. **Kaba kontrol, varsayılan olarak log'layacağın her metrikte daha iyi görünür.** Collision check sayısı 2070'ten 48'e, iterasyon 33'ten 25'e düşer, shortcut'lı path kısalır (3,61 → 3,00 rad). Kısalır, çünkü shortcut'ların engelin içinden kesmesine izin verilmiştir.
3. **0,05 rad bile güvenli değil.** 100 path'in 7'si kutuyu 4,5 mm'ye kadar sıyırıyor. $r = 0{,}6$ m ile tek bir joint'in 0,05 rad'lık adımı bir noktayı zaten 3 cm oynatır; bundan sığ bir iç içe geçme iki kontrolün arasında başlayıp bitebilir.

**RRT ve RRT-Connect** (res 0,05 rad, 100 seed, aynı bütçe):

| planner | başarı | medyan iterasyon | medyan check | medyan uzunluk ham → shortcut (rad) |
|---|---|---|---|---|
| RRT | 42/100 | 151 | 1438 | 5,47 → 3,33 |
| RRT-Connect | 100/100 | 33 | 431 | 5,57 → 3,59 |

İki uçtan büyüyüp açgözlü bağlanmak, RRT-Connect'in bu sorguyu birkaç düzine iterasyonda çözmesini sağlıyor; tek tree'li RRT, 2000 iterasyon içinde 100 seed'in yalnızca 42'sinde başardı.

## Boz

Kutuyu, $x = 0{,}40$ m'de 2 m genişliğinde, ortası $z = 0{,}45$ m'de yatay bir yarığı olan bir duvarla değiştir. Hedefte el duvarın arkasında, önkol yarığın içinde; yani her çözüm yarıktan geçmek zorunda. Yarığı daralt (RRT-Connect, res 0,05 rad, 30 seed, 3000 iterasyon bütçesi):

| yarık yüksekliği (m) | başarı | medyan iterasyon | medyan check | çarpışan path |
|---|---|---|---|---|
| 0,40 | 30/30 | 9 | 128 | 2/30 |
| 0,30 | 30/30 | 102 | 896 | 4/30 |
| 0,25 | 30/30 | 360 | 2747 | 5/30 |
| 0,22 | 30/30 | 856 | 6372 | 5/30 |
| 0,20 | 25/30 | 1487 | 10520 | 0/25 |

0,40 m'den 0,20 m'ye medyan iterasyon 9'dan 1487'ye çıkıyor ve 30 seed'in 5'i bütçeyi bitiriyor. Başlangıç ve hedef her satırda boş (script kontrol ediyor) ve 0,20 m'de başarılı 25 path yoğun yeniden kontrolden geçiyor; yani boş bir yol var: 5 başarısızlık problemin çözümsüz olmasından değil, planner'ın bütçeyi bitirmesinden. Mekanizma: uniform sampling bir bölgeye hacmiyle orantılı olasılıkla isabet eder. Yarık yüksekliğini yarıya indirmek, içinden geçen boş tüpü aynı anda birkaç joint yönünde daraltır; 7 boyuttaki hacmi workspace'teki boşluktan çok daha hızlı küçülür. Sampling-based planning'in narrow passage problemi budur. Ayrıca daha geniş dört satırda res 0,05 rad'da 30 path'in 2 ila 5'i hâlâ duvarı sıyırıyor: kaba edge kontrolünün yanıldığı yer tam da açıklığın az olduğu yer.

## Gerçek robotta

- **Edge çözünürlüğü gerçek bir parametredir.** MoveIt'in OMPL arayüzü bunu `longest_valid_segment_fraction` / `maximum_waypoint_distance` olarak sunar ve dokümantasyonu uyarır: "Set too high and collisions will be missed around small or narrow objects" ([MoveIt OMPL tutorial](https://moveit.picknik.ai/main/doc/examples/ompl_interface/ompl_interface_tutorial.html)). Tablomuz bu cümlenin sayılarla hâli.
- **Panda.** Planning yukarıdaki 7 boyutlu uzayda; joint limitleri Menagerie [`panda.xml`](https://github.com/google-deepmind/mujoco_menagerie/blob/f3b6013/franka_emika_panda/panda.xml)'den gelir (ör. joint 4: [−3,0718, −0,0698] rad). MuJoCo gerçek kol yüzeyini değil, link mesh'lerinin convex hull'larını çarpıştırır.
- **SO-101.** Menagerie [`so101.xml`](https://github.com/google-deepmind/mujoco_menagerie/blob/f3b6013/robotstudio_so101/so101.xml)'de 5 kol joint'i ve bir gripper var; gripper sabitken planning uzayı 5 boyutlu. Prosedür: masayı ve engelleri ölç, her kutuyu ölçüm hatan kadar şişir, MuJoCo'da plan yap, son path'i yoğun yeniden kontrol et, sonra position servo'lar üzerinden yavaşça uygula. Gerçek kolda henüz ölçülmedi.
- **Model hatası collision hatasıdır.** Gerçekte 1 cm kaymış bir kutu, modellenmemiş 1 cm'lik bir penetrasyondur. Engelleri (ya da contact `margin`'ini) en az kalibrasyon hatan kadar şişir (bkz. [ders 20](../part-5-perception-and-estimation/20-pixels-to-table.md)).

## Kendini dene

1. **Kolay.** Başlangıç da hedef de boş, ama aradaki düz doğru parçası değil. Bu, $\mathcal{C}_{free}$'nin hangi özelliğini gösterir?
2. **Orta.** Panda'nın hiçbir noktasının herhangi bir joint axis'ten $r_{max} = 1$ m'den uzak olmadığını varsay. Hangi L2 edge çözünürlüğü, iki kontrol arasında hiçbir noktanın 1 cm'den fazla oynamayacağını garanti eder? ($\sum_i r_i \lvert\Delta q_i\rvert \le r_{max}\sqrt{7}\,\lVert \Delta q \rVert_2$ kullan.)
3. **Zor.** 2 cm şişirilmiş kutuya karşı res 0,25 rad ile plan yap, sonra path'leri orijinal kutuya karşı 0,005 rad'da yeniden kontrol et. Şişirme, res 0,01 rad'ın maliyeti olmadan güvenliği geri getiriyor mu?

### Cevaplar

1. $\mathcal{C}_{free}$ convex değildir: iki boş nokta, boş bir doğru parçası anlamına gelmez. Ölçülen düz çizgi penetrasyonu 139,0 mm.
2. $\text{res} \le 0{,}01 / \sqrt{7} \approx 0{,}0038$ rad. Sınır muhafazakârdır; gerçek yer değiştirmeler daha küçüktür.
3. Burada hesaplanmadı; çalıştır. 0,25 rad'daki 42/100'den daha az çarpışan path bekle, ama kontroller arasındaki en büyük boşluktan küçük bir şişirme yine aşılabilir.

## Akılda kalanlar ve kaynaklar

- Bir planner başarıyı kendi collision checker'ına göre raporlar. Son path'i her zaman ince çözünürlükte yeniden kontrol et; 1,0 rad'da "başarılı" 100 path'in 80'i kutudan geçti.
- RRT-Connect kutu sorgusunu medyan 33 iterasyonda çözdü; tek tree'li RRT 100 seed'in yalnızca 42'sinde başardı. Shortcutting medyan path'i 5,62'den 3,61 rad'a indirdi.
- Narrow passage'lar uniform sampling'i bozar: 0,20 m'lik yarık medyan 1487 iterasyon istedi (0,40 m'de 9) ve 30 seed'in 5'i başarısız oldu.

Sonraki: [ders 23](23-trajectory-optimization.md), geometrik bir path'i dinamik olarak uygulanabilir, torque sınırlı bir trajectory'ye çevirmek.

**Kaynaklar**

- K. M. Lynch, F. C. Park, *Modern Robotics*, böl. 10 "Motion Planning". [Kitap sayfası ve PDF](https://hades.mech.northwestern.edu/index.php/Modern_Robotics).
- S. M. LaValle, *Planning Algorithms*, Cambridge University Press 2006, böl. 4 "The Configuration Space", böl. 5 "Sampling-Based Motion Planning". [lavalle.pl/planning](http://lavalle.pl/planning/).
- S. M. LaValle, "Rapidly-exploring random trees: A new tool for path planning", TR 98-11, Iowa State University, 1998. [PDF](http://lavalle.pl/papers/Lav98c.pdf).
- J. J. Kuffner, S. M. LaValle, "RRT-connect: An efficient approach to single-query path planning", ICRA 2000, s. 995–1001. [PDF](http://lavalle.pl/papers/KufLav00.pdf).
- MuJoCo API referansı: [`mj_kinematics`, `mj_collision`](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html).
- MoveIt, [OMPL interface tutorial](https://moveit.picknik.ai/main/doc/examples/ompl_interface/ompl_interface_tutorial.html).
- Tüm tablolar: [`experiments/22-motion-planning/output.txt`](../../experiments/22-motion-planning/output.txt) (MuJoCo 3.17.0, 2026-10-27).
