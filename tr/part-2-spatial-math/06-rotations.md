# 06 · Rotation: matrix, Euler açıları, axis-angle, quaternion

<!-- meta -->
| | |
|---|---|
| **Bölüm** | [2 · Spatial math](README.md) |
| **Çalışma türü** | CPU · Mac ya da Colab CPU |
| **Önkoşul** | [01](../part-1-simulation/01-world-as-text.md) |
| **Önce öğren** | [Linear Algebra Intuition](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/01-linear-algebra-intuition) · [Matrix Transformations](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/03-matrix-transformations) |
| **Süre** | ~60 dk |
| **Kod** | [`experiments/06-rotations/experiment.py`](../../experiments/06-rotations/experiment.py) |
| **Durum** | ✅ tam |
<!-- /meta -->

> Orientation'ı rotation matrix ya da birim quaternion olarak sakla; Euler açılarını yalnızca göstermek için kullan.

## Problem

Serbest hareket eden bir cisim içeren herhangi bir MuJoCo modelini açıp state vektörüne bakın. Tek bir serbest cismin `nq = 7` konum sayısı (3'ü konum, 4'ü quaternion) ama yalnızca `nv = 6` hız sayısı (3 doğrusal, 3 açısal) vardır; bizim çalıştırmamız tam olarak bunu ve varsayılan quaternion'ı `[1, 0, 0, 0]` yazdırıyor. Bir simülatör üç boyutlu bir büyüklük için neden dört sayı saklar, konum ile hız neden farklı boyuttadır?

Çünkü orientation'ın üç sayılık her tarifi bir yerde bozulur. Euler açılarında 90° pitch'e gelmiş bir gripper bir dönme yönünü kaybeder: hiçbir sonlu Euler açısı hızı o yönü üretemez. Bu ders bunun nerede olduğunu ve neye mal olduğunu ölçüyor, bozulmayan gösterimleri de gösteriyor.

## Ana fikir

**Rotation matrix.** $R \in SO(3)$: $R^\top R = I$, $\det R = +1$. Sütunları, dönmüş frame'in eksenlerinin üst frame'deki ifadesidir. Aynı matrisin iki okunuşu vardır:

| Okunuş | Formül | Anlamı |
|---|---|---|
| Active | $p' = R\,p$ | $p$ vektörünü döndür |
| Passive | $p_B = R_{AB}^\top\, p_A$ | aynı nokta, $R_{AB}$ kadar dönmüş bir $B$ frame'indeki koordinatları |

Bileşke, matris çarpımıdır ve sıra **değiştirilemez**: $R_z(90°)R_x(90°)$ ile $R_x(90°)R_z(90°)$ arasında 120°'lik bir rotation fark vardır (aşağıda ölçüldü). $R_{AC} = R_{AB} R_{BC}$: soldan sağa okununca her yeni rotation, yeni ulaşılan frame'in eksenleri etrafındadır (intrinsic).

**Euler açıları.** Üç eksen etrafında üç açı. Bu ders intrinsic z-y'-x'' kullanıyor (yaw $\psi$, pitch $\theta$, roll $\phi$): $R = R_z(\psi)R_y(\theta)R_x(\phi)$. Gövde açısal hızı, açı hızlarına doğrusal bağlıdır:

$$\omega_b = E(\theta,\phi)\begin{bmatrix}\dot\phi\\ \dot\theta\\ \dot\psi\end{bmatrix},\qquad E=\begin{bmatrix}1&0&-\sin\theta\\0&\cos\phi&\sin\phi\cos\theta\\0&-\sin\phi&\cos\phi\cos\theta\end{bmatrix},\qquad \det E=\cos\theta .$$

$\theta = \pm 90°$ olduğunda birinci ve üçüncü sütun paralel olur: roll ve yaw cismi aynı eksen etrafında döndürür. Bu **gimbal lock**'tur; fiziksel rotation'ın değil, parametrelemenin singularity'sidir.

**Axis-angle ve Rodrigues.** Her rotation, birim bir $\hat k$ ekseni etrafında bir $\theta$ açısıdır. $K = [\hat k]_\times$ (çapraz çarpım matrisi) ile:

$$R = I + \sin\theta\,K + (1-\cos\theta)\,K^2 .$$

**Birim quaternion.** $q = (\cos\tfrac{\theta}{2},\ \sin\tfrac{\theta}{2}\,\hat k)$, skaler önde. MuJoCo bu sırayı kullanır: modelleme rehberi axis-angle rotation'ının quaternion'ını kosinüs terimi önde yazar ve sıfır rotation `1 0 0 0`'dır ([MuJoCo modeling: frame orientations](https://mujoco.readthedocs.io/en/stable/modeling.html)). Kodda iki gerçek önemlidir:

- **Double cover:** $q$ ile $-q$ aynı rotation'dır.
- **Slerp** en kısa büyük çember yayı boyunca interpolasyon yapar: $\mathrm{slerp}(q_0,q_1,t) = \frac{\sin((1-t)\Omega)}{\sin\Omega}q_0 + \frac{\sin(t\Omega)}{\sin\Omega}q_1$, $\cos\Omega = q_0\cdot q_1$; $q_0\cdot q_1 < 0$ ise önce $q_1 \to -q_1$.

MuJoCo bunlar arasında `mju_euler2Quat` (dizi string'i; küçük harf = intrinsic, büyük harf = extrinsic), `mju_axisAngle2Quat`, `mju_quat2Mat`, `mju_mat2Quat`, `mju_mulQuat`, `mju_subQuat` ile dönüşüm yapar ([API reference](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html)). Lineer cebir AIEFS'te: [Matrix Transformations](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/03-matrix-transformations); $SO(3)$ ve exponential coordinate'ların titiz anlatımı Lynch & Park, *Modern Robotics*, Böl. 3'te.

## Deney

Script: [`experiments/06-rotations/experiment.py`](../../experiments/06-rotations/experiment.py). MuJoCo 3.17.0, NumPy 2.5.4, Apple Silicon Mac (Apple M5), sabit seed'ler, 10 s'nin altında.

**Önce tahmin et.** Euler hız haritası hangi pitch'te sorun olur: 45°, 80°, yoksa yalnızca tam 90°'de mi? Saf Euler açısı interpolasyonu gerçek en kısa rotation'dan ne kadar sapabilir?

**Adım 1 · dönüşümler tutarlı.** 1000 rastgele rotation; elle yazılmış matrisler ile MuJoCo fonksiyonları.

```python
q = np.zeros(4)
mujoco.mju_euler2Quat(q, [yaw, pitch, roll], "zyx")   # intrinsic z-y'-x''
R = np.zeros(9); mujoco.mju_quat2Mat(R, q)
```

| Kontrol (1000 rastgele rotation) | en büyük mutlak hata |
|---|---|
| `mju_euler2Quat("zyx")` ile $R_zR_yR_x$ | 5,6e-16 |
| `mju_axisAngle2Quat` ile Rodrigues | 6,7e-16 |
| `mju_mat2Quat` → `mju_quat2Mat` gidiş-dönüş | 4,2e-16 |
| $q$ ile $-q$ matrisi | 0,0e+00 |
| 1000 bileşke sonrası $\lvert R^\top R - I\rvert$ | 1,4e-14 |

**Adım 2 · pitch sweep'i → $E$'nin koşullanması.** Analitik $E$, $R$'nin merkezi farklarıyla kontrol edildi (her satırda uyum ≤ 9,6e-11). Roll 30°'de sabit.

| pitch | det E | σ_min | cond(E) |
|---|---|---|---|
| 0° | 1,00 | 1,00 | 1,000 |
| 60° | 0,500 | 0,366 | 3,732 |
| 80° | 0,174 | 0,123 | 11,43 |
| 89° | 0,0175 | 0,0123 | 114,6 |
| 89,9° | 0,00175 | 0,00123 | 1146 |
| 89,99° | 0,000175 | 0,000123 | 11460 |
| 90° | 6,12e-17 | 3,40e-17 | sonsuz |

Condition number $1/\cos\theta$ gibi büyüyor: 80°'de bile 11,43. Sorun tek bir nokta değil, bir komşuluk.

**Adım 3 · başlangıç pitch'i sweep'i → interpolasyon hatası.** Her satırda 300 çift; hedef, başlangıcın rastgele bir eksen etrafında 60° döndürülmüş hâli. Saf interpolasyon (yaw, pitch, roll) üzerinde doğrusal; yaw ve roll kısa yoldan sarılıyor; referans slerp. "sapma" iki yol arasındaki en büyük açı, "yol/60" Euler yolunun gerçekte ne kadar rotation katettiği.

| başlangıç pitch | medyan sapma | p95 sapma | en büyük sapma | medyan yol/60° | sapması > 90° olan çift |
|---|---|---|---|---|---|
| 0° | 3,71° | 6,30° | 6,95° | 1,010 | 0 |
| 30° | 4,44° | 14,88° | 20,62° | 1,014 | 0 |
| 60° | 7,83° | 23,91° | 179,74° | 1,042 | 7 |
| 85° | 14,96° | 178,59° | 179,83° | 1,149 | 22 |
| 89° | 16,99° | 178,73° | 179,99° | 1,190 | 23 |

Pitch 0'da bile Euler yolu en kısa yol değil (medyan 3,71° sapma). 90° yakınında hedefin pitch'i ±90° sınırını geçebiliyor; çıkarılan yaw ve roll yaklaşık 180° sıçrıyor ve interpolasyonla bulunan orientation uzun yoldan dolaşıyor: katetmesi gereken 60°'nin 5,595 katına kadar.

## Boz

**Euler hızları patlar.** $E$'nin 90°'de kaybettiği yön etrafında (roll 30° için $\omega_b \propto (0, \sin\phi, \cos\phi)$) 1 rad/s'lik gövde rotation'ı iste ve $E\,\dot{\boldsymbol\alpha} = \omega_b$'yi çöz:

| pitch | 0° | 60° | 85° | 89° | 89,9° | 89,99° |
|---|---|---|---|---|---|---|
| gereken ‖Euler hızı‖ (rad/s) | 1,0 | 2,6 | 16,2 | 81,0 | 810,3 | 8102,8 |

Euler açısı hızı komut eden bir controller, tamamen sıradan bir hareketi üretmek için actuator'larını doyuma sokar. Bu, [ders 10](../part-3-kinematics/10-inverse-kinematics.md)'da kinematik singularity olarak tekrar göreceğin mekanizmanın aynısı: hızları hızlara eşleyen bir matris rank kaybediyor.

**İşaret kontrolü olmadan slerp.** Aynı şekilde 60° aralıklı 300 çift, ama $q_1$ rastgele olarak $-q_1$ ile değiştiriliyor (aynı rotation). 300 çiftin 156'sında $q_0\cdot q_1 < 0$ çıktı. Kontrolle her yol 60,0° katediyor; kontrolsüz en kötü yol 300,0° katediyor: uzun yoldan, çünkü quaternion yayı $360° - 60°$'lik rotation'ı kapsıyor.

## Gerçek robotta

- SO-101'in joint'leri hinge'dir; joint açıları hiçbir zaman gimbal lock yaşamaz. Sorun bir üst seviyede çıkar: gripper orientation hedefini roll-pitch-yaw olarak yazdığın her yerde (teleoperation arayüzleri, dataset etiketleri, policy action'ları). Quaternion sakla ve quaternion ile interpolasyon yap; Euler açısına yalnızca bir insana göstermek için çevir.
- Araçlar arasında konvansiyonlar farklıdır. MuJoCo skaleri öne koyar; `eulerseq` compiler varsayılanı `"xyz"`'dir ve dokümantasyon URDF'nin "rpy"sinin MJCF'te `"XYZ"`'ye (extrinsic) karşılık geldiğini söyler ([XML reference: compiler/eulerseq](https://mujoco.readthedocs.io/en/stable/XMLreference.html#compiler-eulerseq)). Birçok başka kütüphane skaleri sona koyar. Kaydedilmiş bir orientation'a güvenmeden önce bilinen bir rotation'ı çevirip kontrol et.
- IMU'lar ve kamera tracker'ları gürültülü orientation tahmini verir. Bu dersteki hiçbir şey donanımda ölçülmedi; yukarıdaki sayılar double precision'da saf matematiktir.

## Kendini dene

1. **Kolay.** $R_x(90°)^\top$, x etrafında −90°'lik active rotation mı, yoksa +90° dönmüş bir frame'e passive koordinat değişimi mi? (İkisini de cevapla.)
2. **Orta.** Adım 2'yi extrinsic `"XYZ"` dizisi için tekrarla. Hız haritası hangi pitch'te singular olur?
3. **Zor.** `interp_stats`'ı, saf interpolasyon hedefin Euler üçlüsünü önce başlangıca en yakın dala açacak şekilde değiştir ($(\psi,\theta,\phi)$ ve $(\psi\pm\pi, \pi-\theta, \phi\pm\pi)$ ikisini de dene). Başlangıç pitch'i 85°'de en büyük sapma 90°'nin altına iner mi? Medyan iner mi?

### Cevaplar

1. İkisi de: active rotation olarak $R_x(90°)^\top = R_x(-90°)$'dir; aynı zamanda üst frame'deki koordinatları x etrafında +90° dönmüş bir frame'e taşır.
2. Sabit eksenler etrafında önce X, sonra Y, sonra Z, $R = R_z(c)R_y(b)R_x(a)$ verir: intrinsic z-y'-x'' ile aynı matris. Bu yüzden aynı yerde, ortadaki açı ±90° iken singular olur. Extrinsic diziyi `mju_euler2Quat`'a verip matrisleri karşılaştırarak kontrol et.
3. Bu dersin script'inde hesaplanmadı. Çalıştır ve Adım 3 tablosuyla karşılaştır; ne bulursan bul, pitch 0'daki medyan sapma sıfıra inemez, çünkü Euler açılarında düz bir çizgi $SO(3)$'te geodesic değildir.

## Akılda kalanlar ve kaynaklar

- Rotation matrix ve birim quaternion singularity içermez; Euler açılarının hız haritasında $\det E = \cos\theta$, 80° pitch'te cond zaten 11,43.
- Euler açıları arasında asla interpolasyon yapma; $q_0\cdot q_1<0$ çevirmesiyle slerp gerçek 60,0°'yi katediyor, saf Euler yolları bunun 5,595 katına kadar katetti.
- Konvansiyonunu bil: skaler önde ya da sonda, intrinsic (küçük harf) ya da extrinsic (büyük harf), active ya da passive.

Sonraki: [ders 07 · Rigid transform'lar ve frame tree'leri](07-rigid-transforms.md).

**Kaynaklar**

- K. M. Lynch, F. C. Park, *Modern Robotics: Mechanics, Planning, and Control*, Cambridge Univ. Press, 2017, Böl. 3 "Rigid-Body Motions". [Kitap sayfası ve ücretsiz preprint](https://hades.mech.northwestern.edu/index.php/Modern_Robotics).
- MuJoCo dokümantasyonu: [Modeling · frame orientations](https://mujoco.readthedocs.io/en/stable/modeling.html), [XML reference · compiler/eulerseq](https://mujoco.readthedocs.io/en/stable/XMLreference.html#compiler-eulerseq), [API functions (`mju_*` matematik)](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html).
