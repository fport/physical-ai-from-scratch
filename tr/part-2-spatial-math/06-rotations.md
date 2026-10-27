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

### Sade anlatımla

Orientation "bu şey hangi yöne bakıyor?" sorusunun cevabıdır. Üç düğmeli bir tripod kafasına takılı bir kamera düşün: pan (sağa/sola çevir), tilt (yukarı/aşağı bak) ve roll (görüntüyü döndür). Üç düğmenin okuması **Euler açıları**dır; insanlar kolay okur, her arayüz bu yüzden bunları gösterir.

Şimdi kamerayı tam yukarı çevir. Pan ve roll artık kamerayı aynı dikey çizgi etrafında döndürür ve bu pozdan onu yana çeviren bir düğme kalmaz: bir yön kayboldu, **gimbal lock**. Kamera hâlâ her yöne bakabilir; bozulan üç düğmeli *tarif*tir ve o pozun bütün bir komşuluğunda sorun çıkarır.

Çözüm, orientation'ı bir dönüş dizisi olmadan tarif etmektir. **Rotation matrix**, cismin üç ekseninin nereyi gösterdiğini üst frame'in koordinatlarında yazar. **Birim quaternion** "bu eksen etrafında bu açı kadar dön" bilgisini dört sayıya sığdırır. İkisinin de kötü bir pozu yoktur; quaternion'ın tek tuhaflığı $q$ ile $-q$'nun aynı orientation olmasıdır, quaternion karşılaştıran ya da karıştıran kod işareti kontrol etmelidir.

```mermaid
flowchart TD
    E["Euler açıları<br/>yaw, pitch, roll<br/>(insana göstermek için)"] -- "mju_euler2Quat" --> Q["birim quaternion q<br/>w, x, y, z<br/>(sakla, interpolasyon)"]
    A["axis-angle<br/>eksen k, açı θ"] -- "mju_axisAngle2Quat" --> Q
    Q -- "mju_quat2Mat" --> R["rotation matrix R, 3×3<br/>(bileşke, vektör döndürme)"]
    R -- "mju_mat2Quat" --> Q
    R --> P["döndürülmüş nokta<br/>p' = R p"]
```

### Rotation'lar kod olarak

```python
import numpy as np

def Rz(a):                         # z ekseni etrafında a açısı (rad) kadar rotation
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
# Ry(a) ve Rx(a), y ve x etrafında aynı şekilde kurulur

p = np.array([1.0, 0, 0])
Rz(np.pi / 2) @ p                  # active: vektörü döndür               -> [0, 1, 0]
Rz(np.pi / 2).T @ p                # passive: aynı vektör, +90° dönmüş frame'deki koordinatları -> [0, -1, 0]

R = Rz(yaw) @ Ry(pitch) @ Rx(roll) # Euler z-y'-x'': bileşke = çarpım, sıra önemli
q = np.r_[np.cos(angle / 2), np.sin(angle / 2) * axis]   # birim quaternion, skaler önde (Sıfırdan yaz: slerp de)
```

İki `Rz` satırı aynı matrisin iki okunuşudur ve çalıştırma tam bu vektörleri yazdırıyor:

| Okunuş | Formül | Anlamı |
|---|---|---|
| Active | $p' = R\,p$ | $p$ vektörünü döndür |
| Passive | $p_B = R_{AB}^\top\, p_A$ | aynı nokta, $R_{AB}$ kadar dönmüş bir $B$ frame'indeki koordinatları |

| Gösterim | Formül | Kötü pozlar |
|---|---|---|
| Rotation matrix | $R^\top R = I,\ \det R = +1$ | yok |
| Euler açıları z-y'-x'' | $R = R_z(\psi)R_y(\theta)R_x(\phi)$ | pitch $\theta = \pm 90°$ (gimbal lock) |
| Axis-angle (Rodrigues) | $R = I + \sin\theta\,K + (1-\cos\theta)\,K^2$ | – |
| Birim quaternion | $q = (\cos\tfrac{\theta}{2},\ \sin\tfrac{\theta}{2}\,\hat k)$ | yok, ama $q$ ile $-q$ aynı rotation'dır |

$R$, sütunları dönmüş frame'in eksenlerinin üst frame'deki ifadesi olan 3×3'lük matristir; $\psi, \theta, \phi$ yaw, pitch ve roll'dur; axis-angle satırlarında $\hat k$ birim dönme ekseni, $\theta$ açıdır; $K = [\hat k]_\times$ çapraz çarpımı hesaplayan matristir, $Kv = \hat k \times v$; $q$ skaler önde yazılır.

Bileşke, matris çarpımıdır ve sıra **değiştirilemez**: $R_z(90°)R_x(90°)$ ile $R_x(90°)R_z(90°)$ arasında 120°'lik bir rotation fark vardır (aşağıda ölçüldü). $R_{AC} = R_{AB} R_{BC}$'de, soldan sağa okununca, her yeni rotation yeni ulaşılan frame'in eksenleri etrafındadır (intrinsic). İki orientation'ı karıştırmak için Euler açılarında düz bir çizgi değil, en kısa yayı izleyen **slerp** (spherical linear interpolation) kullan.

Euler açılarının neden bozulduğu tek satırda: Euler açısı hızlarını cismin açısal hızına çeviren $E$ matrisinin $\det E = \cos\theta$'dır ve pitch ±90°'de sıfırdır.

<details>
<summary>Matematik ayrıntısı: Euler hız haritası, Rodrigues ve slerp</summary>

**Euler hız haritası.** Intrinsic z-y'-x'' için (yaw $\psi$, pitch $\theta$, roll $\phi$) gövde açısal hızı, açı hızlarına doğrusal bağlıdır:

$$\omega_b = E(\theta,\phi)\begin{bmatrix}\dot\phi\\ \dot\theta\\ \dot\psi\end{bmatrix},\qquad E=\begin{bmatrix}1&0&-\sin\theta\\0&\cos\phi&\sin\phi\cos\theta\\0&-\sin\phi&\cos\phi\cos\theta\end{bmatrix},\qquad \det E=\cos\theta .$$

$\theta = \pm 90°$ olduğunda birinci ve üçüncü sütun paralel olur: roll ve yaw cismi aynı eksen etrafında döndürür. Bu gimbal lock'tur; fiziksel rotation'ın değil, parametrelemenin singularity'sidir. Yakınında $E$ hâlâ tersinirdir ama neredeyse singular'dır; sıradan bir açısal hız üretmek için çok büyük açı hızları gerekir (Boz).

**Rodrigues formülü.** Her rotation, birim bir $\hat k$ ekseni etrafında bir $\theta$ açısıdır. $K = [\hat k]_\times$ (çapraz çarpım matrisi, skew-symmetric) ile $R = I + \sin\theta\,K + (1-\cos\theta)\,K^2$.

**Birim quaternion.** $q = (\cos\tfrac{\theta}{2},\ \sin\tfrac{\theta}{2}\,\hat k)$. Yarım açı yüzünden $\theta + 2\pi$ kadar dönmek aynı fiziksel rotation için $-q$ verir: **double cover**, $q$ ile $-q$ tek bir rotation'dır.

**Slerp** en kısa büyük çember yayı boyunca interpolasyon yapar: $\mathrm{slerp}(q_0,q_1,t) = \frac{\sin((1-t)\Omega)}{\sin\Omega}q_0 + \frac{\sin(t\Omega)}{\sin\Omega}q_1$, $\cos\Omega = q_0\cdot q_1$; $q_0\cdot q_1 < 0$ ise önce $q_1 \to -q_1$. Bu çevirme olmadan yay uzun yoldan dolaşabilir.

Daha derine: AIEFS [Linear Algebra Intuition](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/01-linear-algebra-intuition), [Matrix Transformations](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/03-matrix-transformations); $SO(3)$ (bütün rotation matrix'lerin grubu) ve exponential coordinate'ların titiz anlatımı Lynch & Park, *Modern Robotics*, Böl. 3'te.

</details>

### MuJoCo'da

MuJoCo quaternion'ı skaler önde yazar; sıfır rotation `1 0 0 0`'dır ([MuJoCo modeling: frame orientations](https://mujoco.readthedocs.io/en/stable/modeling.html)). Gösterimler arasında `mju_euler2Quat` (dizi string'i; küçük harf = intrinsic, büyük harf = extrinsic), `mju_axisAngle2Quat`, `mju_quat2Mat`, `mju_mat2Quat`, `mju_mulQuat`, `mju_subQuat` ile dönüşüm yapar ([API reference](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html)). Baştaki bilmecenin cevabı da bu: orientation 4 quaternion sayısı, açısal hız bir 3-vektör olarak saklanır; bu yüzden `nq` ile `nv` farklıdır.

<!-- figure: 06-gimbal-lock -->

## Sıfırdan yaz

Şimdi sonraki her dersin kullandığı rotation araç kutusunu, quaternion cebirini, dönüşümleri, Euler açılarını ve slerp'i yaklaşık 80 satır NumPy ile yazacak ve her fonksiyonu MuJoCo'nun `mju_*` rutinleri ve SciPy ile karşılaştıracaksın. Adımları kendi dosyana yaz ve her adımdan sonra çalıştır. Bitmiş fonksiyonlar [`physai/spatial.py`](../../physai/spatial.py) içinde, ders 09'un da kullandığı `quat_to_mat` ve `axis_angle_to_mat`'in (Rodrigues formülü) yanında; `uv run python experiments/06-rotations/build.py` beş adımı kontrolleriyle birlikte koşturur (çıktı [`build_output.txt`](../../experiments/06-rotations/build_output.txt) dosyasında).

### Adım 1 · Quaternion cebiri

Birim quaternion $q = (w, x, y, z)$ Hamilton çarpımıyla birleşir, vektör kısmını ters çevirerek tersini alır ve bir vektörü iki yanından sararak döndürür:

```python
def quat_mul(a, b):
    """Hamilton product a * b: the rotation b followed by a (as matrices, R(a) @ R(b))."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])
```

```python
def quat_conj(q):
    """Conjugate (w, -x, -y, -z): the inverse rotation of a unit quaternion."""
    return np.array([q[0], -q[1], -q[2], -q[3]])
```

```python
def quat_rotate(q, v):
    """Rotate the 3-vector v by the unit quaternion q: the vector part of q * (0, v) * conj(q)."""
    return quat_mul(quat_mul(q, np.r_[0.0, v]), quat_conj(q))[1:]
```

Bir quaternion'ın matrisi, dosyada zaten olan `quat_to_mat`:

```python
def quat_to_mat(q):
    """Unit quaternion (w, x, y, z), MuJoCo's order, to a 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
```

```text
== Step 1. Quaternion algebra vs MuJoCo (1000 random rotations) ==
quat_mul    vs mju_mulQuat     2.2e-16
quat_conj   vs mju_negQuat     0.0e+00
quat_rotate vs mju_rotVecQuat  1.3e-15
R(a * b) vs R(a) @ R(b)        1.2e-15
R(-q) vs R(q)                  0.0e+00
```

Çarpım sırası matris sırasıyla aynı, $R(a\,b) = R(a)\,R(b)$, ve $-q$ tam olarak aynı matrisi veriyor (double cover).

### Adım 2 · Dönüşümler

Axis-angle'dan quaternion'a geçiş yarım açı formülüdür. Matristen quaternion'a geçiş dikkat ister: $w = \tfrac12\sqrt{1 + \operatorname{tr} R}$'yi okumak $w \to 0$ olan 180° civarında bozulur. Shepperd yöntemi $w, x, y, z$'den hangisi en büyükse onu seçer ve ona böler:

```python
def axis_angle_to_quat(axis, angle):
    """Unit quaternion for a rotation by `angle` (rad) about the unit vector `axis`."""
    return np.r_[np.cos(angle / 2), np.sin(angle / 2) * np.asarray(axis, float)]
```

```python
def mat_to_quat(R):
    """Rotation matrix to unit quaternion with w >= 0 (Shepperd: divide by the largest of 4w^2, 4x^2, 4y^2, 4z^2)."""
    t = np.trace(R)
    cands = [t, R[0, 0] - R[1, 1] - R[2, 2], R[1, 1] - R[0, 0] - R[2, 2], R[2, 2] - R[0, 0] - R[1, 1]]
    i = int(np.argmax(cands))
    s = 2 * np.sqrt(1 + cands[i])                      # 4 times the largest component
    if i == 0:
        q = [s / 4, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif i == 1:
        q = [(R[2, 1] - R[1, 2]) / s, s / 4, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif i == 2:
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, s / 4, (R[1, 2] + R[2, 1]) / s]
    else:
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, s / 4]
    q = np.array(q)
    return q if q[0] >= 0 else -q
```

```text
== Step 2. Axis-angle and matrix -> quaternion vs MuJoCo and SciPy ==
axis_angle_to_quat vs mju_axisAngle2Quat       1.1e-16
mat_to_quat vs mju_mat2Quat (w >= 0)           5.6e-16
mat_to_quat vs SciPy as_quat(scalar_first)     2.2e-16
quat_to_mat(mat_to_quat(R)) round trip         7.8e-16
180 deg about x: mat_to_quat gives [0. 1. 0. 0.] (trace -1: the x branch)
```

MuJoCo ve SciPy $-q$ döndürebilir; $w \ge 0$ ile karşılaştırmak bu işareti ortadan kaldırır.

### Adım 3 · İki yönde Euler açıları

```python
def euler_zyx_to_mat(yaw, pitch, roll):
    """Intrinsic z-y'-x'' Euler angles (rad) to a rotation matrix: Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])
```

```python
def mat_to_euler_zyx(R):
    """Rotation matrix to (yaw, pitch, roll), pitch in [-pi/2, pi/2]; at pitch = +-90 deg yaw and roll are not unique."""
    pitch = np.arcsin(np.clip(-R[2, 0], -1, 1))
    return np.arctan2(R[1, 0], R[0, 0]), pitch, np.arctan2(R[2, 1], R[2, 2])
```

```text
== Step 3. Euler z-y'-x'' vs mju_euler2Quat('zyx') and SciPy from_euler('ZYX') ==
vs MuJoCo 'zyx' (lowercase = intrinsic there)  8.9e-16
vs SciPy 'ZYX' (uppercase = intrinsic there)   5.6e-16
vs SciPy 'zyx' (extrinsic: a different rotation) 2.00
mat_to_euler_zyx round trip, angles            9.9e-14 rad
pitch 89.0 deg: (yaw, roll) = (0.7, 0.3) and (0.9, 0.5) give matrices 3.2e-03 apart
pitch 90.0 deg: (yaw, roll) = (0.7, 0.3) and (0.9, 0.5) give matrices 1.1e-16 apart
```

Aynı üç açı MuJoCo'nun küçük harfli `"zyx"`'inde de SciPy'ın büyük harfli `"ZYX"`'inde de $R_z R_y R_x$ demek: iki kütüphane intrinsic diziler için zıt harf büyüklüğü kullanıyor ve SciPy'ın küçük harfi farklı bir rotation veriyor. Son iki satır sayılarla gimbal lock: 90°'de iki farklı (yaw, roll) çifti tek bir matris veriyor, yani hiçbir çıkarım onları geri bulamaz.

### Adım 4 · Euler hız matrisi

```python
def euler_rate_matrix(pitch, roll):
    """E with body angular velocity w_b = E @ (roll', pitch', yaw') for z-y'-x'' angles; det E = cos(pitch)."""
    cp, sp, cr, sr = np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    return np.array([[1, 0, -sp], [0, cr, sr * cp], [0, -sr, cr * cp]])
```

Her sütunu finite difference ile kontrol et: bir açıyı biraz oynat; gövdenin açısal hızı $R^\top \dot R$'nin skew kısmıdır.

```text
== Step 4. The Euler-rate matrix E (roll = 30 deg) ==
pitch  0.00 deg: det E 1.000e+00 (cos 1.000e+00) · cond      1.00 · |E - E_fd| 5e-11
pitch 60.00 deg: det E 5.000e-01 (cos 5.000e-01) · cond      3.73 · |E - E_fd| 8e-11
pitch 80.00 deg: det E 1.736e-01 (cos 1.736e-01) · cond     11.43 · |E - E_fd| 8e-11
pitch 89.00 deg: det E 1.745e-02 (cos 1.745e-02) · cond    114.59 · |E - E_fd| 7e-11
pitch 89.90 deg: det E 1.745e-03 (cos 1.745e-03) · cond   1145.92 · |E - E_fd| 7e-11
pitch 89.99 deg: det E 1.745e-04 (cos 1.745e-04) · cond  11459.16 · |E - E_fd| 7e-11
```

$\det E = \cos\theta$, yazılan her basamakta; condition number'lar da Deney'in Adım 2 tablosu (80°'de 11,43, 89°'de 114,59).

### Adım 5 · Slerp

```python
def slerp(q0, q1, t, shortest=True):
    """Spherical linear interpolation; with shortest=True flip q1 when q0 . q1 < 0 (q1 and -q1 are one rotation)."""
    dot = q0 @ q1
    if shortest and dot < 0:
        q1, dot = -q1, -dot
    om = np.arccos(np.clip(dot, -1, 1))
    if om < 1e-9:
        return q0.copy()
    return (np.sin((1 - t) * om) * q0 + np.sin(t * om) * q1) / np.sin(om)
```

```python
def quat_angle(q0, q1):
    """Angle (rad) of the rotation that takes orientation q0 to q1, in [0, pi]; q and -q give the same answer."""
    return 2 * np.arccos(np.clip(abs(q0 @ q1), 0, 1))
```

Aralarında 60° olan 300 çifti, yarısında $q_1$'i $-q_1$ ile değiştirerek interpole et ve 100 küçük adımda kat edilen rotation'ı topla:

```text
== Step 5. Slerp vs SciPy, and the sign check (300 pairs, 60 deg apart) ==
slerp(q0, q1, 0.3) vs SciPy Slerp (w >= 0): 3.3e-16 · pairs with q0.q1 < 0: 143
sign check True : rotation travelled min   60.0 deg, max   60.0 deg
sign check False: rotation travelled min   60.0 deg, max  300.0 deg
```

Deney'in Boz bölümü, kendi slerp'inden: işaret kontrolü 60° ile 300° arasındaki fark. (Bu çekilişte 143 negatif çift var, Deney'de 156.)

### Elinde kalan

`physai.spatial` artık `quat_mul`, `quat_conj`, `quat_rotate`, `axis_angle_to_quat`, `mat_to_quat`, `euler_zyx_to_mat`, `mat_to_euler_zyx`, `euler_rate_matrix`, `quat_angle` ve `slerp` içeriyor. Ders 07 rigid transform'ları bunların üzerine kuruyor; ders 09 gövde orientation'larını `quat_to_mat` ile okuyor. [`tests/test_06_rotations.py`](../../tests/test_06_rotations.py) bunları MuJoCo ve SciPy'a sabitliyor; bir şey değiştirdikten sonra `uv run pytest tests/test_06_rotations.py` çalıştır.

## Kütüphaneyle kullan

MuJoCo'nun `mju_*` fonksiyonları sonucu önceden ayrılmış bir çıktı dizisine, ilk argümana yazar:

```python
q = np.zeros(4)
mujoco.mju_euler2Quat(q, np.array([yaw, pitch, roll]), "zyx")   # intrinsic z-y'-x''
R = np.zeros(9); mujoco.mju_quat2Mat(R, q)                      # row-major 3x3, flattened
mujoco.mju_mulQuat(q_ab, q_a, q_b)                              # q_a * q_b
mujoco.mju_rotVecQuat(v_out, v, q)

from scipy.spatial.transform import Rotation
r = Rotation.from_quat(q, scalar_first=True)                    # SciPy's default order is (x, y, z, w)
r.as_euler("ZYX")                                               # uppercase = intrinsic in SciPy
```

| Araç | Quaternion sırası | Intrinsic z-y'-x'' |
|---|---|---|
| physai, MuJoCo | $(w, x, y, z)$ | `euler_zyx_to_mat`; `mju_euler2Quat(q, e, "zyx")` |
| SciPy `Rotation` | `scalar_first=True` verilmezse $(x, y, z, w)$ | `from_euler("ZYX", e)`; küçük harf extrinsic |
| MJCF `euler="..."` | – | `<compiler eulerseq="zyx"/>`; default `"xyz"` (yukarıda, MuJoCo'da) |

Tuzaklar:

- **Harf büyüklüğü anlamı çevirir.** Adım 3: `"zyx"` MuJoCo'da intrinsic, SciPy'da extrinsic ve hiçbir şey hata vermez.
- **Matrisler düzdür.** `mju_quat2Mat` ve `d.xmat` 9 sayıyı row-major sırada tutar; çarpmadan önce `reshape(3, 3)`.
- **Quaternion işaretleri.** İki doğru dönüşüm $q$ ve $-q$ döndürebilir; rotation'ları dört sayı üzerinde `np.allclose` ile değil `quat_angle` ile karşılaştır.

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

**Adım 2 · pitch sweep'i → $E$'nin koşullanması.** **Condition number** cond(E), $E$'yi ters çevirdiğinde göreli hataları ne kadar büyütebileceğini söyler: 1 mükemmeldir, sonsuz bir yönün kaybolduğu anlamına gelir. σ_min, yani en küçük singular value, $E$'nin en zayıf yönü ne kadar küçülttüğüdür. Analitik $E$, $R$'nin merkezi farklarıyla kontrol edildi (her satırda uyum ≤ 9,6e-11). Roll 30°'de sabit.

| pitch | det E | σ_min | cond(E) |
|---|---|---|---|
| 0° | 1,00 | 1,00 | 1,000 |
| 60° | 0,500 | 0,366 | 3,732 |
| 80° | 0,174 | 0,123 | 11,43 |
| 89° | 0,0175 | 0,0123 | 114,6 |
| 89,9° | 0,00175 | 0,00123 | 1146 |
| 89,99° | 0,000175 | 0,000123 | 11460 |
| 90° | 6,12e-17 | 3,40e-17 | sonsuz |

Condition number $1/\cos\theta$ gibi büyüyor, 80°'de bile 11,43: sorun bir nokta değil, bir komşuluk.

**Adım 3 · başlangıç pitch'i sweep'i → interpolasyon hatası.** Her satırda 300 çift; hedef, başlangıcın rastgele bir eksen etrafında 60° döndürülmüş hâli. Saf interpolasyon (yaw, pitch, roll) üzerinde doğrusal; yaw ve roll kısa yoldan sarılıyor; referans slerp. "sapma" iki yol arasındaki en büyük açı, "yol/60" Euler yolunun gerçekte ne kadar rotation katettiği.

| başlangıç pitch | medyan sapma | p95 sapma | en büyük sapma | medyan yol/60° | sapması > 90° olan çift |
|---|---|---|---|---|---|
| 0° | 3,71° | 6,30° | 6,95° | 1,010 | 0 |
| 30° | 4,44° | 14,88° | 20,62° | 1,014 | 0 |
| 60° | 7,83° | 23,91° | 179,74° | 1,042 | 7 |
| 85° | 14,96° | 178,59° | 179,83° | 1,149 | 22 |
| 89° | 16,99° | 178,73° | 179,99° | 1,190 | 23 |

![Solda: Euler hız haritasının condition number'ı ve 1 rad/s'lik gövde rotation'ı için gereken Euler hızı; pitch 90 dereceye her on kat yaklaşmada ikisi de on kat artıyor. Sağda: saf Euler interpolasyonunun slerp'ten sapmasının başlangıç pitch'ine göre değişimi; en büyük sapma 60 derece pitch'ten itibaren, 95. yüzdelik 85 dereceden itibaren yaklaşık 180 dereceye sıçrıyor.](../../experiments/06-rotations/euler_breakdown.png)

Pitch 0'da bile Euler yolu en kısa yol değil (medyan 3,71° sapma). 90° yakınında hedefin pitch'i ±90° sınırını geçebiliyor; çıkarılan yaw ve roll yaklaşık 180° sıçrıyor ve interpolasyonla bulunan orientation uzun yoldan dolaşıyor: katetmesi gereken 60°'nin 5,595 katına kadar.

## Boz

**Euler hızları patlar.** $E$'nin 90°'de kaybettiği yön etrafında (roll 30° için $\omega_b \propto (0, \sin\phi, \cos\phi)$) 1 rad/s'lik gövde rotation'ı iste ve $E\,\dot{\boldsymbol\alpha} = \omega_b$'yi çöz:

| pitch | 0° | 60° | 85° | 89° | 89,9° | 89,99° |
|---|---|---|---|---|---|---|
| gereken ‖Euler hızı‖ (rad/s) | 1,0 | 2,6 | 16,2 | 81,0 | 810,3 | 8102,8 |

Euler açısı hızı komut eden bir controller, tamamen sıradan bir hareket için actuator'larını doyuma sokar (hız ya da tork sınırına dayandırır). Aynı mekanizmayı [ders 10](../part-3-kinematics/10-inverse-kinematics.md)'da kinematik singularity olarak göreceksin: hızları hızlara eşleyen bir matris rank kaybediyor.

**İşaret kontrolü olmadan slerp.** Aynı şekilde 60° aralıklı 300 çift, ama $q_1$ rastgele olarak $-q_1$ ile değiştiriliyor (aynı rotation). 300 çiftin 156'sında $q_0\cdot q_1 < 0$ çıktı. Kontrolle her yol 60,0° katediyor; kontrolsüz en kötü yol 300,0° katediyor: uzun yoldan, çünkü quaternion yayı $360° - 60°$'lik rotation'ı kapsıyor.

## Gerçek robotta

- SO-101'in joint'leri hinge'dir (her biri sabit tek bir eksen etrafında döner); joint açıları hiçbir zaman gimbal lock yaşamaz. Sorun bir üst seviyede çıkar: gripper orientation hedefini roll-pitch-yaw olarak yazdığın her yerde (teleoperation arayüzleri, dataset etiketleri, policy action'ları). Quaternion sakla ve quaternion ile interpolasyon yap; Euler açısına yalnızca bir insana göstermek için çevir.
- Araçlar arasında konvansiyonlar farklıdır. MuJoCo skaleri öne koyar; `eulerseq` compiler varsayılanı `"xyz"`'dir ve dokümantasyon URDF'nin "rpy"sinin MJCF'te `"XYZ"`'ye (extrinsic) karşılık geldiğini söyler ([XML reference: compiler/eulerseq](https://mujoco.readthedocs.io/en/stable/XMLreference.html#compiler-eulerseq)). Birçok başka kütüphane skaleri sona koyar. Kaydedilmiş bir orientation'a güvenmeden önce bilinen bir rotation'ı çevirip kontrol et.
- IMU'lar ve kamera tracker'ları gürültülü orientation tahmini verir. Bu dersteki hiçbir şey donanımda ölçülmedi; yukarıdaki sayılar double precision'da saf matematiktir.

## Kendini dene

1. **Kolay.** $R_x(90°)^\top$, x etrafında −90°'lik active rotation mı, yoksa +90° dönmüş bir frame'e passive koordinat değişimi mi? (İkisini de cevapla.)
2. **Orta.** Adım 2'yi extrinsic `"XYZ"` dizisi (her dönüş sabit üst eksenler etrafında) için tekrarla. Hız haritası hangi pitch'te singular olur?
3. **Zor.** `interp_stats`'ı, saf interpolasyon hedefin Euler üçlüsünü önce başlangıca en yakın dala açacak şekilde değiştir ($(\psi,\theta,\phi)$ ve $(\psi\pm\pi, \pi-\theta, \phi\pm\pi)$ ikisini de dene). Başlangıç pitch'i 85°'de en büyük sapma 90°'nin altına iner mi? Medyan iner mi?

### Cevaplar

1. İkisi de: active rotation olarak $R_x(90°)^\top = R_x(-90°)$'dir; aynı zamanda üst frame'deki koordinatları x etrafında +90° dönmüş bir frame'e taşır.
2. Sabit eksenler etrafında önce X, sonra Y, sonra Z, $R = R_z(c)R_y(b)R_x(a)$ verir: intrinsic z-y'-x'' ile aynı matris. Bu yüzden aynı yerde, ortadaki açı ±90° iken singular olur. Extrinsic diziyi `mju_euler2Quat`'a verip matrisleri karşılaştırarak kontrol et.
3. Bu dersin script'inde hesaplanmadı. Çalıştır ve Adım 3 tablosuyla karşılaştır; ne bulursan bul, pitch 0'daki medyan sapma sıfıra inemez, çünkü Euler açılarında düz bir çizgi $SO(3)$'te geodesic (en kısa yol) değildir.

## Akılda kalanlar ve kaynaklar

- Rotation matrix ve birim quaternion singularity içermez; Euler açılarının hız haritasında $\det E = \cos\theta$, 80° pitch'te cond zaten 11,43.
- Euler açıları arasında asla interpolasyon yapma; $q_0\cdot q_1<0$ çevirmesiyle slerp gerçek 60,0°'yi katediyor, saf Euler yolları bunun 5,595 katına kadar katetti.
- Konvansiyonunu bil: skaler önde ya da sonda, intrinsic (küçük harf) ya da extrinsic (büyük harf), active ya da passive.

Sonraki: [ders 07 · Rigid transform'lar ve frame tree'leri](07-rigid-transforms.md).

**Kaynaklar**

- K. M. Lynch, F. C. Park, *Modern Robotics: Mechanics, Planning, and Control*, Cambridge Univ. Press, 2017, Böl. 3 "Rigid-Body Motions". [Kitap sayfası ve ücretsiz preprint](https://hades.mech.northwestern.edu/index.php/Modern_Robotics).
- MuJoCo dokümantasyonu: [Modeling · frame orientations](https://mujoco.readthedocs.io/en/stable/modeling.html), [XML reference · compiler/eulerseq](https://mujoco.readthedocs.io/en/stable/XMLreference.html#compiler-eulerseq), [API functions (`mju_*` matematik)](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html).
