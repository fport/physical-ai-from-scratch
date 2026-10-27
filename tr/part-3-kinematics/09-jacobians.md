# 09 · Jacobian ve differential kinematics

<!-- meta -->
| | |
|---|---|
| **Bölüm** | [3 · Kinematics](README.md) |
| **Çalışma türü** | CPU · Mac ya da Colab CPU |
| **Önkoşul** | [08](08-forward-kinematics.md) |
| **Önce öğren** | [Calculus for Machine Learning](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/04-calculus-for-ml) · [Linear Systems](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/17-linear-systems) |
| **Süre** | ~45 dk |
| **Kod** | [`experiments/09-jacobians/experiment.py`](../../experiments/09-jacobians/experiment.py) |
| **Durum** | ✅ tam |
<!-- /meta -->

> Tek bir matris joint hızlarını alet hızına çevirir; transpozu da alet kuvvetlerini joint torklarına geri çevirir.

## Problem

SO-101'in omuz servosu gripper'daki 2 N'luk bir itmeyi tutmak için ne kadar tork ister? Joint'ler toplamda 1 rad/s ile dönerken gripper ne kadar hızlı hareket eder? İki sorunun cevabı aynı: her pozda değişen 6×5'lik bir matris. Menagerie modeli her servoyu 2,94 N m ile sınırlıyor ([`so101.xml`](https://github.com/google-deepmind/mujoco_menagerie/blob/f3b6013/robotstudio_so101/so101.xml) içinde `forcerange="-2.94 2.94"`); bir yükün güvenli olup olmadığı yalnızca yüke değil, o matrise bağlı.

Bu matrisin tam doğru olması da gerekiyor: ders 10 onunla yüzlerce kez iterasyon yapıyor. Yaygın bir kısayol onu finite difference ile hesaplamaktır. Bu ders o kısayolun ne zaman doğru, ne zaman yanlış olduğunu ölçüyor.

## Ana fikir

Forward kinematics'in türevini al. Doğrusal hızı $v$, açısal hızı $\omega$ olan alet site'ı için (ikisi de world koordinatlarında):

$$\begin{bmatrix} v \\ \omega \end{bmatrix} = J(q)\,\dot q = \begin{bmatrix} J_p(q) \\ J_r(q) \end{bmatrix}\dot q ,\qquad J \in \mathbb R^{6\times n}.$$

World ekseni $\hat a_i$ olan ve $o_i$ noktasından geçen bir $i$ hinge'i için $i$. sütun $J_{p,i} = \hat a_i \times (p - o_i)$, $J_{r,i} = \hat a_i$'dir: **geometric Jacobian** (Lynch & Park, *Modern Robotics*, Böl. 5). MuJoCo onu `mj_jac` (bir body'ye bağlı herhangi bir world noktası), `mj_jacBody` ve `mj_jacSite` ile döndürür ([API reference](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html)).

**Statik kuvvet haritası.** Virtual work ilkesiyle, aletin çevreye bir $F$ kuvveti ve $M$ momenti uygulaması için joint'lerin sağlaması gereken tork:

$$\tau = J^\top \begin{bmatrix} F \\ M \end{bmatrix};$$

aleti iten bir dış $(F, M)$ wrench'i ise $-J^\top[F;M]$ ile tutulur.

**Finite difference**, $i$. sütunu $\frac{x(q + \varepsilon e_i) - x(q)}{\varepsilon}$ (ileri, hata $O(\varepsilon)$) ya da $\frac{x(q+\varepsilon e_i) - x(q-\varepsilon e_i)}{2\varepsilon}$ (merkezi, hata $O(\varepsilon^2)$) ile yaklaşık hesaplar. Kayan nokta yuvarlaması yaklaşık $\epsilon_{\text{mach}}\lvert x\rvert/\varepsilon$ kadar hata ekler; bu yüzden her yöntemin en iyi bir adımı vardır: ileri farkta kabaca $\sqrt{\epsilon_{\text{mach}}}$, merkezi farkta $\sqrt[3]{\epsilon_{\text{mach}}}$. Türev ve doğrusal haritalar: AIEFS [Calculus for Machine Learning](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/04-calculus-for-ml), [Linear Systems](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/17-linear-systems).

## Deney

Script: [`experiments/09-jacobians/experiment.py`](../../experiments/09-jacobians/experiment.py). MuJoCo 3.17.0, NumPy 2.5.4, Apple Silicon Mac (Apple M5), deterministik, birkaç saniye.

```python
mujoco.mj_kinematics(m, d)
mujoco.mj_comPos(m, d)                     # fills the data mj_jac* reads
jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
mujoco.mj_jacSite(m, d, jp, jr, site_id)   # 3 x nv each, world frame
```

**Önce tahmin et.** Daha küçük bir finite difference adımı her zaman daha mı iyi? Hangi adımı seçerdin?

**Adım 1 · hız haritası.** 50 rastgele poz, rastgele birim joint hızları. Tahmin edilen $J\dot q$ ile site'ın $10^{-6}$ s'deki gerçek hareketi: en büyük bağıl hata 7,9e-07 (doğrusal) ve 3,8e-07 (açısal). Her joint $1/\sqrt5$ rad/s'de ($\lvert\dot q\rvert = 1$) iken medyan site hızı 0,306 m/s.

**Adım 2 · ε adımı sweep'i.** 50 pozun medyanı; `mj_jacSite`'a göre en büyük eleman hatası. Orientation farkları `mju_subQuat` ile alınıp world frame'ine döndürülüyor.

| ε (rad) | ileri $J_p$ | ileri $J_r$ | merkezi $J_p$ | merkezi $J_r$ |
|---|---|---|---|---|
| 1e-1 | 1,2e-02 | 4,8e-15 | 4,2e-04 | 2,6e-15 |
| 1e-3 | 1,2e-04 | 4,5e-13 | 4,2e-08 | 2,5e-13 |
| 1e-5 | 1,2e-06 | 4,7e-11 | **5,6e-12** | 2,3e-11 |
| 1e-7 | 1,2e-08 | 4,5e-09 | 4,7e-10 | 2,3e-09 |
| 1e-8 | **9,8e-09** | 4,4e-08 | 4,9e-09 | 2,3e-08 |
| 1e-10 | 9,5e-07 | 4,4e-06 | 4,3e-07 | 2,2e-06 |
| 1e-12 | 8,9e-05 | 4,6e-04 | 4,7e-05 | 2,4e-04 |
| 1e-14 | 8,3e-03 | 4,5e-02 | 4,5e-03 | 2,6e-02 |

Öteleme hatası V biçiminde. Minimumun sağında truncation (ε'un her on kat büyümesinde hata ileri farkta 10 kat, merkezi farkta 100 kat artıyor); solunda yuvarlama (ε'un her on kat küçülmesinde hata 10 kat artıyor). En iyi ileri adım 1e-8 (hata 9,8e-09), en iyi merkezi adım 1e-5 (hata 5,6e-12). Karşılaştırma için $\sqrt{\epsilon_{\text{mach}}} = $ 1,5e-08 ve $\sqrt[3]{\epsilon_{\text{mach}}} = $ 6,1e-06.

Rotation sütunlarında **hiç truncation hatası yok**: bir hinge'i ε kadar oynatmak site'ı tam olarak o hinge'in ekseni etrafında ε kadar döndürür; fark oranı kesindir, geriye yalnızca yuvarlama kalır (ε = 0,1'de 4,8e-15, ε küçüldükçe büyüyor).

**Adım 3 · simülasyonda τ = Jᵀ F doğrulaması.** Yalnızca dış kuvvet etki etsin diye yerçekimi ve joint friction loss kapatıldı. Kol position servo'larıyla bir pozu tutuyor; gripper site'ına `xfrc_applied` ile sabit bir kuvvet uygulanıyor (kuvvet artı $r\times F$ momenti, çünkü `xfrc_applied` body'nin kütle merkezinde etki eder). 20 s'lik simülasyondan sonra joint'ler duruyor (en büyük |q̇| = 8,5e-15 rad/s) ve servo torkları $J^\top F$ ile karşılaştırılıyor:

| site'taki F (N) | $J^\top F$ (N m), joint 1–5 | servo torku (N m), işaret çevrilmiş | en büyük fark |
|---|---|---|---|
| (0, 0, −2) | 0,0000, 0,4383, 0,5191, 0,2584, 0,0042 | 0,0000, 0,4383, 0,5191, 0,2584, 0,0042 | 4,2e-14 |
| (2, 0, 0) | 0,0052, −0,0395, −0,2569, −0,1872, −0,0036 | aynı | 7,1e-14 |
| (0, 1,5, 0) | −0,3745, −0,0000, −0,0001, −0,0001, 0,0175 | aynı | 7,2e-14 |
| (1, −1, 1) | 0,2523, −0,2390, −0,3882, −0,2229, −0,0156 | aynı | 1,6e-13 |

Servo'lar tam olarak $-J^\top F$'ye oturuyor. Yana doğru bir itme (0, 1,5, 0) N neredeyse yalnızca pan joint'ini yüklüyor; aşağı doğru itme üç pitch joint'ini yüklüyor.

## Boz

**Uç yükünü bir servo doyana kadar artır.** Yerçekimi açıkken servo'lar, aşağı doğru bir $f$ uç kuvveti için $\tau = g(q) - f\,J_p^\top(-\hat z)$ sağlamak zorunda. Her joint için modelin 2,94 N m limitini alarak:

| Poz (rad) | lift ekseni → uç (m) | yerçekimi torku, lift / elbow (N m) | en büyük ek uç yükü | sınırlayan joint |
|---|---|---|---|---|
| yakın (0, −1,5, 1,5, 0,5, 0) | 0,172 | −0,059 / −0,469 | 8,81 N | elbow_flex |
| orta (0, −0,6, 0,9, 0,4, 0) | 0,220 | −0,281 / −0,443 | 9,61 N | elbow_flex |
| gergin (0, 1,3, −1,3, 0, 0) | 0,411 | −0,909 / −0,483 | 4,95 N | shoulder_lift |

Sonra gergin durum, kuvvet bu limitin bir kesrine ayarlanarak 5 s simüle ediliyor:

| uç kuvveti (N) | × limit | 5 s sonra uç düşüşü | lift servo torku |
|---|---|---|---|
| 0,00 | 0,0 | 0,5 mm | −0,909 N m |
| 2,48 | 0,5 | 1,2 mm | −1,925 N m |
| 4,46 | 0,9 | 1,8 mm | −2,737 N m |
| 5,45 | 1,1 | 147,4 mm | −2,940 N m |
| 7,43 | 1,5 | 179,7 mm | −2,940 N m |

Limitin altında kol yaklaşık bir milimetre sarkıyor (sonlu servo stiffness'ı). Limitin yüzde on üstünde lift servo'su −2,940 N m'de kırpılıyor ve kol 147,4 mm düşüyor. Kolu 0,220 m'den 0,411 m'ye uzatmak güvenli yükü kabaca yarıya indirdi: limiti yük değil, Jacobian belirliyor.

## Gerçek robotta

- 2,94 N m'lik `forcerange` bir model parametresidir. MJCF yorumu, position gain'lerinin ve kuvvetlerin servo proportional gain'i 16 varsayılarak hesaplandığını ve bunların LeRobot'taki servo gain'lerinin "not a 1-to-1 mapping" olduğunu söylüyor ([so101.xml](https://github.com/google-deepmind/mujoco_menagerie/blob/f3b6013/robotstudio_so101/so101.xml)). Yük tablosunu bir donanım değeri gibi okuma. Gerçek kolda henüz ölçülmedi. Prosedür: gergin pozda gripper'a bilinen ağırlıklar as, joint torkunu kaydet (örneğin sürücün sunuyorsa servo'nun akım okumasından) ve $J^\top F$'ye fit et.
- Donanımda statik harita ayrıca joint sürtünmesini (modelin `frictionloss`'u, Adım 3'te kapatıldı), dişli boşluğunu ve kablo kuvvetlerini taşır; $\tau = J^\top F$'nin ancak bunlar kadar tutmasını bekle.
- Jacobian'ı yine de finite difference ile hesaplıyorsan (örneğin öğrenilmiş bir modelin etrafında), 1e-8 değil, 1e-5 rad civarında ε ile merkezi fark kullan.

## Kendini dene

1. **Kolay.** $F = (0, 0, -2)$ N için $J^\top F$'nin ilk elemanı neden tam 0,0000?
2. **Orta.** Hangi poz, aşağı doğru uç kuvvetinin newton'u başına shoulder lift torkunu en büyük yapar? "Lift ekseni → uç" sütunuyla savun, sonra `mj_jacSite` ile kontrol et.
3. **Zor.** ε sweep'ini float32'de tekrarla (farkı almadan önce konumları `np.float32`'ye çevir). En iyi adımlar nereye kayar, neden?

### Cevaplar

1. Joint 1'in (pan) ekseni düşey; düşey bir kuvvetin düşey bir eksen etrafında momenti yoktur, yani $\hat a_1 \cdot (r \times F) = 0$.
2. Tamamen gergin yatay poz: newton başına lift torku, lift ekseninden uca yatay mesafeye eşittir; script çıktısında gergin pozda 0,410 N m/N. Diğer pozları script ile kontrol et.
3. Burada hesaplanmadı. Minimumların daha büyük ε'a, $\epsilon_{32} \approx 1{,}2\times10^{-7}$ (float32 machine epsilon) ile $\sqrt{\epsilon_{32}}$ ve $\sqrt[3]{\epsilon_{32}}$ civarına kaymasını bekle.

## Akılda kalanlar ve kaynaklar

- `mj_jacSite` geometric Jacobian'ı verir; $J\dot q$ gerçek site hızıyla 7,9e-07 bağıl hatayla tuttu.
- Finite difference'ın en iyi bir adımı vardır: bu sweep'te 1e-8 (ileri, 9,8e-09) ve 1e-5 (merkezi, 5,6e-12); daha küçüğü daha kötü.
- $\tau = J^\top F$ simülasyonda 1,6e-13 N m'ye kadar tuttu; 0,411 m'lik kolda modelin servo'ları yalnızca 4,95 N uç yükü taşıyor.

Sonraki: [ders 10 · Inverse kinematics ve singularity](10-inverse-kinematics.md).

**Kaynaklar**

- K. M. Lynch, F. C. Park, *Modern Robotics*, Böl. 5 "Velocity Kinematics and Statics". [Kitap sayfası](https://hades.mech.northwestern.edu/index.php/Modern_Robotics).
- MuJoCo [API functions](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html) (`mj_jac`, `mj_jacSite`, `mju_subQuat`).
- SO-101 modeli ve servo parametreleri: [MuJoCo Menagerie `robotstudio_so101`](https://github.com/google-deepmind/mujoco_menagerie/blob/f3b6013/robotstudio_so101/so101.xml).
