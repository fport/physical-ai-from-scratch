# 10 · Inverse kinematics ve singularity

<!-- meta -->
| | |
|---|---|
| **Bölüm** | [3 · Kinematics](README.md) |
| **Çalışma türü** | CPU · Mac ya da Colab CPU |
| **Önkoşul** | [09](09-jacobians.md) |
| **Önce öğren** | [Singular Value Decomposition](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/11-singular-value-decomposition) · [Optimization](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/08-optimization) · [Linear Systems](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/17-linear-systems) |
| **Süre** | ~60 dk |
| **Kod** | [`experiments/10-inverse-kinematics/experiment.py`](../../experiments/10-inverse-kinematics/experiment.py) |
| **Durum** | ✅ tam |
<!-- /meta -->

> Inverse kinematics, forward kinematics üzerinde Newton yöntemidir; Jacobian'ın rank kaybettiği yerde onu patlamaktan koruyan şey damping'dir.

## Problem

Bir kavrama planlayıcısı diyor ki: SO-101'in gripper'ını (0,20, 0,05, 0,05) m'ye koy, parmaklar aşağı baksın. Servo'ların beş joint açısına ihtiyacı var. Forward kinematics (ders 08) ters yönde çalışır. Bu geometri için kapalı form tersler yalnızca özel durumlarda vardır ve joint limitlerini hesaba katmaz; bu yüzden problemi genelde ders 09'daki Jacobian ile sayısal olarak çözersin ve üç şey ters gidebilir: iterasyon ıraksar, joint limitlerinin dışında bir çözüme yakınsar ya da gergin bir kolun yakınında saçma joint hızları ister.

Bu ders küçük, yeniden kullanılabilir bir `ik_solve` yazıyor ve her başarısızlığı ölçüyor.

## Ana fikir

$e = x^\* - x(q)$ görev hatası olsun. $x(q + \Delta q) \approx x(q) + J\Delta q$ olarak doğrusallaştır, $\Delta q$'yu çöz ve tekrarla:

| Yöntem | Adım $\Delta q$ | Kaynak |
|---|---|---|
| Newton-Raphson (pseudo-inverse) | $J^{+}e$ | *Modern Robotics*, Böl. 6 |
| Jacobian transpose | $\alpha J^\top e$, $\ \alpha = \dfrac{\langle e, JJ^\top e\rangle}{\lVert JJ^\top e\rVert^2}$ | Buss 2009, §3 |
| Damped least squares (DLS) | $J^\top (JJ^\top + \lambda^2 I)^{-1} e$ | Buss 2009, §5 |

DLS, $\lVert J\Delta q - e\rVert^2 + \lambda^2\lVert\Delta q\rVert^2$'yi minimize eder. SVD $J = \sum_i \sigma_i u_i v_i^\top$ ile fark her yön için tek bir çarpandır:

$$J^{+}e = \sum_i \frac{1}{\sigma_i}\,v_i u_i^\top e, \qquad \Delta q_{\text{DLS}} = \sum_i \frac{\sigma_i}{\sigma_i^2+\lambda^2}\,v_i u_i^\top e .$$

$\sigma_i \to 0$ iken pseudo-inverse kazancı $1/\sigma_i$ patlar; DLS kazancı sıfıra gider. Buss, λ'nın singularity'ler yakınında kararlılık için yeterince büyük olması gerektiğini, ama fazla büyük bir değerin yakınsamayı yavaşlattığını not eder ([Buss 2009](https://mathweb.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf)).

**Singularity ve manipulability.** $J$ rank kaybettiğinde poz singular'dır. Yoshikawa'nın manipulability ölçüsü $w = \sqrt{\det(JJ^\top)} = \prod_i \sigma_i$, birim joint hızlarının ürettiği hız elipsoidinin hacmiyle orantılıdır ve singularity'de sıfıra düşer ([Yoshikawa 1985](https://doi.org/10.1177/027836498500400201)). SVD ve least squares AIEFS'te: [Singular Value Decomposition](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/11-singular-value-decomposition), [Optimization](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/08-optimization).

**Joint limitleri** burada her adımdan sonra kırpılarak uygulanıyor. Görevin denklem sayısı kolun joint sayısından az olduğunda ya da geometri katlandığında (elbow up / elbow down) **birden fazla çözüm** vardır.

## Deney

Script: [`experiments/10-inverse-kinematics/experiment.py`](../../experiments/10-inverse-kinematics/experiment.py). MuJoCo 3.17.0, NumPy 2.5.4, Apple Silicon Mac (Apple M5), deterministik, yaklaşık 6 s CPU süresi.

Çözücü sonraki derslerden import edilebilir (deney kodu `if __name__ == "__main__":` altında):

```python
sys.path.insert(0, "experiments/10-inverse-kinematics")
from experiment import ik_solve
res = ik_solve(model, data, target_pos, method="dls", damping=0.1)   # res.q, res.success, res.iters
```

Çekirdek döngüsü:

```python
e = target_pos - data.site_xpos[sid]
mujoco.mj_jacSite(model, data, jp, jr, sid); J = jp[:, dadr]
dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(len(e)), e)   # DLS
q = np.clip(data.qpos[qadr] + dq, lo, hi)                              # joint limits
```

**Önce tahmin et.** En yüksek başarı oranını hangi λ verir: 0 (saf Newton), 0,01, 0,1 ya da 0,3 (birim metre, çünkü $J_p$ m/rad cinsinden)?

**Adım 1 · λ sweep'i.** 500 rastgele **erişilebilir** konum hedefi (limitler içindeki rastgele joint vektörlerinin FK'sı), seed $q_0 = 0$, başarı = 200 iterasyon içinde 0,5 mm'nin altında hata.

| Yöntem | başarı | medyan iterasyon | p95 iterasyon | medyan en büyük ‖Δq‖ (rad) | joint limitinde biten başarısızlık |
|---|---|---|---|---|---|
| Jacobian transpose | %85,6 | 35 | 151 | 0,699 | 72'de 24 |
| pseudo-inverse (λ = 0) | %73,2 | 6 | 34 | 4,901 | 134'te 132 |
| DLS λ = 0,001 | %73,2 | 6 | 34 | 4,900 | 134'te 130 |
| DLS λ = 0,01 | %76,0 | 6 | 34 | 4,737 | 120'de 120 |
| DLS λ = 0,03 | %84,6 | 7 | 30 | 3,920 | 77'de 77 |
| **DLS λ = 0,1** | **%88,6** | 15 | 92 | 1,598 | 57'de 56 |
| DLS λ = 0,3 | %74,6 | 64 | 167 | 0,485 | 127'de 50 |
| DLS λ = 0,1 + 4 rastgele yeniden başlatma | %96,8 | 17 (toplam) | – | – | – |

Az damping ile Newton birkaç radyanlık adımlar atıyor, kırpma bir joint'i limitine çiviliyor ve iterasyon orada takılıyor: neredeyse her başarısızlık bir limitte bitiyor. λ = 0,3 ile adımlar kısa; başarısızlıkların çoğu basitçe iterasyon bütçesini bitiriyor (127'nin 77'si limitte bitmiyor). Burada en iyi nokta λ = 0,1; başarısız çözümleri rastgele seed'lerden yeniden başlatmak başarıyı %96,8'e çıkarıyor.

**Adım 2 · birden fazla çözüm.** Hedef (0,20, 0,05, 0,05) m, parmaklar (site x ekseni) aşağı bakıyor, wrist_roll 0'da sabit; yani dört joint dört bağımsız kısıtla karşı karşıya. 100 rastgele seed, DLS λ = 0,01:

| Joint limitleri | yakınsayan | farklı çözüm | gerçek limitlerin içinde |
|---|---|---|---|
| yok sayıldı | 96 / 100 | 4 | 1 |
| uygulandı | 92 / 100 | 1 | 1 |

Limitler olmadan çözücü ders kitabındaki dört dalı buluyor: ön/arka (pan −17,6° ya da 163,1°) × elbow up/down. Yalnızca biri, (−17,6°, −17,2°, 29,3°, 77,9°), SO-101'in aralıkları içinde. Limitleri yok sayan bir çözücü, yakınsayan 96 seed'inin 37 + 20 + 13'ünde erişilemez bir dala düşüyor.

**Adım 3 · gerilirken manipulability.** Bükülü bir pozdan (lift 1,3 rad, elbow 0), $\sigma_3$'ü minimize ederek bulunan düz poza (elbow −73,74°, wrist 2,01°); $s$ aradeğerliyor. Konum Jacobian'ı, 3×5:

| s | σ₁ | σ₂ | σ₃ | w | cond | erişim (m) |
|---|---|---|---|---|---|---|
| 0,00 | 0,4764 | 0,2419 | 7,71e-02 | 8,879e-03 | 6,2 | 0,3477 |
| 0,50 | 0,5155 | 0,3885 | 4,42e-02 | 8,857e-03 | 12 | 0,3945 |
| 0,90 | 0,5295 | 0,4397 | 9,20e-03 | 2,143e-03 | 58 | 0,4104 |
| 0,99 | 0,5301 | 0,4412 | 9,35e-04 | 2,187e-04 | 570 | 0,4110 |
| 1,00 | 0,5301 | 0,4411 | 1,05e-04 | 2,444e-05 | 5100 | 0,4110 |

İki singular value yerinde duruyor; üçüncüsü, radyal yön, çöküyor. s = 0,90 ile 1,00 arasında erişim bir milimetreden az değişirken $w$ 2,143e-03'ten 2,444e-05'e düşüyor: workspace sınırına yakın kol dışarı doğru neredeyse hiç hareket edemiyor.

## Boz

Her pozda zayıf yön $u_3$ boyunca 5 cm/s'lik bir alet hızı iste ve joint hızını hesapla:

| s | ‖q̇‖ pinv (rad/s) | ‖q̇‖ DLS λ=0,03 | DLS gerçekleşen / istenen | ‖q̇‖ DLS λ=0,1 | DLS gerçekleşen / istenen |
|---|---|---|---|---|---|
| 0,500 | 1,13 | 0,774 | 0,685 | 0,185 | 0,164 |
| 0,900 | 5,43 | 0,467 | 0,086 | 0,046 | 0,008 |
| 0,990 | 53,47 | 0,052 | 0,001 | 0,005 | 0,000 |
| 0,999 | 345,39 | 0,008 | 0,000 | 0,001 | 0,000 |
| 1,000 | 478,33 | 0,006 | 0,000 | 0,001 | 0,000 |

Pseudo-inverse, gergin pozda 5 cm/s'lik bir istek için 478,33 rad/s istiyor. DLS joint hızlarını sınırlı tutuyor ve bedelini takiple ödüyor: kolun hizmet edemediği yönde basitçe hareket etmiyor. λ ile ayarladığın denge budur: singularity'lerden uzakta doğruluk, yakınlarında sınırlı komutlar.

## Gerçek robotta

- Ham pseudo-inverse hızlarını asla servo'lara gönderme. DLS kullan (ya da $\Delta q$'yu kırp) ve joint limitlerini yalnızca firmware'de değil çözücüde de uygula: firmware limiti joint'i durdurur ama çözücü joint'in hareket ettiğine inanmaya devam eder.
- Modelin joint aralıkları Menagerie dosyasından gelir (yorumuna göre elbow aralığı 5 derecelik bir kalibrasyon ofseti taşıyor, [so101.xml](https://github.com/google-deepmind/mujoco_menagerie/blob/f3b6013/robotstudio_so101/so101.xml)). Fiziksel kolunun kullanılabilir aralıkları LeRobot kalibrasyonuna bağlıdır ([LeRobot SO-101](https://huggingface.co/docs/lerobot/so101)); IK'ya güvenmeden önce onları geri oku.
- Yalnızca simülasyon sonuçları. Gerçek SO-101'de IK doğruluğu (komut edilen bir IK çözümünde ucun gerçekte nereye düştüğü) gerçek kolda henüz ölçülmedi. Prosedür: bir ızgara üzerinde 20 hedef için IK çöz, joint'leri komut et, ucu ölç, hata dağılımını raporla.

## Kendini dene

1. **Kolay.** Adım 1'de λ = 0,001 olan DLS neden neredeyse tam olarak pseudo-inverse gibi davranıyor?
2. **Orta.** `ik_solve`'u `target_axis` ve `joints=ARM[:4]` ile (Adım 2'deki gibi wrist_roll sabit) kullanarak (0,20, 0,0, 0,15) m'de "parmaklar yatay, +x yönünde" iste. 1e-5'e yakınsar mı? Ders 08'deki kol düzlemiyle açıkla.
3. **Zor.** $w < w_0$ için $\lambda(w) = \lambda_0 (1 - w/w_0)^2$, değilse 0 olan uyarlamalı bir damping uygula (Buss, λ'yı konfigürasyondan seçen birkaç yayımlanmış yöntem listeliyor) ve Adım 1 sweep'ini yeniden koş. Başarıda ve iterasyonda sabit λ = 0,1'i geçer mi?

### Cevaplar

1. λ² = 1e-6 m², singularity'ye çok yakın olmadıkça $\sigma_i^2$ yanında ihmal edilebilir; dolayısıyla $\sigma_i/(\sigma_i^2+\lambda^2) \approx 1/\sigma_i$.
2. Yakınsamasını bekleme. Pitch joint'leri parmakları, yönünü yalnızca pan'ın değiştirebildiği tek bir düşey düzleme paralel tutar; parmakların +x boyunca olması pan'ı sabitler ve wrist_roll 0'da tutulunca site y = 0'ın biraz dışında kalır, dolayısıyla konum ve eksen aynı anda tam sağlanamaz. SO-101'in aralıkları içinde ıskalama çok daha büyüktür, çünkü shoulder_lift ve wrist_flex limitlerinde biter. Çalıştır ve kalan hatayı oku (bu dersin script'inde hesaplanmadı).
3. Burada hesaplanmadı.

## Akılda kalanlar ve kaynaklar

- Newton adımı $J^{+}e$ hızlı yakınsıyor (medyan 6 iterasyon) ama $q_0=0$'dan hedeflerin yalnızca %73,2'sinde başarılı oldu; DLS λ = 0,1 %88,6'da, yeniden başlatmalarla %96,8'de başarılı oldu.
- Joint limitlerini uygulamak cevabı 4 çözümden SO-101'in gerçekten ulaşabildiği 1 çözüme indirdi.
- Kol düzleşirken manipulability $w$ 8,879e-03'ten 2,444e-05'e düştü; pseudo-inverse o noktada 5 cm/s için 478,33 rad/s istedi, DLS sınırlı kaldı.

Sonraki: [ders 11 · Rigid-body dynamics](../part-4-dynamics-and-control/11-rigid-body-dynamics.md).

**Kaynaklar**

- K. M. Lynch, F. C. Park, *Modern Robotics*, Böl. 5 "Velocity Kinematics and Statics" (manipulability) ve Böl. 6 "Inverse Kinematics". [Kitap sayfası](https://hades.mech.northwestern.edu/index.php/Modern_Robotics).
- S. R. Buss, "Introduction to Inverse Kinematics with Jacobian Transpose, Pseudoinverse and Damped Least Squares methods", yayımlanmamış rapor, UC San Diego, 2009. [PDF](https://mathweb.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf).
- T. Yoshikawa, "Manipulability of Robotic Mechanisms", *The International Journal of Robotics Research* 4(2):3–9, 1985. [doi:10.1177/027836498500400201](https://doi.org/10.1177/027836498500400201).
- MuJoCo [API functions](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html) (`mj_jacSite`).
