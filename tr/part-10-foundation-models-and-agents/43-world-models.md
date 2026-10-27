# 43 · World model'ler: öğrenilmiş dynamics

<!-- meta -->
| | |
|---|---|
| **Bölüm** | [10 · Foundation model'ler ve agent'lar](README.md) |
| **Çalışma türü** | CPU · Mac ya da Colab CPU |
| **Önkoşul** | [28](../part-7-learning-from-demonstration/28-behavior-cloning.md) |
| **Önce öğren** | [World Models & Video Diffusion](https://aiengineeringfromscratch.com/lesson?path=phases/04-computer-vision/28-world-models-video-diffusion) · [Multi-Layer Networks and Forward Pass](https://aiengineeringfromscratch.com/lesson?path=phases/03-deep-learning-core/02-multi-layer-networks) |
| **Süre** | ~60 dk |
| **Kod** | [`experiments/43-world-models/experiment.py`](../../experiments/43-world-models/experiment.py) |
| **Durum** | ✅ tam |
<!-- /meta -->

> Öğrenilmiş bir $s' = f(s, a)$ modeli, dünyayı sürekli yeniden ölçen bir planner onu kullanabiliyorsa yeterince iyidir; tek adımlık doğruluk yanlış ölçüttür.

## Problem

Model tabanlı kontrol ([ders 24](../part-6-planning-and-optimal-control/24-lqr-and-mpc.md)) dynamics'in elinde olduğunu varsayar. Yük taşıyan, kablo sürtünmesi ya da aşınmış dişlileri olan gerçek bir kolda çoğu zaman elinde değildir, ama $(s_t, a_t, s_{t+1})$ geçişlerini loglayabilirsin. Nagabandi et al. rastgele action'lardan gelen böyle bir veriyle bir sinir ağı eğitti ve simüle robotları onun içinde MPC ile kontrol etti ([arXiv 1708.02596](https://arxiv.org/abs/1708.02596)); world-model agent'ları fikri görüntülere ve latent state'lere taşıyor (Ha & Schmidhuber, [arXiv 1803.10122](https://arxiv.org/abs/1803.10122); Dreamer, [arXiv 1912.01603](https://arxiv.org/abs/1912.01603)). Senin sisteminde işe yarayıp yaramayacağına iki soru karar verir: model kendi çıktılarıyla rollout edildiğinde tahmin hatası ne hızla büyüyor, ve öğrenilmiş modelde planlamak gerçek modelde planlamakla aynı sonuca ulaşıyor mu?

## Ana fikir

**Öğrenilmiş dynamics.** $\hat f_\theta$'yı bir geçiş dataset'i $\mathcal D$ üzerinde fit et. Değişimi tahmin etmek, state'i tahmin etmekten kolaydır (Nagabandi et al., §IV-A):

$$\hat s_{t+1} = s_t + \hat f_\theta(s_t, a_t), \qquad \min_\theta \sum_{\mathcal D} \big\| (s_{t+1} - s_t) - \hat f_\theta(s_t, a_t) \big\|^2$$

girişler ve hedefler normalize edilmiş olarak (bkz. AIEFS: [Multi-Layer Networks](https://aiengineeringfromscratch.com/lesson?path=phases/03-deep-learning-core/02-multi-layer-networks)).

**Biriken hata.** Training tek adımlık hatayı minimize eder. Planlama ise $H$ adımlık açık döngü tahminler kullanır: $\hat s_{t+h} = \hat s_{t+h-1} + \hat f_\theta(\hat s_{t+h-1}, a_{t+h-1})$; bunlar modele kendi hatalarını besler. Aynı makale modelleri bu yüzden bu $H$ adımlık hatayla doğruluyor.

**Random-shooting MPC.** Uzunluğu $H$ olan $K$ action dizisi örnekle, her birini modelle tahmin et, bir cost ile puanla, en iyisinin yalnızca ilk action'ını uygula, yeni state'i ölç, tekrarla (Nagabandi et al., §IV-C). PETS, model belirsizliğini hesaba katmak için tek ağın yerine olasılıksal bir ensemble koyuyor ([arXiv 1805.12114](https://arxiv.org/abs/1805.12114)); Dreamer davranışı hayali latent trajectory'ler üzerinden backpropagation ile öğreniyor ([arXiv 1912.01603](https://arxiv.org/abs/1912.01603)). Bkz. AIEFS: [World Models & Video Diffusion](https://aiengineeringfromscratch.com/lesson?path=phases/04-computer-vision/28-world-models-video-diffusion).

| terim | burada |
|---|---|
| state $s$ | $(\theta, \dot\theta)$, $\theta = 0$ aşağı sarkık |
| action $a$ | motor torku $u$, $\lvert u\rvert \le 2$ N·m |
| model girişi | $(\cos\theta, \sin\theta, \dot\theta, u)$ (θ'da periyodik) |
| cost | $\sum_h \mathrm{wrap}(\theta_h - \pi)^2 + 0{,}1\,\dot\theta_h^2 + 0{,}001\,u_h^2$ |

## Deney

**Sistem.** Bir MuJoCo sarkacı: 0,5 m'de 1 kg'lık noktasal kütle, hinge damping 0,05 N·m·s/rad, tork limiti 2 N·m; bu, yataydaki 4,91 N·m'lik yerçekimi torkundan küçük, dolayısıyla swing-up pompalama gerektiriyor. Kontrol adımı 0,05 s (5 × 10 ms, semi-implicit Euler). **Veri:** rastgele başlangıçlardan ve rastgele torklardan 400 trajectory × 50 adım = 20.000 geçiş. **Model:** MLP 4-128-128-2 (17.410 parametre), batch 512 ile 1200 Adam adımı, train MSE 2,47e-06.

```python
def forward(self, s, u):                                   # predict the change, not the state
    x = (features(s, u) - self.x_mu) / self.x_sd
    return s + self.net(x) * self.d_sd + self.d_mu

U = np.repeat(rng.uniform(-U_MAX, U_MAX, (K, H // 5)), 5, axis=1)   # K piecewise-constant sequences
u = U[np.argmin(planner(s, U)), 0]                         # best first action
s = sim_step(s, u)                                         # always applied to the TRUE simulator
```

Script'in tamamı: [`experiments/43-world-models/experiment.py`](../../experiments/43-world-models/experiment.py). MuJoCo 3.17.0, NumPy 2.5.4, PyTorch 2.14.1 (CPU, 1 thread), Python 3.14, Apple M5. Deterministik: yeniden koşu her hata ve başarı sayısını aynen üretiyor; yalnızca süreler değişiyor. Ağır yüklü bir makinede toplam ~2 dk (yaklaşık).

**A. Rollout ufku → tahmin hatası.** 200 ayrılmış trajectory; model ve simülatöre aynı rastgele torklar veriliyor. *Tahmin et:* tek adımlık hata 0,0004 rad ise, 100 adım sonra ne olur?

| H adım | H [s] | ort. \|açı hatası\| [rad] | medyan [rad] | ort. \|hız hatası\| [rad/s] |
|---|---|---|---|---|
| 1 | 0,05 | 0,0004 | 0,0004 | 0,0019 |
| 5 | 0,25 | 0,0018 | 0,0012 | 0,0059 |
| 10 | 0,50 | 0,0030 | 0,0023 | 0,0074 |
| 20 | 1,00 | 0,0068 | 0,0031 | 0,0222 |
| 40 | 2,00 | 0,0222 | 0,0042 | 0,0585 |
| 60 | 3,00 | 0,0547 | 0,0047 | 0,1310 |
| 100 | 5,00 | 0,0966 | 0,0072 | 0,2400 |

Ortalama 100 adımda yaklaşık 240× büyüyor, 1 s'den sonra doğrusaldan hızlı; medyan yalnızca 18×. Aradaki fark, hatanın birkaç trajectory'de toplandığını söylüyor. Makul bir mekanizma (burada ölçülmedi): dik konuma yakın geçen trajectory'ler; orada yakın state'ler ayrışır ve küçük bir model hatası sarkacın hangi yöne düşeceğine karar verir.

**B. Planlama ufku, öğrenilmiş ve gerçek model.** Aşağı sarkık konumdan swing-up (3 başlangıç, K = 100 dizi, her 5 adımda yeni bir rastgele tork), 5 s'lik episode'lar; başarı = son saniyenin tamamında dik konuma 0,2 rad içinde olmak. Gerçek planner adayları MuJoCo'da (`mujoco.rollout`), öğrenilmiş olan MLP'de rollout ediyor; iki durumda da seçilen action gerçek simülatöre uygulanıyor.

| H | planner | başarı | ort. kuyruk hatası [rad] | ilk dik adım | plan süresi [s] |
|---|---|---|---|---|---|
| 10 | gerçek | 0/3 | 0,831 | – | 5,7 |
| 10 | öğrenilmiş | 0/3 | 0,853 | – | 1,2 |
| 20 | gerçek | 2/3 | 0,046 | 76 | 10,9 |
| 20 | öğrenilmiş | 2/3 | 0,059 | 77 | 2,6 |
| 30 | gerçek | 3/3 | 0,023 | 66 | 18,2 |
| 30 | öğrenilmiş | 3/3 | 0,021 | 67 | 4,0 |

Başarıya model değil planlama ufku karar veriyor: 0,5 s, pompalamanın karşılığını görmek için kısa; 1,5 s ikisi için de yetiyor. Öğrenilmiş model, dik konuma ulaşma zamanında gerçek modelle bir adım içinde eşleşiyor ve bu koşuda 4–5× daha hızlı planladı, çünkü 100 aday tek bir batch'li tensor işlemiyle değerlendiriliyor (plan süreleri ağır yüklü bir makinedeki duvar saati ve koşudan koşuya değişiyor; başarı ve hata sütunları değişmiyor).

## Boz

**Modeli aç bırak.** Yalnızca düşük enerjili veriyle eğitilmiş bir model (görülen |θ| ≤ 0,42 rad, torklar ≤ 0,5 N·m) tam aralıklı test setinde 0,0371 rad tek adım hatasına sahip, yaklaşık 90× kötü; 10 adım sonra 0,1160 rad. Onunla MPC yine de 3 başlangıcın 2'sinde swing-up yaptı (H = 20). Tam aralıklı dataset'i 2 trajectory'ye (100 geçiş) küçültmek 20 adımlık hatayı 0,0307 rad'a çıkardı ve H = 30 ile MPC yine 3/3 başardı. Feedback model hatasını gizliyor: planner her 0,05 s'de ölçülen state'ten yeniden başlıyor.

**Feedback'i kaldır.** Her planın ilk k action'ını yeniden planlamadan önce açık döngü yürüt (H = 30):

| k | yeniden planlama aralığı | gerçek model | kuyruk hatası [rad] | öğrenilmiş model | kuyruk hatası [rad] |
|---|---|---|---|---|---|
| 1 | 0,05 s | 3/3 | 0,023 | 3/3 | 0,021 |
| 5 | 0,25 s | 3/3 | 0,074 | 2/3 | 0,126 |
| 10 | 0,50 s | 1/3 | 0,173 | 1/3 | 0,173 |
| 20 | 1,00 s | 0/3 | 0,665 | 0/3 | 0,620 |
| 30 | 1,50 s | 0/3 | 1,216 | 0/3 | 1,216 |

k = 5'te öğrenilmiş model, gerçek modelin tuttuğu bir başlangıcı kaybediyor: açık döngü hatası artık önemli. k = 10'un ötesinde mükemmel model bile başarısız, çünkü 100 rastgele dizinin en iyisi kararsız bir dengeyi açık döngüde tutamıyor. Yeniden planlama sıklığı, kusurlu bir modeli (ve kusurlu bir optimizer'ı) kullanılabilir kılan düğmedir.

## Gerçek robotta

- **Gerçek kolda henüz ölçülmedi.** SO-101 için prosedür: rastgele ama limitlere uyan hareketler sırasında ([ders 15](../part-4-dynamics-and-control/15-safety.md)'in validation'ı) eklem konumlarını ve komutlanan hedefleri kontrol frekansında logla, $\hat f_\theta$'yı $(q, \dot q, q^{\text{goal}}) \to \Delta(q, \dot q)$ üzerinde eğit, herhangi bir kapalı döngü kullanımından önce ayrılmış loglarda H adımlık hata tablosunu raporla.
- Gerçek veri, bu sarkaçta olmayan sensör gürültüsü, latency ve backlash içerir; model bunları yalnızca veride varsa öğrenir. Sarkaçta rastgele torklar ucuzdur; bir kolda konum, hız ve torkta sınırlandırılmalıdır.
- Yeniden planlama sıklığını yüksek, ufku bir kontrol periyodunda hesaplanabilecek kadar kısa tut; donanımda B tablosundaki planlama süresi latency bütçesinin parçasıdır.

## Kendini dene

1. **Kolay.** A tablosuna göre ortalama açı hatası 1 s'den 5 s'lik açık döngü tahmine kaç kat büyüyor?
2. **Orta.** Ortalama H adımlık hata neden medyandan çok daha hızlı büyüyebilir? H = 100 satırını kullan.
3. **Zor.** Tek MLP'yi 5'li bir ensemble ile değiştir (farklı seed'ler) ve PETS'teki gibi ensemble dağılımını ek bir cost terimi olarak kullan. Düşük enerji durumunu yeniden koştur. Koşturmadan önce tahmin et: başarı 2/3'ün üstüne çıkar mı?

### Cevaplar

1. 0,0966 / 0,0068 ≈ 14.
2. H = 100'de ortalama 0,0966 rad, medyan 0,0072 rad: trajectory'lerin çoğu doğru kalıyor, birkaçı büyük açılarla ayrışıp ortalamayı domine ediyor (muhtemelen kararsız dik state'e yakın geçenler; burada ölçülmedi).
3. Burada hesaplanmadı; koştur.

## Akılda kalanlar ve kaynaklar

- 20.000 rastgele geçişle eğitilen 17.410 parametrelik bir MLP tek adımı 0,0004 rad hatayla tahmin ediyor; 100 adım rollout edildiğinde ortalama hata 240× büyüyor.
- Öğrenilmiş modelin içinde random-shooting MPC, sarkacı MuJoCo içindeki MPC kadar güvenilir biçimde kaldırdı (H = 30'da 3/3), bu makinede planlama süresinin bir kesriyle.
- Ölçülen state'lerden sık yeniden planlama büyük model hatalarını emiyor; 5 adımlık açık döngü yürütme onları şimdiden açığa çıkarıyor.

Sıradaki: Bitirme projesi · [ders 44 · Söyle, gör, planla, tut](../capstone/44-capstone.md).

Kaynaklar:
- Nagabandi, Kahn, Fearing, Levine, [Neural Network Dynamics for Model-Based Deep RL with Model-Free Fine-Tuning](https://arxiv.org/abs/1708.02596) (arXiv 1708.02596).
- Ha & Schmidhuber, [World Models](https://arxiv.org/abs/1803.10122) (arXiv 1803.10122); Hafner et al., [Dream to Control](https://arxiv.org/abs/1912.01603) (arXiv 1912.01603); Chua et al., [PETS](https://arxiv.org/abs/1805.12114) (arXiv 1805.12114).
- MuJoCo [Python bindings, `rollout` modülü](https://mujoco.readthedocs.io/en/stable/python.html).
- Ölçülen tüm sayılar: [`experiments/43-world-models/output.txt`](../../experiments/43-world-models/output.txt).
