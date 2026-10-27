# 29 · Distribution shift ve DAgger

<!-- meta -->
| | |
|---|---|
| **Bölüm** | [7 · Learning from demonstration](README.md) |
| **Çalışma türü** | CPU · Mac ya da Colab CPU |
| **Önkoşul** | [28](28-behavior-cloning.md) |
| **Önce öğren** | [Bias-Variance Tradeoff](https://aiengineeringfromscratch.com/lesson?path=phases/02-ml-fundamentals/10-bias-variance) |
| **Süre** | ~45 dk |
| **Kod** | [`experiments/29-dagger/experiment.py`](../../experiments/29-dagger/experiment.py) |
| **Durum** | ✅ tam |
<!-- /meta -->

> Behavior cloning expert'in state'lerinde eğitilir ve kendi state'lerinde çalışır. DAgger bu boşluğu, öğrenenin gerçekten ulaştığı state'lerde ne yapılacağını expert'e sorarak kapatır.

## Problem

Ders 28, 50 demonstration'la eğitilen ve girdi gürültüsü olmadan eğitildiğinde 0/20 yapan bir network'le bitti. Regression'da bir sorun yok; başarısızlıklar closed loop'ta oluyor: policy'nin kendi çıktısı robotu hareket ettiriyor, robotun yeni state'i de policy'nin bir sonraki girdisi oluyor. Ross ve Bagnell (2010) sebebe bir ad verdi: imitation'ı i.i.d. supervised learning gibi (her örnek tek bir sabit dağılımdan bağımsız çekilmiş gibi) ele almak, öğrenenin kendi action'larının bir sonra hangi state'leri göreceğine karar verdiğini yok sayar ve ortaya çıkan regret görev süresinde karesel büyür. Eğitimde görülen state'lerle çalışırken karşılaşılan state'ler arasındaki bu farka **distribution shift** denir. Bu ders onu SO-101 pick'inde ölçüyor ve standart çözümü, DAgger'ı (Ross, Gordon, Bagnell 2011), expert olarak ders 16'nın scripted pick'iyle uyguluyor.

## Ana fikir

### Sade anlatımla

Araba kullanmayı, şeridin hep tam ortasında kalan bir eğitmeni izleyerek öğreniyorsun. Yarım metre soldayken ne yapılacağını hiç görmedin, çünkü eğitmen oraya hiç gitmedi. İlk tek başına sürüşünde küçük bir hata seni tam oraya götürüyor; bir sonraki kararın bir tahmin, tahmin biraz yanlış ve araba hiç görmediğin yerlere doğru daha da kayıyor. Hatalar birikir: her hata bir sonrakini daha olası kılar. Distribution shift budur.

İyi bir eğitmen bunu başka türlü çözer: arabayı sana verir, yanına oturur ve her an, arabanın *gerçekten* bulunduğu yerden *kendisinin* ne yapacağını söyler. Bu düzeltmeleri not alırsın, bildiğin her şeyin üstüne eklersin, hepsiyle çalışırsın ve yeniden sürersin. Birkaç turdan sonra notların yalnız şeridin ortasını değil, *senin* kaydığın yerleri de kapsar. DAgger (Dataset Aggregation) budur.

İki şeye ihtiyaç duyar: yalnızca kendi sürüşünü tekrar oynatan değil, her state için "burada sen ne yapardın?" sorusunu cevaplayabilen bir expert ve önceki bütün notları saklamak. İkincisi olmazsa ne olduğunu Boz bölümü gösteriyor.

```mermaid
flowchart TD
    D["expert demonstration'ları"] --> P["dataset<br/>(bütün turlar saklanır)"]
    P --> T["policy'yi eğit"]
    T --> R["policy 10 episode<br/>boyunca sürer"]
    R --> S["gerçekten ziyaret ettiği<br/>state'leri kaydet"]
    S --> L["expert her state'i etiketler:<br/>burada ben ne yapardım?"]
    L --> P
    T --> E["20 test konumunda<br/>evaluate et"]
```

### DAgger kod olarak

```python
def dagger(expert, demos, n_iters, m):
    data = list(demos)                       # (state, action) pairs recorded while the expert drove
    policy = train(data)                     # iteration 0: this is plain behavior cloning
    policies = [policy]
    for i in range(n_iters):
        for _ in range(m):                   # m episodes per iteration (here 10)
            for s in rollout(policy):        # the LEARNER drives, so it picks the states...
                data.append((s, expert(s)))  # ...and the EXPERT picks the action label
        policy = train(data)                 # retrain on everything, not only the newest round
        policies.append(policy)
    return best_on_validation(policies)      # success is not monotonic in the iteration

def drift_deg(q, demo_frames):               # q: 5 joints + jaw of one visited state, in degrees
    return min(max(abs(q - f)) for f in demo_frames)   # 0 = a state the dataset contains
```

| Ne | Formül |
|---|---|
| BC maliyet sınırı | $J(\hat\pi_{\text{BC}}) \le J(\pi^*) + T^2 \epsilon$ |
| DAgger maliyet sınırı | $J(\hat\pi_{\text{DAgger}}) \le J(\pi^*) + u\,T\,\epsilon_N + O(1)$ |
| Kayma ölçüsü | $\min_j \max_i \lvert q_i - q_i^{(j)}\rvert$ |

$J$, $T$ adımlık bir episode boyunca beklenen maliyet (bizde $T = 150$ policy adımı); $\pi^*$ expert, $\hat\pi$ öğrenilmiş bir policy; $\epsilon$, BC'nin expert'in state'lerinde hata yapma olasılığı; $\epsilon_N$, öğrenilen policy'lerin ziyaret ettiği state'lerde geriye dönük en iyi policy'nin loss'u; $u$, tek bir hatanın (ardından expert devam ederken) kalan maliyete ne kadar ekleyebileceğinin sınırı; $q_i$ ziyaret edilen bir state'in $i$. joint'i, $q^{(j)}$ ise $j$. demonstration frame'i. Kısacası: BC'nin ek maliyeti episode uzunluğunun karesiyle büyüyebilir, DAgger'ınki yalnızca doğrusal.

<details>
<summary>Matematik ayrıntısı: BC'nin hatası neden T² ile, DAgger'ınki neden T ile büyür</summary>

**İki state dağılımı.** BC loss'u $d_{\pi^*}$ altında, yani expert'in ziyaret ettiği state'lerde minimize eder. Çalışma anında policy $d_{\pi_\theta}$ ile, kendi ziyaret ettiği state'lerle karşılaşır. 0-1 loss ve adım başına [0, 1] içinde bir görev maliyeti varsay; policy expert state'lerinde $\epsilon$ olasılıkla hata yapsın.

**$T^2$ sezgisi.** Tek bir hata öğreneni hiçbir şeyin eğitilmediği state'lere taşıyabilir; en kötü durumda kalan her adımda en yüksek maliyeti öder. Hata $T$ adımın herhangi birinde başlayabilir (her birinde yaklaşık $\epsilon$ olasılıkla) ve $T$'ye kadar maliyet çıkarabilir; ek maliyet bu yüzden $T \cdot \epsilon T = T^2\epsilon$ mertebesindedir. DAgger makalesi bu BC sınırını ve sıkılığını veriyor (Ross & Bagnell 2010 ona ulaşan bir örnek veriyor).

**DAgger sınırı.** Algoritma 3.1, $i$. iterasyonda $\pi_i = \beta_i \pi^* + (1-\beta_i)\hat\pi_i$ karışımını koşar, ziyaret edilen state'leri $\pi^*(s)$ ile etiketler, $\mathcal D \leftarrow \mathcal D \cup \{(s, \pi^*(s))\}$ diye biriktirir ve $\hat\pi_{i+1}$'i $\mathcal D$'nin tamamında eğitir. Bütün turların birleşiminde eğitmek, no-regret bir online öğrenici olan Follow-The-Leader'dır; $T^2$'yi $u\,T\,\epsilon_N$'ye çeviren budur. Sınır, $N$ $uT$ mertebesine ulaşınca $\hat\pi_{1:N}$ içindeki bir policy için geçerli (Teorem 2.1 ve 3.2); algoritmanın "validation'da en iyi policy'yi döndür" ile bitmesinin sebebi bu.

**Bizim programımız.** $\beta_i = \mathbb 1(i = 1)$: ilk veri expert'ten gelir, sonrasında öğrenen tek başına sürer. Makale bunu basit, parametresiz versiyon olarak adlandırıyor ve pratikte çoğu zaman en iyi sonucu verdiğini bildiriyor.

Daha derine: [Bias-Variance Tradeoff](https://aiengineeringfromscratch.com/lesson?path=phases/02-ml-fundamentals/10-bias-variance) (AIEFS); [Ross, Gordon, Bagnell 2011](https://arxiv.org/abs/1011.0686), Algoritma 3.1 ve Teorem 2.1, 3.2; [Ross & Bagnell 2010](https://proceedings.mlr.press/v9/ross10a.html); [CS 285](https://rail.eecs.berkeley.edu/deeprlcourse/), ders 2–3.

</details>

### Etiketleyici olarak expert

DAgger, expert'in hiç ziyaret etmediği state'lerde $\pi^*(s)$'e ihtiyaç duyar. Bizim expert'imiz ders 16'nın zaman indeksli state machine'i; bu yüzden policy adımı $k$'deki bir öğrenen state'inin etiketi, scripted pick'in küpün şu anki konumu için $k$ adımında komutladığı action. Öğrenen küpü 1 cm'den fazla ittiyse expert yeni konum için yeniden planlıyor (planlar küp konumlarının 2 mm'lik bir ızgarasında önbelleğe alınıyor). Bu etiketleyici zamanında doğru, öğrenen geç ya da erken kaldığında daha zayıf; bir insan operatörün de farklı biçimde aynı sınırı var.

## Deney

Kurulum: ders 28'in feature'ları (state + saat) ve MLP'si, girdi gürültüsü **olmadan** eğitilmiş ("düz BC"), 20 demonstration, aynı 20 test konumu, 3 seed. DAgger her iterasyonda yeni küp konumlarında 10 öğrenen episode'u ekliyor, 4 iterasyon. Karşılaştırma, aynı sayıda expert etiketli episode'la, scripted demonstration'lardan eğitilen düz BC. Tam script: [`experiments/29-dagger/experiment.py`](../../experiments/29-dagger/experiment.py), çıktı: [`output.txt`](../../experiments/29-dagger/output.txt). MuJoCo 3.17.0, NumPy 2.5.4, PyTorch 2.14.1 (CPU), Apple Silicon Mac (Apple M5), deterministik, yaklaşık 4,5 dk CPU (yaklaşık; çok yüklü bir makinede duvar saati süresi birkaç kat uzundu).

```python
for it in range(1, ITERS + 1):
    for xy in L26.sample_positions(ROLLOUTS, seed=1000 * seed + it):
        _, x, y, _ = run_episode(scene, labeler, pol, xy, label=True)   # learner drives
        nx += x; ny += y                                                # expert labels
    X, Y = np.concatenate([X, nx]), np.concatenate([Y, ny])             # aggregate
    pol = L28.train_mlp(X, Y, seed=seed)                                # retrain on all
```

*Tahmin et:* düz BC 20 demonstration'da başarısız. 40 ya da 60 demonstration'la toparlanır mı? Aynı sayıda expert etiketli episode'la DAgger toparlanır mı?

**B. DAgger ve düz BC.** "Birleşik" üç seed'in toplamı; "uzaklık", ziyaret edilen kol state'lerinin 20 başlangıç demonstration'ının en yakın frame'ine ortalama uzaklığı (40 ve 60 ile eğitilen BC satırlarında da).

| İterasyon | expert etiketli episode | DAgger, seed 0/1/2 | birleşik | Wilson %95 | uzaklık [derece] | düz BC, seed 0/1/2 | birleşik | uzaklık [derece] |
|---|---|---|---|---|---|---|---|---|
| 0 | 20 | 0/0/0 | 0/60 | [0,00, 0,06] | 29,35 | 0/0/0 | 0/60 | 29,35 |
| 1 | 30 | 3/18/5 | 26/60 | [0,32, 0,56] | 15,79 | – | – | – |
| 2 | 40 | 19/11/12 | 42/60 | [0,57, 0,80] | 12,59 | 0/0/0 | 0/60 | 26,75 |
| 3 | 50 | 19/16/15 | 50/60 | [0,72, 0,91] | 6,09 | – | – | – |
| 4 | 60 | 15/15/15 | 45/60 | [0,63, 0,84] | 7,50 | 0/0/0 | 0/60 | 24,76 |

![Expert etiketli episode sayısına (20'den 60'a) göre iki panel. Solda: DAgger'ın birleşik başarısı 0/60'tan 50 episode'da 50/60'a çıkıyor, 60 episode'da 45/60'a hafifçe düşüyor; düz BC 0/60'ta kalıyor. Sağda: DAgger'ın demonstration'lara ortalama uzaklığı 29 dereceden yaklaşık 6 ila 8 dereceye iniyor, düz BC'ninki yalnızca yaklaşık 25 dereceye.](../../experiments/29-dagger/dagger_iterations.png)

(Düz BC, CPU bütçesi içinde kalmak için yalnız 40 ve 60 episode'da yeniden eğitildi.) Demonstration'ları üç katına çıkarmak düz BC'ye hiçbir şey kazandırmıyor: 20, 40 ve 60 episode'da 0/60 ve state'leri başlangıç demonstration'larından yaklaşık 25–29° uzakta kalıyor. Aynı expert bütçesini öğrenenin kendi state'lerine harcayan DAgger üç iterasyon sonra başarıyı 50/60'a çıkarıyor ve uzaklık yaklaşık 6°'ye iniyor. İterasyon 4, iterasyon 3'ten düşük (45/60 ve 50/60, örtüşen aralıklar): DAgger monoton iyileşmiyor; "validation'da en iyi policy'yi döndür" adımı bu yüzden var. Etiketler bedava değil: scripted expert 1327 kez koşturuldu (etiketleme ve drift referansı). Bunların en fazla 180'i yeni başlangıç konumları için (20 test + 160 toplama episode'u); kalanı, öğrenen küpü 1 cm'den fazla ittikten sonraki yeniden planlamalar.

**A. Öğrenen nereye gidiyor.** Zaman penceresi başına en yakın demonstration frame'ine ortalama uzaklık [derece], seed 0:

| Policy | 0–1 s | 1–2 s | 2–3 s | 3–4,5 s | 4,5–6 s | başarı | expert trajectory'sinden ortalama maksimum sapma [derece] |
|---|---|---|---|---|---|---|---|
| düz BC, 20 demo | 22,44 | 29,05 | 28,96 | 28,94 | 28,87 | 0/20 | 62,6 |
| BC + girdi gürültüsü (ders 28), 20 demo | 1,80 | 4,52 | 4,94 | 4,46 | 3,85 | 20/20 | 10,7 |
| 4 iterasyon sonra DAgger | 2,27 | 4,97 | 7,38 | 15,10 | 15,95 | 15/20 | 19,0 |

Düz BC daha ilk saniyede gösterilmiş her kol konfigürasyonundan 22° uzakta, sonrasında yaklaşık 29°: birikme ilk 25 policy adımında oluyor, ondan sonra policy verisi olmayan bir bölgede çalışıyor. İki çözüm de yaklaşma sırasında veriye yakın kalıyor (ilk iki saniyede 5°'nin altında). DAgger'ın state'leri daha sonra kayıyor (3 s'den sonra 15°), ama dataset'i artık bu state'leri expert etiketleriyle içeriyor; yani orada *demonstration'lardan* uzak olmak *training verisinden* uzak olmakla aynı şey değil. Burada üçünün en iyisi ders 28'in gürültüyle eğitilmiş BC policy'si: ek expert sorgusu olmadan 20/20. Girdi gürültüsü, DAgger'ın on-policy verisinin (öğrenenin kendi ziyaret ettiği state'lerden gelen verinin) ucuz ve yerel bir ikamesi ve öğrenenin hataları küçük kaldığında yetiyor.

## Boz

DAgger'daki "Agg" aggregation'dır. Her iterasyonu yalnız en yeni 10 öğrenen episode'unda eğit (demonstration'ları ve önceki turları atarak), seed 0:

| İterasyon | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| aggregation olmadan başarı | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| aggregation ile başarı | 0/20 | 3/20 | 19/20 | 19/20 | 15/20 |

Aggregation olmadan her iterasyon başarısız. Bizim yorumumuz: her yeni dataset yalnız önceki, başarısız policy'nin ziyaret ettiği state'leri, çoğunlukla hatalarından sonraki state'leri içeriyor. Yalnız bunlarla eğitilen bir policy yaklaşmanın doğru yapıldığını hiç görmemiş oluyor, bu yüzden yeni bir şekilde başarısız oluyor ve sonraki dataset yalnız o başarısızlığı kapsıyor: yalnız son dersin düzeltmelerini saklayan öğrenci şeritte sürmeyi unutur. Makalenin analizi bütün turların birleşiminde eğitmeye (no-regret bir öğrenici olan Follow-The-Leader'a) dayanıyor; yalnız en yeni tur öyle değil. Demonstration'ları saklamak policy'yi expert'in state'lerine bağlıyor; öğrenenin state'leri ona kendi hatalarından toparlanmayı öğretiyor.

## Gerçek robotta

- **LeRobot DAgger tarzı veri toplamayı destekliyor.** `lerobot-rollout --strategy.type=dagger`, bir insan izlerken eğitilmiş bir policy'yi robotta çalıştırır. Operatör onu durdurabilir, leader kolla (teleoperation için elle yönlendirilen ikiz kol) kontrolü alabilir, bir toparlanma ve bir düzeltme teleoperate edebilir ve kontrolü geri verebilir; hepsi tek bir episode içinde. Hem otonom hem insan segmentleri kaydedilir ve policy sonra demonstration'lar artı yeni veri üzerinde fine-tune edilir ([Human-In-the-Loop Data Collection](https://huggingface.co/docs/lerobot/hil_data_collection)). Dokümantasyon uyumlu teleoperator'lar arasında `so_leader`'ı sayıyor, çünkü leader duraklatıldığında follower'ı yansıtmak için kendini hareket ettirebilmeli; arkasındaki fikirler olarak DAgger, HG-DAgger ve RaC'yi adlandırıyor.
- **İnsan etiketleyiciyle ne değişir.** Human-in-the-loop (HIL) toplamada insan yalnızca başarısızlık yakın göründüğünde müdahale eder (HG-DAgger kapılaması); yani etiketler bizim döngümüzdeki gibi ziyaret edilen her state'i değil, öğrenenin başarısız olmak üzere olduğu state'leri kapsar. Etiketler ayrıca bir script'inkinden daha gürültülü ve daha yavaştır.
- Gerçek kolda henüz ölçülmedi. Prosedür: 20 teleoperate edilmiş episode üzerinde bir policy eğit, onu 20 işaretli konumda evaluate et (ders 25), her biri 10 episode'luk 2–3 tur `--strategy.type=dagger` koş, fine-tune et ve aynı konumlarda yeniden evaluate et; her turdan sonra k/20'yi ve Wilson aralıklarını, aynı sayıda düz teleoperate edilmiş episode'la eğitilmiş bir policy'nin yanında raporla.

## Kendini dene

1. **Kolay.** Yukarıdaki DAgger döngüsünde dataset'e giren state'leri kim, action'ları kim seçiyor?
2. **Orta.** Ders 28'in girdi gürültüsü de düz BC'yi 20 demonstration'da 20/20'ye çıkardı. Neden çalıştığını ve neden DAgger'ın genel bir ikamesi olmadığını açıkla.
3. **Zor.** Evaluation'a episode ortasında bir bozulma ekle (küpü t = 1,6 s'de 3 cm kaydır) ve DAgger döngüsünü toplama sırasında aynı bozulmayla tekrarla. Düz BC ile farkı tahmin et, sonra ölç.

### Cevaplar

1. State'leri öğrenen seçiyor (sürüyor); action'ları (etiketleri) expert seçiyor.
2. Gürültü, policy'yi demonstration'lara yakın state'lerde, yakındaki demonstration state'inden gelen expert action'ıyla eğitir; bu ancak doğru action o komşulukta az değişiyorsa iyi bir etikettir. DAgger öğrenenin gerçekten ulaştığı state'leri, ne kadar uzak olursa olsun, o state için expert'in action'ıyla etiketler.
3. Burada hesaplanmadı; çalıştır.

## Akılda kalanlar ve kaynaklar

- Simülasyonda düz BC (girdi gürültüsü yok) 20, 40 ve 60 demonstration'da 0/60 yaptı; aynı expert bütçesiyle DAgger üç iterasyon sonra 50/60'a ulaştı.
- Kayma ölçülebilir: düz BC'nin kol state'leri gösterilmiş her konfigürasyondan 22–29° uzaktaydı, DAgger'ınkiler üç–dört iterasyon sonra (ortalama) 6–8°.
- Aggregate et: yalnız en yeni öğrenen verisiyle yeniden eğitmek her iterasyonda 0/20 yaptı. Bu görev için ders 28'in girdi gürültüsü (ek expert sorgusu olmadan 20/20) daha ucuz çözümdü; DAgger genel çözüm ve LeRobot'un `--strategy.type=dagger`'ı onun human-in-the-loop biçimi.

Sonraki: [ders 30 · Multimodal action'lar: diffusion ve flow policy'ler](30-diffusion-and-flow-policies.md).

Kaynaklar:
- S. Ross, G. J. Gordon, J. A. Bagnell, ["A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning"](https://arxiv.org/abs/1011.0686), AISTATS 2011 (Algoritma 3.1, Teorem 2.1 ve 3.2).
- S. Ross, J. A. Bagnell, ["Efficient Reductions for Imitation Learning"](https://proceedings.mlr.press/v9/ross10a.html), AISTATS 2010.
- LeRobot, [Human-In-the-Loop Data Collection](https://huggingface.co/docs/lerobot/hil_data_collection) ve [Imitation learning on real-world robots](https://huggingface.co/docs/lerobot/il_robots) (okundu 2026-10-27).
- UC Berkeley [CS 285 Deep Reinforcement Learning](https://rail.eecs.berkeley.edu/deeprlcourse/), ders 2–3 "Behavioral Cloning".
- Ölçülen tüm sayılar: [`experiments/29-dagger/output.txt`](../../experiments/29-dagger/output.txt).
