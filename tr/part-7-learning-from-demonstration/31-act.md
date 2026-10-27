# 31 · Action chunking ve ACT

<!-- meta -->
| | |
|---|---|
| **Bölüm** | [7 · Learning from demonstration](README.md) |
| **Çalışma türü** | CPU · Mac ya da Colab CPU |
| **Önkoşul** | [28](28-behavior-cloning.md), [30](30-diffusion-and-flow-policies.md) |
| **Önce öğren** | [The Full Transformer — Encoder + Decoder](https://aiengineeringfromscratch.com/lesson?path=phases/07-transformers-deep-dive/05-full-transformer) · [Autoencoders & Variational Autoencoders (VAE)](https://aiengineeringfromscratch.com/lesson?path=phases/08-generative-ai/02-autoencoders-vae) |
| **Süre** | ~75 dk |
| **Kod** | [`experiments/31-act/experiment.py`](../../experiments/31-act/experiment.py) |
| **Durum** | ✅ tam |
<!-- /meta -->

> Bir sonraki action'ı değil, sonraki k action'ı tahmin et: chunk, observation'ın süresini ölçemediği bir beklemenin ötesine uzanabilir.

## Problem

Bir behavior cloning policy'si şimdiki observation'ı bir sonraki action'a eşler. Gösterimlerde observation'ın aynı kaldığı ama doğru action'ın zamanla değiştiği anlar vardır: kol, gripper kapanırken bekler, sonra yoluna devam eder. ACT makalesi ([Zhao vd. 2023](https://arxiv.org/abs/2304.13705), Böl. I), action dizilerini tahmin etmenin "temporally correlated confounders" ile, örneğin gösterimlerdeki duraklamalarla başa çıkmaya yardım ettiğini söyler. Chunk boyutu ablation'ında (Şek. 8a: 2 simülasyon görevi × scripted/insan verisi, ortalama, ensembling yok) başarı $k = 1$'de %1'den $k = 100$'de %44'e çıkıyor.

Bizim sürümümüz, simülasyonda: düzlemsel bir kol hedefe gider, 0,5 s hareketsiz bekler, sonra bir bırakma noktasına gider. 100 gösterimle eğitilmiş tek adımlık bir policy, **600 episode'un 543'ünde hedefte takılı kalıyor**.

## Ana fikir

**Action chunking.** $\pi(a_t \mid o_t)$ yerine $\pi(a_{t:t+k} \mid o_t)$ öğren. Bir chunk'ı uygulayıp her $k$ adımda bir yeniden sorgulamak görevin etkin ufkunu $k$ kat kısaltır (Böl. I): daha az karar, hataların birikmesi için daha az fırsat ([ders 29](29-dagger.md)).

**Bekleme tek adımlık policy'yi neden bozar.** Bekleme boyunca observation (eklemler durağan) neredeyse hiç değişmez; veride bir sonraki action, sonuncusu hariç her bekleme adımında "bekle"dir. ACT'nin L1 loss'unu boyut boyut medyan minimize eder: bekle, sonsuza kadar (L2 ortalamayı, yani yavaş bir sürünmeyi verirdi; [ders 30](30-diffusion-and-flow-policies.md)). O observation'dan tahmin edilen bir chunk'ta $i$. eleman gösterimlerin giderek artan bir kısmında "çoktan hareket ediyor"dur; bu yüzden medyan chunk bir süre bekler, sonra ayrılır, yeter ki $k$ ayrılışı içerecek kadar uzun olsun.

**Bir chunk'ı uygulamanın iki yolu.**

| | open-loop chunk'lar | temporal ensembling |
|---|---|---|
| sorgu | her $k$ adımda | her adımda |
| $t$ adımındaki action | son chunk'ın $t \bmod k$. elemanı | $t$ için tahmin edilmiş tüm chunk elemanlarının ağırlıklı ortalaması |
| bozucuya tepki | chunk başına bir kez | her adımda |
| chunk sınırları | sıçrar | yumuşar |

Temporal ensembling (Böl. IV-A) şimdiki adım için yapılmış (en fazla $k$) tahmini ortalar:

$$a_t = \frac{\sum_i w_i\, \hat a_t^{(i)}}{\sum_i w_i}, \qquad w_i = e^{-m i},$$

burada $i = 0$ en eski tahmindir; yani eski tahminler daha ağırdır ve $m \to 0$ düz bir ortalamadır. Makaleye göre "a smaller m means faster incorporation", yani küçük $m$ yeni observation'ları daha hızlı dahil eder. Makale $m$ için bir değer vermiyor; LeRobot'un config yorumu orijinalin $m = 0{,}01$ kullandığını söylüyor ([`configuration_act.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/act/configuration_act.py)).

**ACT'nin mimarisi** (Böl. IV; bkz. AIEFS: [Full Transformer](https://aiengineeringfromscratch.com/lesson?path=phases/07-transformers-deep-dive/05-full-transformer), [VAE](https://aiengineeringfromscratch.com/lesson?path=phases/08-generative-ai/02-autoencoders-vae)):

| parça | ne yapar |
|---|---|
| CVAE encoder (yalnız eğitimde) | [CLS], joint pozisyonları ve gösterilen chunk üzerinde transformer → bir "stil değişkeni" $z$ için diyagonal Gauss |
| backbone | kamera başına ResNet18; dört 480×640 görüntü → 2D sinüzoidal pozisyon embedding'li 15×20×512 feature map'ler |
| transformer encoder | görüntü feature'ları + joint pozisyonları + $z$, her biri 512 boyuta eşlenmiş |
| transformer decoder | cross-attention, $k \times 14$'lük bir chunk verir (iki 7-DoF kol) |
| loss, inference | L1 reconstruction + KL; test zamanında $z = 0$ (prior ortalaması), deterministik |

Yaklaşık 80M parametre, görev başına tek bir RTX 2080 Ti'da ~5 saat eğitim, 50 Hz control, $k = 100$, gerçek görevlerin çoğunda 50 gösterim. CVAE hedefi olmadan başarı insan verisinde %35,3'ten %2'ye düştü, scripted veride neredeyse fark yoktu (Şek. 8c, 2 simülasyon görevi); ensembling ACT'ye %3,3 ekledi (Şek. 8b).

## Deney

Script'in tamamı: [`experiments/31-act/experiment.py`](../../experiments/31-act/experiment.py). MuJoCo 3.17.0, NumPy 2.5.4, PyTorch 2.14.1 (CPU, 2 thread), Apple Silicon Mac (Apple M5), deterministik seed'ler. `output.txt`'teki duvar saati süreleri yaklaşıktır (ağır yüklü makine).

**Görev.** 2 segmentli düzlemsel bir kol (0,30 m + 0,25 m, position servo'lar `kp=20`, `kv=1.5`), 50 Hz control, rastgele tork bozucusu (eklem başına 0,2 N·m standart sapma, her control adımında yeniden çekilir). Scripted gösterici hedefin inverse kinematics çözümüne doğru gider (hedef komutu q'nun en fazla 0,12 rad önünde), 1 cm içine girince iç sayacını başlatır, 25 adım (0,5 s) bekler, sonra sabit bir bırakma pozuna gider. Policy $(q, \dot q, q_{goal})$'ü görür: sayaç yok, faz yok. 100 gösterim × 150 adım.

**ACT'den korunan:** mutlak joint hedeflerinden chunk'lar, L1 loss, $m = 0{,}01$ ile open-loop'a karşı ensembling. **Atılan:** görüntüler, transformer, CVAE (en az scripted veride önemliydi; bizimki de öyle). Policy: MLP 6-256-256-$2k$, 3.000 Adam adımı, her $k$ ve eğitim seed'i için bir model (3 seed).

```python
for t in range(len(o)):                                # dataset: o_t -> a_{t..t+k-1}
    X.append(o[t]); Y.append(a_pad[t:t + k].reshape(-1))

w = np.exp(-m * (k - 1 - np.arange(k)))                # chunk offset j: older prediction, larger weight
chunk = pol(obs)                                       # temporal ensembling: query every step
num[:, t:t + k] += w[None, :, None] * chunk
den[:, t:t + k] += w[None, :, None]
a = num[:, t] / den[:, t]
```

**Başarı:** en az 0,2 s hedefin 3 cm içinde kalmak, sonra 4,0 s'de bırakma noktasının 3 cm içinde olmak; seed başına 200 yeni hedef, satır başına 600 episode, her satırda aynı bozucular. Hatalar: *ıska* (hedefe hiç ulaşmadı), *kısa* (0,2 s dolmadan ayrıldı), *takılı* (4,0 s'de hâlâ hedefte). Bekleme: başarılı episode'larda hedefte kalınan en uzun sürenin medyanı. Jerk: komut edilen joint hedeflerinin üçüncü farkının RMS'i (rad/s³).

**Tahmin:** küçük $k$ takılır; büyük open-loop chunk'lar çok seyrek tepki verir; ensembling yumuşatır.

| policy | başarı | %95 GA | ıska | kısa | takılı | bekleme (s) | jerk |
|---|---|---|---|---|---|---|---|
| gösterici | 200/200 | %98,1–100 | 0 | 0 | 0 | 0,66 | 3506 |
| $k=1$ | 18/600 | %1,9–4,7 | 2 | 26 | **543** | 0,44 | 2443 |
| $k=4$ open-loop | 20/600 | %2,2–5,1 | 0 | 27 | 535 | 0,33 | 1734 |
| $k=4$ ensembled | 14/600 | %1,4–3,9 | 1 | 23 | 545 | 0,59 | 677 |
| $k=8$ open-loop | 32/600 | %3,8–7,4 | 3 | 25 | 534 | 0,37 | 1694 |
| $k=8$ ensembled | 23/600 | %2,6–5,7 | 2 | 34 | 538 | 0,82 | 376 |
| $k=16$ open-loop | 431/600 | %68,1–75,3 | 62 | 87 | 6 | 0,48 | 3085 |
| $k=16$ ensembled | 160/600 | %23,3–30,3 | 5 | 1 | 362 | 0,81 | 543 |
| $k=32$ open-loop | 297/600 | %45,5–53,5 | 200 | 103 | 0 | 0,38 | 3875 |
| $k=32$ ensembled | **552/600** | %89,6–93,9 | 39 | 9 | 0 | 0,58 | **433** |

8 ile 16 arasındaki sıçrama neden? Her policy'yi (seed 0) 200 hedef için tam hedefin üstünde durağanken sorgula ve hedeften 0,02 rad'dan fazla ayrılan ilk chunk elemanını bul:

| $k$ | 1 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| ayrılış içeren chunk'lar | 0/200 | 0/200 | 8/200 | 200/200 | 200/200 |
| medyan ilk ayrılış indeksi | – | – | 6,0 | 9,0 | 8,0 |

Bekleme state'inden tahmin edilen chunk yaklaşık 8–9 adım bekliyor, sonra ayrılıyor. Daha kısa bir chunk baştan sona "bekle"dir: $k \le 8$, iki şemada da 600 episode'un 534–545'inde takılı kalıyor. Bu eşiğin üstünde şema belirleyici:

- **$k=16$ open-loop (%71,8)** chunk'ın tamamına bağlı kalır, ayrılış elemanları uygulanır. **$k=16$ ensembled (%26,7)** ayrılan elemanları (eski sorgulardan) bekleyen elemanlarla (aynı durağan state'teki yeni sorgulardan) ortalar. 362 takılı episode'la tutarlı, ölçülmemiş bir hipotez: ortalanmış komut bekleme state'inden çıkmaya yetecek kadar oynamıyor, sonraki sorgular yine "bekle" diyor.
- **$k=32$ open-loop (%49,5)** 600 episode'un 200'ünde hedefi ıskalıyor: kör uygulanan 0,64 s'lik komut, 3 cm'lik toleransta bozucuyu düzeltemiyor. Aynı modelleri ensembling ile kullanmak **%92,0** veriyor, en iyi satır.
- **Jerk.** Ensembling komut jerk'ünü her $k$'da düşürüyor ($k=4$'te 1734 → 677, $k=32$'de 3875 → 433); her adımda gürültülü state'e tepki veren göstericinin (3506) bile altında.

## Boz

Beklemeyi uzat; $k = 32$ ensembled kalsın (seed 0, 200 episode, bütçe = bekleme + 3,5 s):

| bekleme | 0,5 s | 1,0 s | 1,5 s | 2,0 s |
|---|---|---|---|---|
| başarı | 185/200 | 65/200 | 2/200 | 0/200 |
| takılı | 0 | 1 | 196 | 154 |

Chunk 0,64 s'yi kapsıyor. Medyan argümanıyla tutarlı olarak: durağan state'ten bakınca gösterimlerin çoğu chunk'ın sonunda hâlâ bekliyorsa chunk baştan sona "bekle" olur ve kol hiç ayrılmaz (1,5 s'de 196/200 takılı). 1,0 s'de 200 episode'un 126'sı hedeften ayrıldı ama bütçe bittiğinde bırakma noktasında değildi (daha fazla ayrıştırılmadı). Pratik kural: chunk, gösterimlerdeki en uzun gözlenemeyen beklemeye göre uzun olmalı ya da eksik state (bir sayaç, gripper sinyali, geçmiş) observation'a girmeli.

## Gerçek robotta

- **LeRobot.** `lerobot-train --policy.type=act --dataset.repo_id=...` ACT'yi eğitir. Dokümantasyon onu denenecek ilk policy olarak öneriyor (~80M parametre, tek GPU'da 100k eğitim adımı için birkaç saat); SO-101'de değerlendirme `lerobot-rollout --robot.type=so101_follower ...` ile ([LeRobot ACT dokümanı](https://huggingface.co/docs/lerobot/act)). Config varsayılanları: `chunk_size=100`, `n_action_steps=100` (chunk'ın tamamı open-loop), `temporal_ensemble_coeff=None`; ensembling `n_action_steps=1` ister (her adımda sorgu); `kl_weight=10`, `latent_dim=32`, `n_decoder_layers=1` (yoruma göre orijinal 7 ayarlıyor ama bir bug yüzünden yalnızca ilki kullanılıyor) ([`configuration_act.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/act/configuration_act.py)).
- **Chunk uzunluğu zamandır.** 30 fps'te 100 adım 100/30 ≈ 3,3 s sürer; Boz tablosuna göre gösterimlerindeki beklemelerden uzun sürmeli.
- **SO-101 prosedürü.** Gripper kapanırken bilerek beklenen bir pick görevinin 50 episode'unu kaydet. ACT'yi varsayılanlarla eğit ve üç ayarı aynı 20+ nesne yerleşiminde değerlendir: varsayılanlar, `temporal_ensemble_coeff=0.01` ile `n_action_steps=1` ve kısa bir `chunk_size`. Başarıları ve takılı episode'ları Wilson aralıklarıyla say (ders 25). Gerçek kolda henüz ölçülmedi.

## Kendini dene

1. **Kolay.** $k=1$ policy'si neden geç ayrılmak yerine hedefte kalıyor?
2. **Orta.** Ölçülen ayrılış indeksi yaklaşık 9 iken $k = 16$ ensembled neden takılabiliyor da $k = 16$ open-loop ayrılıyor?
3. **Zor.** Observation'a varıştan beri geçen süreyi (ya da önceki action'ı) ekle ve $k = 1$'i yeniden koş. Takılı sayısı düşüyor mu? O zaman chunking ne zaman gerekir?

### Cevaplar

1. Hedefin üstünde durağanken gösterilen bir sonraki action, sonuncusu hariç her bekleme adımında "bekle"dir; L1-optimal çıktı olan medyan da "bekle"dir. Ölçülen: 543/600 takılı.
2. Olası mekanizma (ölçülmedi): şimdiki adım için yapılmış 16 tahminden yalnızca en az ~9 adım önce yapılanlar ayrılışı içerir; bekleyenlerle ağırlıklı ortalaması az oynar, kol durağan state'in yakınında kalır ve yeni sorgular "bekle" der. Open-loop ayrılış elemanlarını karıştırmadan uygular. Ölçülen: 362/600'e karşı 6/600 takılı.
3. Burada hesaplanmadı; koş ve gör.

## Akılda kalanlar ve kaynaklar

- Observation'ın süresini ölçemediği bir bekleme, L1 ile eğitilmiş tek adımlık bir policy'yi sonsuza kadar bekletir (simülasyonda 543/600 takılı); chunk ancak $k$ öğrenilen beklemeyi (burada 8–9 adım) aşarsa onun ötesine uzanır.
- Open-loop chunk'lar bağlanır ($k=16$'da %71,8) ama seyrek tepki verir ($k=32$'de 200 ıska); $k=32$'de ensembling en iyi sonucu (%92,0) verdi ve jerk'ü her $k$'da düşürdü.
- Chunk, gözlenemeyen en uzun beklemeden uzun sürmeli: 0,64 s'lik chunk'la başarı 0,5 s beklemede 185/200'den 2,0 s'de 0/200'e düştü.

Sonraki: [ders 32](../part-8-rl-for-robots/32-rl-environment.md), bir MuJoCo görevini RL environment'ı olarak sarmak.

**Kaynaklar**

- T. Z. Zhao, V. Kumar, S. Levine, C. Finn, "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware", 2023. [arXiv 2304.13705](https://arxiv.org/abs/2304.13705).
- LeRobot, [ACT dokümanı](https://huggingface.co/docs/lerobot/act) ve [`configuration_act.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/act/configuration_act.py).
- Tüm tablolar: [`experiments/31-act/output.txt`](../../experiments/31-act/output.txt) (27.10.2026).
