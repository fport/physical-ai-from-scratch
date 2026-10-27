# 41 · Vision-language-action model'leri

<!-- meta -->
| | |
|---|---|
| **Bölüm** | [10 · Foundation model'ler ve agent'lar](README.md) |
| **Çalışma türü** | ART · hazır sonuçlar, CPU'da çalışır |
| **Önkoşul** | [31](../part-7-learning-from-demonstration/31-act.md) |
| **Önce öğren** | [Embodied VLAs: RT-2, OpenVLA, π0, GR00T](https://aiengineeringfromscratch.com/lesson?path=phases/12-multimodal-ai/21-embodied-vlas-openvla-pi0-groot) · [The Full Transformer — Encoder + Decoder](https://aiengineeringfromscratch.com/lesson?path=phases/07-transformers-deep-dive/05-full-transformer) |
| **Süre** | ~60 dk |
| **Kod** | [`experiments/41-vla/experiment.py`](../../experiments/41-vla/experiment.py) |
| **Durum** | ✅ tam |
<!-- /meta -->

> VLA, observation'ı bir cümle de içeren bir policy'dir. Robot için belirleyici tasarım kararı çıkış tarafındadır: joint başına bir token mı, yoksa sürekli action'lardan bir chunk mı?

## Problem

LeRobot'un SmolVLA kılavuzu bir SO-101 sahibine yaklaşık 50 episode kaydetmesini, `lerobot/smolvla_base`'i fine-tune etmesini ve `--task="Grasp a lego block and put it in the bin."` ile çalıştırmasını söylüyor ([docs](https://huggingface.co/docs/lerobot/smolvla)). Buna GPU saati harcamadan önce bir mühendisin bilmesi gerekenler: ağ tam olarak neyi okuyor (hangi kameralar, hangi state, ne kadar metin), ne yazıyor (bir action mı elli mi, token mı gerçek sayı mı) ve parametreleri nerede duruyor. Bu cevaplar, weight'leri indirmeden incelenebilen açık dosyalarda. Bu ders onları SmolVLA, π0 ve OpenVLA için okuyor, sonra token tabanlı çıkışın [ders 16](../part-4-dynamics-and-control/16-grasping-and-pick-place.md)'nın scripted SO-101 pick'i üzerinde neye mal olduğunu ölçüyor.

## Ana fikir

Vision-language-action model'i (VLA), observation'ı

$$o_t = [I_t^1,\dots,I_t^n,\ \ell_t,\ q_t]$$

olan bir policy'dir: kamera görüntüleri, dil talimatının token'ları ve proprioceptive state (eklem açıları). Çıkışı bir action chunk'ıdır: $A_t = [a_t, \dots, a_{t+H-1}]$ (π0 makalesi, [arXiv 2410.24164](https://arxiv.org/abs/2410.24164)). Gövde pretrained bir vision-language model'idir (VLM): bir vision encoder, dil modelinin embedding uzayına bir projector ve bir transformer dil modeli (bkz. AIEFS: [Embodied VLAs](https://aiengineeringfromscratch.com/lesson?path=phases/12-multimodal-ai/21-embodied-vlas-openvla-pi0-groot)). Robotiğe özgü kısım action head'dir.

**Ayrık action token'ları (OpenVLA).** Her action boyutu ayrı ayrı 256 bin'den birine ayrıklaştırılıyor; bin genişliği, training action'larının 1. ve 99. quantile'ı arasındaki aralığı eşit bölüyor. Llama tokenizer'ının en az kullanılan 256 token'ı bu bin'lerle değiştiriliyor ve model olağan next-token loss'uyla eğitiliyor ([arXiv 2406.09246](https://arxiv.org/abs/2406.09246), §3.2). 7 boyutlu bir action 7 token'dır ve token'lar art arda decode edilir.

**Flow matching ile sürekli chunk'lar (π0, SmolVLA).** Ayrı bir "action expert" transformer'ı VLM'in feature'larına attention yapar ve Gauss gürültüsünü action chunk'ına taşıyan bir hız alanı öğrenir. $A^\tau = \tau A + (1-\tau)\epsilon$, $\epsilon \sim \mathcal N(0, I)$ ile:

$$\mathcal L(\theta) = \mathbb E\,\big\|v_\theta(A^\tau, o) - (A - \epsilon)\big\|^2, \qquad A^{\tau+\delta} = A^\tau + \delta\, v_\theta(A^\tau, o),\ \ A^0 \sim \mathcal N(0, I)$$

π0 10 Euler adımı (δ = 0,1) ve H = 50 kullanıyor; 3B'lik bir PaliGemma VLM'i artı 300M'lik bir action expert, toplam 3,3B ([arXiv 2410.24164](https://arxiv.org/abs/2410.24164)). SmolVLA aynı fikri 450M parametrede tutuyor (expert'te yaklaşık 100M), SmolVLM-2'nin yalnızca ilk 16 katmanını, frame başına 64 visual token'ı ve n = 50'lik chunk'ları kullanıyor, expert'te cross- ve self-attention'ı dönüşümlü diziyor ([arXiv 2506.01844](https://arxiv.org/abs/2506.01844)).

| | OpenVLA | π0 | SmolVLA |
|---|---|---|---|
| action head | 256-bin token, autoregressive | flow-matching expert | flow-matching expert |
| çağrı başına çıkış | 1 action | chunk, H = 50 | chunk, n = 50 |
| pretraining verisi | 970k episode, Open X-Embodiment | 7 robot konfigürasyonu ve 68 görevden 10.000 saat, artı OXE, DROID, Bridge | 481 community dataset'i, 22,9k episode, 10,6M frame |

**Cross-embodiment.** Robotların action boyutu farklıdır. Aşağıdaki config'ler state ve action'ı sabit bir `max_action_dim` = 32'ye pad'liyor; böylece 6 boyutlu bir SO-101 ile daha yüksek boyutlu iki kollu bir sistem tek bir ağı paylaşıyor. **Fine-tuning** ardından pretrained model'i tek bir robota ve göreve uyarlıyor: SmolVLA'nın config'i vision encoder'ı donduruyor ve yalnızca expert'i eğitiyor; OpenVLA'nın LoRA'sı model parametrelerinin yalnızca %1,4'ünü eğiterek full fine-tuning'e denk geliyor ([arXiv 2406.09246](https://arxiv.org/abs/2406.09246), §5.3).

## Deney

**A. Artifact'leri oku.** Her model için `config.json`'u ve her `.safetensors` dosyasının yalnızca JSON header'ını çekiyoruz (8 baytlık bir uzunluk artı tensor adları ve şekillerinden oluşan bir tablo, HTTP Range isteğiyle), sonra parametreleri bileşen başına topluyoruz.

```python
n = struct.unpack("<Q", fetch(repo, fn, (0, 7)))[0]      # header length
header = json.loads(fetch(repo, fn, (8, 7 + n)))           # {tensor: {dtype, shape, offsets}}
counts[group_of(name)] += prod(shape)
```

*Tahmin et:* SmolVLA'nın 450M parametresinin çoğu nerede: vision'da mı, dilde mi, action'da mı?

Config'lerden okunan: `smolvla_base` 6 boyutlu bir state ve üç adet 3 × 256 × 256 kamera bekliyor, en fazla 48 dil token'ı alıyor, 32'ye pad'lenmiş 6 boyutlu bir action veriyor, `chunk_size` 50, `n_action_steps` 50, 10 flow adımı. `pi0_base`: üç adet 224 × 224 kamera, 32 boyutlu state ve action, chunk 50, 10 inference adımı, `gemma_2b` + `gemma_300m`. `openvla-7b`: 224 × 224 görüntü, `n_action_bins` 256, 7 boyutlu action'lar, 25 dataset için normalizasyon istatistikleri.

| parametre (tensor şekillerinden) | smolvla_base | pi0_base | openvla-7b |
|---|---|---|---|
| dil modeli | 204,6M | 1981,9M | 6607,6M |
| vision encoder | 86,4M | 412,4M | 730,9M |
| action expert | 98,2M | 311,5M | – |
| LM çıkış head'(ler)i | 47,3M | 790,0M | 131,3M |
| vision-language connector | 11,8M | 2,4M | 71,4M |
| action / state / time projeksiyonları | 1,6M | 3,2M | – |
| **toplam** | **450,0M** | **3501,4M** | **7541,2M** |
| çekilen header baytı | 73.056 | 129.872 | 126.432 |

Toplamlar makalelerdeki 450M ve 7B ile uyuşuyor. Saklanan π0 checkpoint'i makaledeki 3,3B'ye karşı 3501,4M tutuyor; farkın çoğu, π0'ın kullanmadığı action expert'in 263,3M'lik vocabulary head'i (action'lar küçük `action_out_proj` üzerinden çıkıyor): 3501,4M − 263,3M = 3238,1M. PaliGemma'nın 526,6M'lik head'i ise gerekli: header'da ayrı bir token embedding tensor'ü yok, yani bu head tied input embedding olarak da kullanılıyor. Üç header toplam 329.360 bayt; hiç weight indirilmedi. SmolVLA'nın bütçesinin nereye gittiğine dikkat: 204,6M dil modeli, 98,2M action expert. Çağrı başına SmolVLA ve π0, 10 flow adımından sonra 50 × 32 sürekli değer döndürüyor; OpenVLA 7 token döndürüyor, yani kontrol adımı başına 7 decoder geçişi.

**B. Bin sayısını sweep et.** Ders 16'nın scripted pick'ini al (MuJoCo, 50 Hz'de 6 joint hedefinden oluşan 229 action), her boyutu OpenVLA tarzında token'laştır, bin merkezlerine geri çevir ve sonucu aynı sahnede fiziksel başarı testiyle (küpün altı ≥ 4 cm, iki çene de contact'ta) açık döngü replay et. Tip hatası, forward kinematics'ten gelen gripper frame konumu. Script'in tamamı: [`experiments/41-vla/experiment.py`](../../experiments/41-vla/experiment.py). MuJoCo 3.17.0, NumPy 2.5.4, Python 3.14, Apple M5, deterministik.

*Tahmin et:* kol joint'leri kabaca bir radyan aralığa yayılıyor; 256 bin derecenin küçük bir kesri kadar hata vermeli.

| bin | bit/boyut | RMS kol joint hatası | maks kol joint hatası | ort. tip hatası | maks tip hatası | replay | küpün altı |
|---|---|---|---|---|---|---|---|
| 1024 | 10 | 0,012° | 0,034° | 0,07 mm | 0,17 mm | kalktı | 7,24 cm |
| 256 | 8 | 0,048 | 0,136 | 0,26 | 0,60 | kalktı | 7,21 |
| 64 | 6 | 0,209 | 0,542 | 1,24 | 2,00 | kalktı | 6,96 |
| 16 | 4 | 0,745 | 2,168 | 3,51 | 9,54 | kalktı | 6,68 |
| 8 | 3 | 1,566 | 4,336 | 8,83 | 19,75 | kalktı | 5,19 |
| 4 | 2 | 3,161 | 8,672 | 15,69 | 40,36 | **kalkmadı** | 0,00 |

Orijinal demonstration küpü 7,37 cm'ye kaldırdı; token'laştırılmamış 50 Hz replay'i 7,25 cm'ye. 256 bin'de hata gripper'da 0,26 mm, döngüdeki diğer her şeyin çok altında: token'ın bedeli çözünürlük değil. Bedel, bir action'ın boyut başına bir decoder geçişine mal olması ve clipping aralığının training verisinden dondurulmuş olması. Burada değerlerin %0,00'ı aralığın dışına düştü: bu demonstration uç pozlarını uzun süre tutuyor, bu yüzden 1. ve 99. percentile'ları minimum ve maksimumuna eşit.

## Boz

Bin sayısını düşür. Çıkarılan her bit hatayı kabaca ikiye katlıyor, uniform quantization'ın öngördüğü gibi (kol joint'i RMS 0,745° → 1,566° → 3,161°, 16 → 8 → 4 bin). Grasp 8 bin'e kadar 19,75 mm'lik en kötü tip hatasıyla dayanıyor, ama küp daha alçakta kalıyor (5,19 cm). 4 bin'de en kötü tip hatası, 40,36 mm, 30 mm'lik küpten büyük ve replay küpü kaldırmıyor. Token head'i, hata sınırı bilinen bir quantizer'dır; bin sayısını vocabulary boyutundan değil, görevin toleransından seç (burada: çenede birkaç milimetre).

## Gerçek robotta

- **SmolVLA ile SO-101.** LeRobot'un prosedürü: ~50 episode kaydet (referans dataset'lerinde 5 küp pozisyonunun her biri için "10 episodes"; 25 episode "was not enough"), `lerobot-train --policy.path=lerobot/smolvla_base` ile fine-tune et (20k adım "~4 hrs on a single A100"), `lerobot-rollout --robot.type=so101_follower` ile çalıştır ([docs](https://huggingface.co/docs/lerobot/smolvla)). GPU kısmı [Ek B](../appendix/b-smolvla-finetuning.md)'de. **Bizim kolumuzda henüz ölçülmedi.**
- **Latency bir bütçedir.** π0 üç kamerayla RTX 4090'da chunk başına 73 ms raporluyor ([arXiv 2410.24164](https://arxiv.org/abs/2410.24164), Tablo I); OpenVLA bir RTX 4090'da "approximately 6Hz" ([arXiv 2406.09246](https://arxiv.org/abs/2406.09246)). Yavaş bir modelin 30–50 Hz'lik bir servo döngüsünü sürmesini sağlayan chunking'dir; SmolVLA buna asenkron inference ekliyor, böylece sıradaki chunk mevcut olan yürütülürken hesaplanıyor ([arXiv 2506.01844](https://arxiv.org/abs/2506.01844)). Mac sayıları: [Ek C](../appendix/c-inference-speed-on-mac.md).
- **π0 için donanım.** openpi, inference için > 8 GB, LoRA fine-tuning için > 22,5 GB GPU belleği listeliyor ve yalnızca Ubuntu 22.04'te test edildiğini belirtiyor ([openpi](https://github.com/Physical-Intelligence/openpi)). Bu çıkışların herhangi biriyle servo'lar arasında [ders 15](../part-4-dynamics-and-control/15-safety.md)'in safety katmanı durur.

## Kendini dene

1. **Kolay.** OpenVLA tarzı token'larla 6 boyutlu bir SO-101 action'ı kontrol adımı başına kaç token ister ve 30 Hz'de saniyede kaç decoder geçişi eder?
2. **Orta.** 256 bin satırında en büyük joint hatası 0,136°. Bu, aralığı en geniş joint'in yarım bin'iyse, o aralık kabaca ne kadar geniş?
3. **Zor.** Uniform bin'leri quantile bin'leriyle (bin başına eşit sayıda training değeri) değiştir ve B bölümünü yeniden koştur. Koşturmadan önce tahmin et: en çok hangi satırlar değişir ve neden?

### Cevaplar

1. Adım başına 6 token; saniyede 6 × 30 = 180 decoder geçişi (chunking olmadan).
2. Bir bin ≈ 2 × 0,136° = 0,272°, dolayısıyla aralık ≈ 0,272° × 256 ≈ 70°.
3. Burada hesaplanmadı; koştur.

## Akılda kalanlar ve kaynaklar

- VLA (görüntüler, talimat, eklem state'i) → action eşlemesi yapar; π0 ve SmolVLA bir flow-matching expert'ten 50 adımlık sürekli chunk'lar verir, OpenVLA ise her seferinde bir boyut olmak üzere 256-bin token'lar.
- Açık config'ler ve safetensors header'ları (üç model için 329.360 bayt header) girişleri, chunk uzunluğunu ve parametre bütçesini verir: SmolVLA 450,0M, bunun 98,2M'si action expert'te.
- SO-101 pick'inde 256 bin 0,26 mm tip hatasına mal oluyor; 4 bin (en kötü durumda 40,36 mm) grasp'i başarısız kılıyor.

Sıradaki: [ders 42 · Robot tool'ları olan LLM agent'ları](42-llm-agents.md).

Kaynaklar:
- Kim et al., [OpenVLA](https://arxiv.org/abs/2406.09246) (arXiv 2406.09246); config [`openvla/openvla-7b`](https://huggingface.co/openvla/openvla-7b).
- Black et al., [π0: A Vision-Language-Action Flow Model](https://arxiv.org/abs/2410.24164) (arXiv 2410.24164); config [`lerobot/pi0_base`](https://huggingface.co/lerobot/pi0_base); [openpi](https://github.com/Physical-Intelligence/openpi).
- Shukor et al., [SmolVLA](https://arxiv.org/abs/2506.01844) (arXiv 2506.01844); [LeRobot SmolVLA docs](https://huggingface.co/docs/lerobot/smolvla); config [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base) (Apache-2.0).
- Ölçülen tüm sayılar: [`experiments/41-vla/output.txt`](../../experiments/41-vla/output.txt).
