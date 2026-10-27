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

### Sade anlatımla

Resimlere bakabilen bir sohbet modeli düşün. Ona üç kamera karesini, robotun o anki eklem açılarını ve "lego'yu kutuya koy" cümlesini göster; cevabı kelimelerle değil motor komutlarıyla versin. Vision-language-action model'i (VLA) budur: çıkış token'ları joint hedefleri olan bir LLM. Ağın çoğu pretrained görüntü-metin modelidir; robota özgü kısım sondaki küçük bir parçadır, yani modelin iç feature'larını bir servonun izleyebileceği sayılara çeviren **action head**.

Bu head'i kurmanın iki yolu var. Birincisi action'a metin gibi davranır: her joint hedefini 256 seviyeden birine yuvarlar ve her seviyeyi bir kelime olarak yazar; joint başına bir kelime, kelime başına bir decoder geçişi. İkincisi küçük ek bir ağa, gerçek sayılardan oluşan bütün bir **chunk**'ı, örneğin sonraki 50 adımı, tek seferde yazdırır: rastgele gürültüden başlar ve onu 10 küçük adımda temizler, tıpkı görüntü diffusion modellerinin gürültüyü resme çevirmesi gibi.

Ödünleşim, yazılım açısından bekleyeceğin şeydir. Yuvarlamak çözünürlüğe mal olur (ne kadar, deneyin sorusu), joint başına bir kelime yazmak ise joint başına milyarlarca parametrelik bir modelin bir forward pass'ine mal olur. Bir chunk 50 action için tek bir çağrıdır; yavaş bir modelin hızlı bir servo döngüsünü sürebilmesini sağlayan da budur.

### Veri akışı

```mermaid
flowchart TD
    I["kamera görüntüleri"] --> V["vision encoder<br/>+ projector"]
    L["talimat metni"] --> M["dil modeli<br/>(pretrained VLM)"]
    V --> M
    Q["eklem açıları"] --> M
    M --> T["token head (OpenVLA)<br/>256 bin, joint başına bir token"]
    M --> F["action expert (π0, SmolVLA)<br/>flow matching, 50 adımlık chunk"]
    T --> S["joint hedefleri<br/>→ safety katmanı → servolar"]
    F --> S
```

### İki action head, kod olarak

Token head'i, muhtemelen başka bir iş için daha önce yazdığın bir quantizer'dır. Flow head'i ise hızını bir ağdan alan küçük bir ODE çözücüsüdür.

```python
# Token head (OpenVLA-style), per action dimension
def tokenize(a, lo, hi, n_bins=256):          # lo, hi: 1st and 99th percentile of the training actions
    x = (np.clip(a, lo, hi) - lo) / (hi - lo)  # 0..1
    return np.minimum((x * n_bins).astype(int), n_bins - 1)   # bin index = the "word"

def detokenize(k, lo, hi, n_bins=256):
    return lo + (k + 0.5) / n_bins * (hi - lo)  # centre of the bin

# Flow head (π0 / SmolVLA): turn noise into a chunk of H actions in N Euler steps
def sample_chunk(v_theta, obs, H=50, dim=32, N=10):
    A = np.random.randn(H, dim)                 # start: pure Gaussian noise
    for i in range(N):
        tau = i / N                             # 0 = noise, 1 = clean actions
        A = A + (1 / N) * v_theta(A, tau, obs)  # follow the learned velocity field
    return A                                    # H future actions, real numbers
```

| Head | Çıkış | Hata | Kontrol adımı başına maliyet |
|---|---|---|---|
| token | $k = \big\lfloor B\,\frac{a - a_{lo}}{a_{hi} - a_{lo}} \big\rfloor$, $\hat a = a_{lo} + (k + \tfrac12)\Delta$ | $\lvert a - \hat a\rvert \le \Delta/2$, $\Delta = \frac{a_{hi} - a_{lo}}{B}$ | $D$ decoder geçişi |
| flow matching | $A^{\tau+\delta} = A^\tau + \delta\, v_\theta(A^\tau, o)$, $A^0 \sim \mathcal N(0, I)$ | öğrenilmiş, sabit bir sınırı yok | $H$ action'lık chunk başına $1/\delta$ expert geçişi |

$a$ bir action boyutu (bir joint hedefi), $B$ bin sayısı, $[a_{lo}, a_{hi}]$ clipping aralığı, $\Delta$ bin genişliği, $D$ action boyutu, $A$ $H$ action'lık bir chunk, $\tau \in [0, 1]$ flow zamanı, $\delta$ adım boyu ve $v_\theta$ observation $o$'yu gören öğrenilmiş hız ağıdır.

<details>
<summary>Matematik ayrıntısı: observation, flow-matching loss'u ve quantization hatası</summary>

**Observation ve çıkış.** Bir VLA'nın observation'ı

$$o_t = [I_t^1,\dots,I_t^n,\ \ell_t,\ q_t]$$

kamera görüntüleri, dil talimatının token'ları ve proprioceptive state'tir (robotun kendi eklem açıları). Çıkışı bir action chunk'ıdır: $A_t = [a_t, \dots, a_{t+H-1}]$ (π0 makalesi, [arXiv 2410.24164](https://arxiv.org/abs/2410.24164)). Gövde pretrained bir vision-language model'idir (VLM): bir vision encoder, dil modelinin embedding uzayına (kelimeler için kullandığı vektörlere) bir projector ve bir transformer dil modeli (bkz. AIEFS: [Embodied VLAs](https://aiengineeringfromscratch.com/lesson?path=phases/12-multimodal-ai/21-embodied-vlas-openvla-pi0-groot)).

**Ayrık action token'ları (OpenVLA).** Her action boyutu ayrı ayrı 256 bin'den birine ayrıklaştırılıyor; bin genişliği, training action'larının 1. ve 99. quantile'ı arasındaki aralığı eşit bölüyor. Llama tokenizer'ının en az kullanılan 256 token'ı bu bin'lerle değiştiriliyor ve model olağan next-token loss'uyla eğitiliyor ([arXiv 2406.09246](https://arxiv.org/abs/2406.09246), §3.2). 7 boyutlu bir action 7 token'dır ve token'lar autoregressive, yani art arda, her biri öncekilere bakarak decode edilir.

**Quantization hatası.** Bin merkezine decode etmek en fazla $\Delta/2$ hata bırakır; gerçek değer bin içinde eşit dağılmışsa RMS hata $\Delta/\sqrt{12}$'dir. İkisi de $1/B$ ile ölçeklenir; dolayısıyla çıkarılan her bit ($B \to B/2$) hatayı ikiye katlar. $[a_{lo}, a_{hi}]$ dışındaki değerler clip'lenir: aralık training verisinden dondurulmuştur.

**Flow matching (π0, SmolVLA).** Ayrı bir "action expert" transformer'ı VLM'in feature'larına attention yapar ve Gauss gürültüsünü action chunk'ına taşıyan bir hız alanı öğrenir. $A^\tau = \tau A + (1-\tau)\epsilon$, $\epsilon \sim \mathcal N(0, I)$ ile gürültüden veriye giden düz yolun hızı sabittir, $\frac{d}{d\tau}A^\tau = A - \epsilon$; ağ bunu tahmin etmek üzere eğitilir:

$$\mathcal L(\theta) = \mathbb E\,\big\|v_\theta(A^\tau, o) - (A - \epsilon)\big\|^2, \qquad A^{\tau+\delta} = A^\tau + \delta\, v_\theta(A^\tau, o),\ \ A^0 \sim \mathcal N(0, I)$$

Inference'ta ODE, $\tau = 0$'dan $1$'e $\delta$ boyutlu Euler adımlarıyla integre edilir; $\delta = 0{,}1$ 10 adım verir.

Daha derine: [Embodied VLAs: RT-2, OpenVLA, π0, GR00T](https://aiengineeringfromscratch.com/lesson?path=phases/12-multimodal-ai/21-embodied-vlas-openvla-pi0-groot), [The Full Transformer](https://aiengineeringfromscratch.com/lesson?path=phases/07-transformers-deep-dive/05-full-transformer); birincil kaynaklar: [OpenVLA](https://arxiv.org/abs/2406.09246), [π0](https://arxiv.org/abs/2410.24164), [SmolVLA](https://arxiv.org/abs/2506.01844).

</details>

### Üç model yan yana

π0 10 Euler adımı (δ = 0,1) ve H = 50 kullanıyor; 3B'lik bir PaliGemma VLM'i artı 300M'lik bir action expert, toplam 3,3B ([arXiv 2410.24164](https://arxiv.org/abs/2410.24164)). SmolVLA aynı fikri 450M parametrede tutuyor (expert'te yaklaşık 100M), SmolVLM-2'nin yalnızca ilk 16 katmanını, frame başına 64 visual token'ı ve n = 50'lik chunk'ları kullanıyor, expert'te cross- ve self-attention'ı dönüşümlü diziyor ([arXiv 2506.01844](https://arxiv.org/abs/2506.01844)).

| | OpenVLA | π0 | SmolVLA |
|---|---|---|---|
| action head | 256-bin token, autoregressive | flow-matching expert | flow-matching expert |
| çağrı başına çıkış | 1 action | chunk, H = 50 | chunk, n = 50 |
| pretraining verisi | 970k episode, Open X-Embodiment | 7 robot konfigürasyonu ve 68 görevden 10.000 saat, artı OXE, DROID, Bridge | 481 community dataset'i, 22,9k episode, 10,6M frame |

**Cross-embodiment**, gövdeleri farklı robotlar için tek bir ağ demek. Robotların action boyutu farklıdır; bu yüzden aşağıdaki config'ler state ve action'ı sabit bir `max_action_dim` = 32'ye pad'liyor (küçük bir robot fazla slot'ları padding olarak bırakır): 6 boyutlu bir SO-101 ile daha yüksek boyutlu iki kollu bir sistem tek bir ağı paylaşıyor. **Fine-tuning** ardından pretrained model'i tek bir robota ve göreve uyarlıyor: SmolVLA'nın config'i vision encoder'ı donduruyor ve yalnızca expert'i eğitiyor; OpenVLA'nın LoRA'sı (dondurulmuş weight'lere eklenen küçük low-rank matrisler) model parametrelerinin yalnızca %1,4'ünü eğiterek full fine-tuning'e denk geliyor ([arXiv 2406.09246](https://arxiv.org/abs/2406.09246), §5.3).

## Deney

**A. Artifact'leri oku.** Bir `.safetensors` weight dosyası 8 baytlık bir uzunlukla ve ardından tensor adları ile şekillerinden oluşan bir JSON tablosuyla başlar. Her model için `config.json`'u ve yalnızca bu header'ı çekiyoruz (HTTP Range isteğiyle; bu istek sunucudan dosyanın tamamı yerine bir bayt aralığı ister), sonra parametreleri bileşen başına topluyoruz.

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

Toplamlar makalelerdeki 450M ve 7B ile uyuşuyor. Saklanan π0 checkpoint'i makaledeki 3,3B'ye karşı 3501,4M tutuyor; farkın çoğu, π0'ın kullanmadığı action expert'in 263,3M'lik vocabulary head'i (vocabulary'deki her kelimeye puan veren katman); action'lar küçük `action_out_proj` üzerinden çıkıyor: 3501,4M − 263,3M = 3238,1M. PaliGemma'nın 526,6M'lik head'i ise gerekli: header'da ayrı bir token embedding tensor'ü yok, yani bu head tied input embedding olarak da kullanılıyor. Üç header toplam 329.360 bayt; hiç weight indirilmedi. SmolVLA'nın bütçesinin nereye gittiğine dikkat: 204,6M dil modeli, 98,2M action expert. Çağrı başına SmolVLA ve π0, 10 flow adımından sonra 50 × 32 sürekli değer döndürüyor; OpenVLA 7 token döndürüyor, yani kontrol adımı başına 7 decoder geçişi.

**B. Bin sayısını sweep et.** Ders 16'nın scripted pick'ini al (MuJoCo, 50 Hz'de 6 joint hedefinden oluşan 229 action), her boyutu OpenVLA tarzında token'laştır, bin merkezlerine geri çevir ve sonucu aynı sahnede fiziksel başarı testiyle (küpün altı ≥ 4 cm, iki çene de contact'ta) açık döngü replay et (kayıtlı action'lar sahneye bakılmadan oynatılır). Tip hatası, forward kinematics'ten ([ders 8](../part-3-kinematics/08-forward-kinematics.md)) gelen gripper frame konumu. Script'in tamamı: [`experiments/41-vla/experiment.py`](../../experiments/41-vla/experiment.py). MuJoCo 3.17.0, NumPy 2.5.4, Python 3.14, Apple M5, deterministik.

*Tahmin et:* kol joint'leri kabaca bir radyan aralığa yayılıyor; 256 bin derecenin küçük bir kesri kadar hata vermeli.

![Gripper tip hatası bin sayısı her iki katına çıktığında yaklaşık yarıya iniyor: 4 bin'de en kötü durumda 40,36 mm, 1024 bin'de 0,17 mm. Replay küpü 8 bin'den itibaren kaldırıyor; en kötü hatanın 30 mm'lik küpü aştığı 4 bin'de başarısız.](../../experiments/41-vla/tokenization_vs_bins.png)

| bin | bit/boyut | RMS kol joint hatası | maks kol joint hatası | ort. tip hatası | maks tip hatası | replay | küpün altı |
|---|---|---|---|---|---|---|---|
| 1024 | 10 | 0,012° | 0,034° | 0,07 mm | 0,17 mm | kalktı | 7,24 cm |
| 256 | 8 | 0,048 | 0,136 | 0,26 | 0,60 | kalktı | 7,21 |
| 64 | 6 | 0,209 | 0,542 | 1,24 | 2,00 | kalktı | 6,96 |
| 16 | 4 | 0,745 | 2,168 | 3,51 | 9,54 | kalktı | 6,68 |
| 8 | 3 | 1,566 | 4,336 | 8,83 | 19,75 | kalktı | 5,19 |
| 4 | 2 | 3,161 | 8,672 | 15,69 | 40,36 | **kalkmadı** | 0,00 |

Orijinal demonstration küpü 7,37 cm'ye kaldırdı; token'laştırılmamış 50 Hz replay'i 7,25 cm'ye. 256 bin'de hata gripper'da 0,26 mm, döngüdeki diğer her şeyin çok altında: token'ın bedeli çözünürlük değil. Bedel, bir action'ın boyut başına bir decoder geçişine mal olması ve clipping aralığının training verisinden dondurulmuş olması. Burada değerlerin %0,00'ı aralığın dışına düştü: bu demonstration uç pozlarını uzun süre tutuyor, bu yüzden 1. ve 99. percentile'ları minimum ve maksimumuna eşit. (Grafik, `output.txt`'teki 32 ve 128 bin satırlarını da gösteriyor.)

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
- Ölçülen tüm sayılar: [`experiments/41-vla/output.txt`](../../experiments/41-vla/output.txt), grafik [`plot.py`](../../experiments/41-vla/plot.py) ile.
