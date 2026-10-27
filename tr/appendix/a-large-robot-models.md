# Ek A · Büyük robot model'leri: π0, GR00T, Gemini Robotics

> Büyük robot policy'lerinin üç ailesi, makalelerinden okunmuş hâliyle: neyi tüketiyorlar, ne üretiyorlar, ne kadar hızlılar ve sen neyi kendin çalıştırabilirsin.

Üçü de [ders 41](../part-10-foundation-models-and-agents/41-vla.md) anlamında vision-language-action model'idir: pretrained bir vision-language backbone'u artı action chunk'ları üreten bir action modülü. Aşağıdaki sayılar birincil kaynaklardan alıntıdır; ders 41'e atıf yapılan yerler dışında hiçbirini biz ölçmedik.

| | π0 (Physical Intelligence) | GR00T N1 (NVIDIA) | Gemini Robotics (Gemini Robotics Team) |
|---|---|---|---|
| backbone | PaliGemma, 3B | Eagle-2 VLM ("System 2"), 1,34B | Gemini 2.0 |
| action modülü | flow-matching action expert, 300M | flow matching ile eğitilmiş diffusion transformer ("System 1") | bulutta VLA backbone'u + yerel action decoder |
| boyut | 3,3B | 2,2B (yayımlanan GR00T-N1-2B) | okuduğumuz bölümlerde belirtilmiyor |
| chunk | H = 50 | H = 16 | action chunk'ları |
| hız | RTX 4090'da chunk başına 73 ms (3 kamera); 50 Hz'e kadar kontrol | L40'ta 16 action'lık chunk başına 63,9 ms (bf16); System 2 10 Hz'de, System 1 action'ları 120 Hz'de | backbone < 160 ms, gözlemden chunk'a ~250 ms, efektif 50 Hz |
| veri | 10.000 saat dexterous veri, 7 robot konfigürasyonu, 68 görev, artı OXE, DROID, Bridge | gerçek robot trajectory'leri, insan videoları, sentetik veri | bir ALOHA 2 filosundan büyük bir teleoperasyon dataset'i |
| gösterilen donanım | tek kol, iki kol, mobil manipülatörler | Fourier GR-1 humanoid, iki elli görevler | iki kollu ALOHA 2; başka bir iki kollu platforma ve bir humanoid'e uyarlanmış |

**π0.** Black et al. ([arXiv 2410.24164](https://arxiv.org/abs/2410.24164)). Kod ve checkpoint'ler [openpi](https://github.com/Physical-Intelligence/openpi)'de: "pre-trained on 10k+ hours of robot data" olan π0, π0-FAST ve π0.5 temel modelleri, artı fine-tune edilmiş checkpoint'ler (örn. DROID, ALOHA, LIBERO). Orada listelenen gereksinimler: inference için > 8 GB GPU belleği, LoRA fine-tuning için > 22,5 GB, full fine-tuning için > 70 GB; yalnızca Ubuntu 22.04'te test edilmiş. LeRobot portu `lerobot/pi0_base`, 32 boyutlu action'lardan 50 adımlık chunk'larla float32'de 3501,4M parametre saklıyor ([ders 41](../part-10-foundation-models-and-agents/41-vla.md)'de dosyalarından okundu, [çıktı](../../experiments/41-vla/output.txt)).

**GR00T N1.** NVIDIA ([arXiv 2503.14734](https://arxiv.org/abs/2503.14734)), "an open foundation model for generalist humanoid robots". Kopyalamaya değer mühendislik fikri iki sistemli bölünme: yavaş bir akıl yürütme modülü (10 Hz), 16 adımlık chunk'lar üreten hızlı bir action modülünü (120 Hz) koşullandırıyor; ikisi uçtan uca birlikte eğitiliyor.

**Gemini Robotics.** Gemini Robotics Team ([arXiv 2503.20020](https://arxiv.org/abs/2503.20020)). İki model: robotları doğrudan kontrol eden ve yeni embodiment'lara fine-tune edilebilen bir VLA olan Gemini Robotics, ve embodied reasoning ekleyen Gemini Robotics-ER (nesne tespiti, işaretleme, trajectory ve grasp tahmini, 3D bounding box'lar). Latency bölünmesi (bulutta backbone, robotta decoder, chunking) büyük bir modelin 50 Hz'lik bir döngüye nasıl yetiştiğini gösteriyor.

**SO-101'e ne taşınır.** Üçü de yukarıdaki boyutlar için NVIDIA sınıfı bir GPU istiyor; hiçbiri bizim kolumuzda çalıştırılmadı. Küçük açık seçenek SmolVLA (450M, [Ek B](b-smolvla-finetuning.md)). Hangi modeli kullanırsan kullan, çıkışı [ders 42](../part-10-foundation-models-and-agents/42-llm-agents.md)'nin validation katmanından ve [ders 15](../part-4-dynamics-and-control/15-safety.md)'in limitlerinden geçer.

Kaynaklar: [π0, arXiv 2410.24164](https://arxiv.org/abs/2410.24164) · [openpi](https://github.com/Physical-Intelligence/openpi) · [GR00T N1, arXiv 2503.14734](https://arxiv.org/abs/2503.14734) · [Gemini Robotics, arXiv 2503.20020](https://arxiv.org/abs/2503.20020).
