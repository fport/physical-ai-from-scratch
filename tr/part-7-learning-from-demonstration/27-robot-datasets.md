# 27 · Robot dataset'leri

<!-- meta -->
| | |
|---|---|
| **Bölüm** | [7 · Learning from demonstration](README.md) |
| **Çalışma türü** | CPU · Mac ya da Colab CPU |
| **Önkoşul** | [26](26-teleop-replay-data.md) |
| **Önce öğren** | yok |
| **Süre** | ~40 dk |
| **Kod** | [`experiments/27-robot-datasets/experiment.py`](../../experiments/27-robot-datasets/experiment.py) |
| **Durum** | ✅ tam |
<!-- /meta -->

> Bir robot dataset'i episode'lara bölünmüş bir zaman serisidir. Onun istatistikleriyle normalize et ve onu frame'e göre değil, her zaman episode'a göre böl.

## Problem

Ders 26, scripted SO-101 demonstration'larından 13680 frame üretti. Bunlar üzerinde eğitmeden önce, training sırasında gördüğün sayıların bir anlam taşıyıp taşımadığına üç karar belirler: her frame'in ne içerdiği ve episode'ların nerede başladığı, feature'ların nasıl ölçeklendiği ve verinin training ve validation'a nasıl bölündüğü. Sonuncusu yanlış yapması en kolay olanı. 40 ms arayla gelen ardışık frame'ler neredeyse aynıdır; rastgele bir frame seviyesi bölme her iki tarafa da neredeyse kopyaları koyar ve validation hatası komşu frame'ler arasındaki interpolation'ın ölçüsüne dönüşür. Bu ders o yanılsamanın verimizde ne kadar büyük olduğunu ölçüyor.

## Ana fikir

**Bir frame ne tutar.** Bir robot dataset'i episode'lardan $e$ oluşan bir kümedir; her biri sabit hızda frame'lerden $t = 0 \dots T_e - 1$ oluşan bir dizidir:

| Sütun | Anlamı | Bizim veri |
|---|---|---|
| `observation.state` | propriosepsiyon (joint pozisyonları), burada artı küp pozu | 9 float |
| `observation.images.*` | kamera frame'leri | yok (state tabanlı) |
| `action` | bu frame'de gönderilen komut | 6 joint hedefi |
| `timestamp` | episode başından beri saniye | $t/25$ |
| `frame_index`, `episode_index`, `index` | episode içindeki konum, episode id, global satır | int |
| `task_index` | hangi doğal dil görevi | 0: "pick up the cube" |

**LeRobot formatı (v3.0).** Dokümantasyonda ve kodda doğrulandı ([LeRobotDataset v3.0](https://huggingface.co/docs/lerobot/lerobot-dataset-v3); [`dataset_metadata.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/datasets/dataset_metadata.py) içinde `CODEBASE_VERSION = "v3.0"`):

- `meta/info.json`: şema (feature adları, dtype'lar, shape'ler), fps, codebase sürümü, yol şablonları.
- `meta/stats.json`: normalizasyon için feature başına istatistikler; kod min, max, mean, std, count ve 0,01, 0,10, 0,50, 0,90, 0,99 quantile'larını hesaplıyor ([`compute_stats.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/datasets/compute_stats.py)).
- `meta/episodes/`: episode başına kayıtlar (uzunluk, görevler, ortak dosyalardaki offset'ler), parçalı Parquet olarak.
- `meta/tasks.parquet`: tamsayı id'lere eşlenmiş görev metinleri (dokümantasyon sayfası hâlâ `tasks.jsonl` diyor; kod onu eski yol olarak tutuyor, [`utils.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/datasets/utils.py)).
- `data/chunk-000/file-000.parquet`: frame satırları, dosya başına birçok episode; `videos/<kamera>/chunk-000/file-000.mp4` de öyle.

`timestamp`, `frame_index`, `episode_index`, `index` ve `task_index` kullanıcı tarafından değil, kayıt aracı tarafından doldurulur ([`feature_utils.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/datasets/feature_utils.py)). v3'te episode sınırları dosya adlarında değil, metadata'da yaşar.

**Normalizasyon.** Girdiler ve hedefler training'den önce dataset istatistikleriyle yeniden ölçeklenir:

$$\text{MEAN\_STD: } \tilde x = \frac{x - \mu}{\sigma}, \qquad \text{MIN\_MAX: } \tilde x = 2\,\frac{x - x_{\min}}{x_{\max} - x_{\min}} - 1 .$$

LeRobot MIN_MAX, MEAN_STD, IDENTITY, QUANTILES ve QUANTILE10'u tanımlıyor ([`types.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/configs/types.py)); ACT config'i state ve action için varsayılan olarak MEAN_STD, Diffusion Policy config'i MIN_MAX kullanıyor ([ACT](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/act/configuration_act.py), [Diffusion](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/diffusion/configuration_diffusion.py)). İstatistikler training verisinden gelmeli ve policy ile birlikte taşınmalı: farklı istatistiklerle evaluate edilen bir policy farklı girdiler görür.

**Bölme.** Bir episode içindeki frame'ler güçlü biçimde korelelidir; bağımsız birimler episode'lardır (burada: küp konumları). Validation, [ders 25](25-evaluating-policies.md)'teki evaluation'ın test konumlarını dışarıda tutması gibi, episode'ları bütün olarak dışarıda tutmalı.

## Deney

Girdi: ders 26'dan [`demos.npz`](../../experiments/26-teleop-replay-data/demos.npz). Model: 9 state değerinden 6 action değerini tahmin eden bir MLP (2 × 256 ReLU, Adam, batch 256 ile 4000 adım), ders 28'in eğittiği network'ün aynısı, orada iki ek bileşenle (bkz. AIEFS: [Model Evaluation](https://aiengineeringfromscratch.com/lesson?path=phases/02-ml-fundamentals/09-model-evaluation)). Hata, derece cinsinden action RMSE'si, open loop (simülasyon yok). Tam script: [`experiments/27-robot-datasets/experiment.py`](../../experiments/27-robot-datasets/experiment.py), çıktı: [`output.txt`](../../experiments/27-robot-datasets/output.txt). NumPy 2.5.4, PyTorch 2.14.1 (CPU), Apple Silicon Mac (Apple M5), deterministik.

**A. Anatomi.** 120 episode, 13680 frame, 25 fps, 120'si de başarılı; episode sınırları 0, 114, 228, …, 13566. satırlarda. İstatistikler `stats.json`'ın tutacağı biçimde (alıntı):

| Feature | mean | std | min | max | q01 | q99 |
|---|---|---|---|---|---|---|
| shoulder_lift [rad] | −0,240 | 0,515 | −1,200 | 0,702 | −1,200 | 0,620 |
| gripper [rad] | 0,731 | 0,505 | 0,141 | 1,200 | 0,162 | 1,200 |
| cube_x [m] | 0,222 | 0,037 | 0,160 | 0,332 | 0,161 | 0,321 |
| cube_z [m] | 0,036 | 0,032 | 0,015 | 0,114 | 0,015 | 0,101 |
| action gripper [rad] | 0,561 | 0,674 | −0,170 | 1,200 | −0,170 | 1,200 |

İki ayrıntı yalnızca istatistiklerde görünüyor. Komutlanan gripper −0,170 rad'a (kapalı limit) iniyor, ölçülen çene ise hiç 0,141 rad'ın altına inmiyor: küp onu durduruyor ve servo sıkıyor. Ve küpler en fazla 0,28 m uzağa konduğu halde cube_x 0,332 m'ye ulaşıyor: en uzak episode'larda kaldırılan küp lift ve hold sırasında gripper'la birlikte dışarı doğru gidiyor (episode başına trajectory'leri kontrol ettik).

**B. Normalizasyon** (episode 0–79'da eğit, 80–99'da doğrula):

| Girdiler | normalizasyon | val RMSE [derece] | en kötü joint (RMSE) |
|---|---|---|---|
| küp m | yok | 0,306 | gripper (0,480) |
| küp mm | yok | 4,649 | gripper (8,164) |
| küp m | MEAN_STD | 0,264 | gripper (0,507) |
| küp mm | MEAN_STD | 0,265 | gripper (0,514) |
| küp m | MIN_MAX | 0,245 | gripper (0,448) |

Metre cinsinden ham feature'larımız zaten 0,01–1 mertebesinde; normalizasyon az kazandırıyor (0,306 → 0,264). Küpü milimetre olarak logla, normalize edilmemiş model 15 kat kötüleşiyor; normalize edilmiş olan değişmiyor. Normalizasyon modeli, loglarken birinin seçtiği birimlerden bağımsız yapar.

**C. Frame'e göre ve episode'a göre bölme.** *Tahmin et:* 10 episode'la hangi bölme daha düşük validation hatası raporlar ve hangisi doğrudur? İki bölme de 80/20: frame bölmesi $n$ episode'un tüm frame'lerini karıştırır, episode bölmesi episode'ların %80'inde (8, 24, 72) eğitir ve kalanında (2, 6, 18) doğrular. Test kümesi, hiçbir modelin görmediği 20 episode (100–119):

| Kullanılan episode | bölme | val RMSE [derece] | test RMSE [derece] | test / val |
|---|---|---|---|---|
| 10 | frame | 0,252 | 1,595 | 6,33 |
| 10 | episode | 0,640 | 1,804 | 2,82 |
| 30 | frame | 0,280 | 0,446 | 1,59 |
| 30 | episode | 0,544 | 0,479 | 0,88 |
| 90 | frame | 0,270 | 0,279 | 1,03 |
| 90 | episode | 0,343 | 0,323 | 0,94 |

Frame seviyesi validation hatası dataset büyüklüğü ne olursa olsun 0,25–0,28°'de kalıyor. Modelin zaten gördüğü frame'ler arasındaki 40 ms'lik boşlukları ne kadar iyi doldurduğunu ölçüyor ve 10 episode'la yeni küp konumlarındaki gerçek hatayı 6,33 kat küçük gösteriyor. Episode seviyesi sayı test hatasıyla birlikte hareket ediyor. Yalnız 2 validation episode'uyla (10 episode'luk satır) hâlâ iyimser, çünkü 2 episode küçük bir örneklem.

## Boz

Bozan parametre, frame bölmesi altında dataset büyüklüğü. 90 episode'la sızıntı neredeyse önemsiz (test/val 1,03), çünkü training episode'ları küp alanını zaten sık kaplıyor. 10 episode'la hatanın çoğunu gizliyor. Frame bölmesinin en çok zarar verdiği yer küçük dataset'ler ve robot dataset'leri küçüktür: LeRobot'un kendi rehberi 50 episode'dan başlıyor ([il_robots](https://huggingface.co/docs/lerobot/il_robots)). Aynı sızıntı `episode_index`'i yok sayan her frame başına karıştırmada ortaya çıkar: sızdıran bir validation kümesiyle yürüyen bir early-stopping kuralı ya da hyperparameter araması overfit olmuş modeller seçer.

## Gerçek robotta

- `lerobot-record` ile yapılmış gerçek bir SO-101 kaydı zaten `episode_index`, `frame_index`, `timestamp` ve `task_index` içerir; `meta/stats.json`'ını kayıt aracı yazar ([LeRobotDataset v3.0](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)). `episode_index` ile böl; birkaç kayıt oturumu varsa bütün bir oturumu dışarıda tut, çünkü ışık, kalibrasyon ve operatör oturumlar arasında kayar.
- Gerçek gözlemler görüntü içerir ve onların istatistikleri kanal başınadır. Gerçek action'lar leader koldan gelir; bizim scripted olanlardan daha gürültülüdür ve iki operatör aynı görev için farklı action dağılımları üretir.
- Gerçek kolda henüz ölçülmedi: C kısmındaki frame-episode farkı, teleoperate edilmiş bir SO-101 dataset'inde. Prosedür: 50 episode kaydet, aynı MLP'yi state → action üzerinde eğit ve iki validation sayısını dışarıda tutulan 10 episode'la karşılaştır.

## Kendini dene

1. **Kolay.** Episode'larımızın hepsi 25 fps'te 114 frame. Her episode ne kadar sürüyor ve episode 2'nin frame 5'inin `index`'i ne?
2. **Orta.** Action −0,170 rad'a inerken ölçülen çene açısı neden hiç 0,141 rad'ın altına inmiyor? Yalnız `observation.state` üzerinde eğitilen bir policy bu fark hakkında ne öğrenmek zorunda?
3. **Zor.** 100 training episode'unu rastgele değil küp konumuna göre böl: y < 4 cm olan küplerde eğit, y ≥ 4 cm olanlarda doğrula. Validation hatasını rastgele episode bölmesine göre tahmin et, sonra çalıştır.

### Cevaplar

1. 114 / 25 = 4,56 s. `index` = 2 · 114 + 5 = 233.
2. Küp çeneyi durduruyor; komut kapalı limit olduğu için servo torque cap'ine kadar itiyor (ders 16). Policy, çeneyi küp genişliğinde durmuş gördüğünde −0,17 rad vermeli, yani hiçbir zaman state olarak gözlenmeyen bir komutu öğrenmeli.
3. Burada hesaplanmadı; çalıştır. Extrapolation bölmesi rastgele bir episode bölmesinden farklı bir soruyu test eder.

## Akılda kalanlar ve kaynaklar

- Bir robot dataset'i `episode_index` ile gruplanmış (observation, action, timestamp) frame'leridir; görev metinleri ve istatistikler metadata'dadır. LeRobot v3.0 Parquet/MP4 dosyası başına birçok episode saklar ve episode'ları `meta/episodes/` üzerinden çözer.
- Training kümesi istatistikleriyle normalize et: küpü mm olarak loglamak normalize edilmemiş hatayı 0,306°'den 4,649°'ye çıkardı, normalize edilmiş olan 0,265°'de kaldı.
- Episode'a göre böl: 10 episode'la frame bölmesi 0,252° raporladı, görülmemiş episode'larda gerçek hata 1,595°'ydi.

Sonraki: [ders 28 · Behavior cloning](28-behavior-cloning.md).

Kaynaklar:
- LeRobot, [LeRobotDataset v3.0](https://huggingface.co/docs/lerobot/lerobot-dataset-v3) (okundu 2026-10-27).
- LeRobot kaynak kodu (main, okundu 2026-10-27): [`datasets/utils.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/datasets/utils.py), [`datasets/compute_stats.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/datasets/compute_stats.py), [`datasets/feature_utils.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/datasets/feature_utils.py), [`datasets/dataset_metadata.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/datasets/dataset_metadata.py), [`configs/types.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/configs/types.py), [ACT config](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/act/configuration_act.py), [Diffusion config](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/diffusion/configuration_diffusion.py).
- LeRobot, [Imitation learning on real-world robots](https://huggingface.co/docs/lerobot/il_robots).
- Ölçülen tüm sayılar: [`experiments/27-robot-datasets/output.txt`](../../experiments/27-robot-datasets/output.txt).
