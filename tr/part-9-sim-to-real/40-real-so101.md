# 40 · Gerçek SO-101'e deploy etmek

<!-- meta -->
| | |
|---|---|
| **Bölüm** | [9 · Sim-to-real](README.md) |
| **Çalışma türü** | HW · gerçek SO-101 |
| **Önkoşul** | [15](../part-4-dynamics-and-control/15-safety.md), [28](../part-7-learning-from-demonstration/28-behavior-cloning.md), [38](38-system-identification.md) |
| **Önce öğren** | yok |
| **Süre** | ~120 dk |
| **Kod** | [`experiments/40-real-so101/experiment.py`](../../experiments/40-real-so101/experiment.py) |
| **Durum** | ✅ tam |
<!-- /meta -->

> Donanımda hataların çoğu kayıt işidir: yanlış port, yanlış id, yanlış kalibrasyon, yanlış birim. Kol hareket etmeden önce bunları simülasyonda ve yazılı olarak kontrol et.

## Problem

Şimdiye kadar her şey simülasyonda çalıştı. LeRobot'ta gerçek SO-101 iş akışını yazmak kısa, ince bir şekilde yanlış yapmak kolay: kalibrasyon robotun `id`'si altında saklanıyor ve LeRobot rehberi teleoperation, kayıt ve değerlendirmede aynı `id`'nin kullanılması gerektiği konusunda uyarıyor ([LeRobot imitation learning rehberi](https://huggingface.co/docs/lerobot/il_robots)). Bir kalibrasyonla kaydedilmiş veride eğitilip başka bir kalibrasyonla çalıştırılan bir policy, aradaki fark kadar kaymış hedefler gönderir ve yazılımda hiçbir şey şikâyet etmez. Bu ders bütün deploy sürecini bir prosedür olarak yazıyor ve komut verilen bir trajectory'yi motorlara ulaşmadan reddeden bir simülasyon preflight'ı ekliyor.

## Ana fikir

### Sade anlatımla

Öğrenilmiş bir policy'yi kola deploy etmek bir build pipeline'ıdır: donanım kurulumu, veri, model, yayın. Her aşama daha önceki bir aşamanın ürettiği konfigürasyonu kullanır; hepsinden önemlisi, ham motor okumalarını policy'nin gördüğü joint değerlerine çeviren *kalibrasyon* dosyası. Bir kalibrasyonla eğitip başka biriyle çalıştırırsan hiçbir şey çökmez; her hedef sessizce kayar. Bu, bir modeli bir feature normalizasyonuyla eğitip başka biriyle servis etmenin robot versiyonudur.

Veriyi *teleoperation* ile kaydedersin: elle tuttuğun *leader* kolu hareket ettirirsin, motorlu *follower* kol onu taklit eder. Bu kayıtlarla bir policy eğitilir; *rollout* da onu sen olmadan follower üzerinde çalıştırır. Yani işin çoğu kayıt tutmaktır: ilk teleoperation'dan son rollout'a kadar port, id, kalibrasyon, kameralar ve frame rate aynı kalsın; komutlar motorlara ulaşmadan önce bir modele karşı kontrol edilsin.

```mermaid
flowchart TD
    P["portları bul,<br/>motor ID'lerini ayarla"] --> C["calibrate et<br/>(dosya adı id'den)"]
    C --> T["teleoperation<br/>leader → follower"]
    T --> R["episode kaydet<br/>LeRobotDataset"]
    R --> L["ACT'yi eğit"]
    L --> F["hedeflerin<br/>sim preflight'ı"]
    F --> O["kolda rollout"]
    C -. "aynı id, kameralar, fps" .-> O
```

### Pipeline ve değişmezleri

| Aşama | Komut | Ürettiği | Korunacak değişmez |
|---|---|---|---|
| Portlar | `lerobot-find-port` | her kol için bir port | aynı kol için aynı USB cihazı |
| Motor kurulumu | `lerobot-setup-motors` | EEPROM'da ID 1–6, ortak baudrate | her motor için bir kez |
| Kalibrasyon | `lerobot-calibrate` | `HF_LEROBOT_CALIBRATION` altında `<id>.json` | sonraki her komutta aynı `id` |
| Teleoperation | `lerobot-teleoperate` | leader → follower | kayıttan önce follower düzgün davranıyor |
| Kayıt | `lerobot-record` | `LeRobotDataset`, 30 fps | sabit kameralar, tutarlı görev |
| Eğitim | `lerobot-train` | ACT checkpoint'i | tam olarak o dataset üzerinde eğitilmiş |
| Rollout | `lerobot-rollout` | otonom episode'lar | kayıttakiyle aynı kameralar, fps, kalibrasyon |

(Baudrate seri bus'ın hızıdır; EEPROM, servo'nun güç kesilince silinmeyen kendi belleğidir.)

**Birimler.** Varsayılan `use_degrees = True` ile LeRobot beş kol joint'ini kalibrasyonda kaydedilen aralığın ortasına göre derece cinsinden, gripper'ı 0'dan 100'e raporluyor ([`config_so_follower.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/config_so_follower.py), [`motors_bus.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/motors/motors_bus.py)). Menagerie modeli kendi sıfırından radyan kullanıyor. Bir sim kontrolü ancak ikisi arasındaki ofseti ve işareti ölçtükten sonra anlamlıdır (ders 38).

## Deney

### Bölüm 1 · Donanım prosedürü (gerçek kolda henüz ölçülmedi)

Komutlar ve bilgiler [SO-101 sayfası](https://huggingface.co/docs/lerobot/so101), [imitation learning rehberi](https://huggingface.co/docs/lerobot/il_robots), [ACT sayfası](https://huggingface.co/docs/lerobot/act) ve [hesaplama donanımı rehberinden](https://huggingface.co/docs/lerobot/hardware_guide), 2026-10-27'de okundu.

1. **Kur:** LeRobot (kurulum rehberi) ve Feetech SDK'sı: `pip install -e ".[feetech]"`.
2. **Monte et.** Follower 1/345 dişli oranlı altı STS3215 servo kullanıyor; leader ise kendini taşıyabilsin ama elle kolayca hareket etsin diye 1/191 (pan, elbow), 1/345 (shoulder lift) ve 1/147 (wrist flex, wrist roll, gripper) karıştırıyor.
3. **Portları bul.** `lerobot-find-port`'u çalıştır ve istendiğinde kolun kablosunu çıkar; Linux'ta erişimi `sudo chmod 666 /dev/ttyACM0` ile ver.
4. **Motor ID'lerini ve baudrate'i ayarla**, her seferinde tek motor bağlıyken, gripper'dan (ID 6) başlayıp shoulder pan ile (ID 1) bitirerek; Waveshare kartında iki jumper da B kanalında:
   ```bash
   lerobot-setup-motors --robot.type=so101_follower --robot.port=/dev/ttyACM0
   lerobot-setup-motors --teleop.type=so101_leader --teleop.port=/dev/ttyACM1
   ```
5. **İki kolu da calibrate et:** her joint'i aralığının ortasına getir, Enter'a bas, sonra her joint'i tüm aralığı boyunca hareket ettir. `wrist_roll` tam tur dönen bir motor ve bütün encoder'ı kapsıyor ([`so_follower.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/so_follower.py)).
   ```bash
   lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower
   lerobot-calibrate --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=my_leader
   ```
6. **Teleoperation yap**, follower çalışacağın her yerde leader'ı izleyene kadar:
   ```bash
   lerobot-teleoperate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower \
     --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=my_leader
   ```
7. **Kaydet**, `hf auth login`'den sonra. Varsayılanlar: 60 s episode, 60 s reset, 50 episode (`--dataset.episode_time_s`, `--dataset.reset_time_s`, `--dataset.num_episodes`). Tuşlar: → sonraki, ← yeniden kaydet, Esc dur. Rehber en az 50 episode, nesne konumu başına 10, sabit kameralar ve yalnızca kamera görüntülerine bakarak kendinin yapabileceği bir görev öneriyor.
   ```bash
   lerobot-record --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower \
     --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
     --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=my_leader \
     --display_data=true --dataset.repo_id=${HF_USER}/so101_pick --dataset.num_episodes=50 \
     --dataset.single_task="Pick the cube and place it in the bin"
   ```
   Eğitimden önce bir episode'u `lerobot-replay ... --dataset.episode=0` ile kontrol et.
8. **ACT'yi eğit** ([ders 31](../part-7-learning-from-demonstration/31-act.md)), bir GPU'da (`--policy.device=cuda`; Apple silicon'da `mps`):
   ```bash
   lerobot-train --dataset.repo_id=${HF_USER}/so101_pick --policy.type=act \
     --output_dir=outputs/train/act_so101_pick --job_name=act_so101_pick \
     --policy.device=cuda --wandb.enable=true --policy.repo_id=${HF_USER}/act_so101_pick
   ```
   LeRobot'un kendi rakamları (ACT sayfası; rakamlarını "indicative" olarak etiketleyen donanım rehberi): ACT'nin yaklaşık 80M parametresi var ve batch size 8'de yaklaşık 2–6 GB VRAM istiyor; ~45k frame'lik bir dataset üzerinde 5 epoch, RTX 4090 / 3090'da yaklaşık 30–60 dk, MPS ile bir M1/M2/M3 Max'te (batch size 4) yaklaşık 6–14 sa sürüyor; yalnızca CPU ile eğitim önerilmiyor.
9. **Rollout yap**, aynı port, id ve kameralarla:
   ```bash
   lerobot-rollout --strategy.type=base --policy.path=${HF_USER}/act_so101_pick \
     --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower \
     --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
     --task="Pick the cube and place it in the bin" --duration=60
   ```
   Kayıtlı değerlendirme için eval verisine bir `--dataset.repo_id` vererek `--strategy.type=sentry` ya da `episodic` kullan. Başarıyı bir güven aralığıyla $k/n$ olarak raporla ([ders 25](../part-7-learning-from-demonstration/25-evaluating-policies.md)).

### Bölüm 2 · Simülasyon preflight'ı

*Preflight*, herhangi bir şey hareket etmeden önce çalışan bir kontroldür. 9. adımdan önce ve kaydedilmiş ya da üretilmiş herhangi bir trajectory'yi tekrar oynatmadan önce onu modelden geçir. Burada: bir pick-and-place hedef dizisi (9 waypoint, 1,2 s'lik min-jerk segmentleri, yani yumuşak kalkış ve duruş; 30 fps'de 303 hedef, 10,1 s) Menagerie SO-101'in servo'larından oynatılıyor. Altı kontrol, eşikler bizim:

| kontrol | kural |
|---|---|
| range | her hedef model joint aralığının ≥ 5° içinde |
| step | 30 fps'lik tick başına hedef değişimi ≤ 0,10 rad (bkz. `max_relative_target`, ders 15) |
| track | kol joint'leri her tick'in sonunda hedeflerinin 5° yakınında |
| gravity | statik yerçekimi torku 2,94 N·m'lik servo limitinin ≤ %80'i |
| contact | oynatma sırasında kol–zemin ya da kol–kol contact'ı yok |
| sat (bilgi) | bir servo'nun tork limitinde olduğu adımların payı |

```python
for g in goals:                       # static gravity torque at every goal
    d.qpos[:6] = g; d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    grav = max(grav, np.abs(d.qfrc_bias[:5]).max())
```

Tam script: [`experiments/40-real-so101/experiment.py`](../../experiments/40-real-so101/experiment.py). MuJoCo 3.17.0, NumPy 2.5.4, Python 3.14, Apple Silicon Mac (Apple M5), deterministik, birkaç saniye CPU. *Tahmin et:* trajectory ne kadar hızlı oynatılabilir ve hangi yük fazla?

**A. Oynatma hızı** (yük yok):

| hız | margin [°] | step [rad] | track [°] | sat % | gravity [N·m] | contact | sonuç |
|---|---|---|---|---|---|---|---|
| ×0,5 | 10,00 | 0,026 | 0,32 | 37,2 | 0,446 | 0 | PASS |
| ×1 | 10,00 | 0,052 | 0,62 | 52,9 | 0,446 | 0 | PASS |
| ×2 | 10,00 | 0,103 | 1,00 | 63,8 | 0,446 | 0 | FAIL: step |
| ×3 | 10,00 | 0,153 | 1,56 | 65,4 | 0,446 | 0 | FAIL: step |
| ×4 | 10,00 | 0,207 | 5,38 | 74,8 | 0,446 | 0 | FAIL: step, track |

**B. Gripper frame'inde yük** (hız ×1):

| yük [kg] | 0,0 | 0,2 | 0,4 | 0,6 | 0,8 |
|---|---|---|---|---|---|
| gravity [N·m] | 0,446 | 1,096 | 1,747 | 2,397 | 3,047 |
| track [°] | 0,62 | 0,67 | 0,70 | 0,72 | 2,11 |
| sonuç | PASS | PASS | PASS | FAIL: gravity | FAIL: gravity |

**Gözlem.** Step kontrolü ×2'de devreye giriyor, takip çok daha sonra (×4) bozuluyor: robotta da uygulayabileceğin ucuz ve muhafazakâr bir vekil. Modeldeki servo her hızda adımların %37,2–74,8'inde tork limitinde; $k_p$ = 998 N·m/rad ile doğrusal bandı yalnızca 2,94/998 ≈ 2,9 mrad, bu yüzden saturation tek başına burada pek bir şey söylemiyor. 0,6 kg'da statik tork (2,397 N·m) limitin %80'ini geçiyor, kol ise hâlâ 0,72° ile takip ediyor: gravity kontrolü eksik margin'i kol bunu kanıtlamak zorunda kalmadan işaretliyor (ders 12'de uzatılmış kol 0,8 kg'da düştü).

## Boz

Her shoulder_lift hedefini, gerçek kolun sıfırı modelinkinden farklıymış gibi bir kalibrasyon ofsetiyle kaydır:

| ofset [°] | 0 | 5 | 10 | 20 | 30 |
|---|---|---|---|---|---|
| range margin [°] | 10,00 | 10,00 | 10,00 | 10,00 | 10,00 |
| track [°] | 0,62 | 0,62 | 21,28 | 45,43 | 39,53 |
| zemin contact'lı adım | 0 | 0 | 1521 | 4190 | 4927 |
| sonuç | PASS | PASS | FAIL: track, contact | FAIL | FAIL |

Range kontrolü her satırda geçiyor (10,00°'lik margin, gripper'ın kapalı hedefinin −10°'lik limitine uzaklığı); 10°'lik ofsetten itibaren gripper "aşağı" pozlarında zemine giriyor. Joint limitleri çalışma alanının bir modeli değildir. Preflight ancak LeRobot'un kalibre edilmiş dereceleri ile modelin açıları arasındaki eşleme kadar iyidir: onu ölç.

## Gerçek robotta

İlk rollout'tan önce **safety kontrol listesi** (ders 15):

- Güç anahtarı ya da fiş el altında: yazılıma bağlı olmayan bir e-stop.
- Varsayılan `disable_torque_on_disconnect = True`: bir çökme ya da kablo çıkması kolu düşürür. Yumuşak bir yüzeyin üstünde, alçakta başla.
- `max_relative_target` (varsayılan `None`) follower config'inin bir alanı, hedeflerle aynı birimde (varsayılan olarak derece). İlk rollout'larda ayarla; komut satırından verilmesini test etmedik.
- Bağlanırken LeRobot yalnızca gripper için %50'lik bir tork limiti yazıyor ([`so_follower.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/so_follower.py)); beş kol joint'i için yazmıyor.
- Boş çalışma alanı, eller dışarıda, önce kısa `--duration`, kameralar tam kayıttaki gibi.

Her oturum için **ne log'lanmalı**: LeRobot sürümü, iki kalibrasyon dosyası (`<id>.json`), dataset repo id'si, episode sayısı ve fps, kamera konfigürasyonu, policy checkpoint'i (push edilen checkpoint'ler step'e göre etiketleniyor, örn. `--policy.pretrained_revision=010000`) ve her deneme için: sonuç, hata türü, bir video. Bus ve kamera okuma süreleri debug log'unda görünüyor (ders 37).

**Gerçek kolda henüz ölçülmedi:** bu dersin tablolarındaki her sayı Menagerie modelinden geliyor.

## Kendini dene

1. **Kolay.** Kalibrasyonu hangi komut yazıyor ve nerede saklanacağını hangi argüman belirliyor?
2. **Orta.** ×2'de step kontrolü başarısız ama takip 1,00°. Daha katı kontrolü yine de tutmak için iki sebep say.
3. **Zor.** `preflight()`'ı zeminin üstünde minimum bir gripper frame yüksekliğiyle genişlet. 2 cm'lik açıklık ilk olarak hangi shoulder_lift ofsetinde başarısız olur? Tahmin et, sonra çalıştır.

### Cevaplar

1. `lerobot-calibrate`; `id` (`--robot.id` / `--teleop.id`) dosyaya `<id>.json` adını verir.
2. Model servo'su gerçek servo değil (gain'leri 1-to-1 bir eşleme değil, ders 38) ve büyük bir hedef sıçraması, yanlış bir hedef nereyi gösteriyorsa kolun oraya servo'nun hız limitinde gitmesi demek (ders 15).
3. 6°'de: gripper frame'in en düşük yüksekliği 1,75 cm'ye iniyor. Ana preflight'ın geçirdiği 5°'lik ofset bile yalnızca 2,05 cm bırakıyor (0°: 3,57 cm). `output.txt`'de "Exercise 3" bölümü.

## Akılda kalanlar ve kaynaklar

- Deploy değişmezleri olan bir pipeline'dır: teleoperation'dan rollout'a kadar aynı port, `id`, kalibrasyon, kameralar ve fps.
- Ucuz bir preflight güvensiz komutları yakalar: ×2'den itibaren 0,10 rad/tick'in üstündeki step'ler, 0,6 kg'dan itibaren limitin %80'ini geçen statik tork.
- Joint aralığı kontrolleri bir kalibrasyon ofsetini yakalamaz; simülasyonda 10°'lik bir omuz ofseti gripper'ı zemine sokuyor.

Sonraki: [ders 41 · Vision-language-action model'leri](../part-10-foundation-models-and-agents/41-vla.md).

Kaynaklar:
- LeRobot dokümanları, 2026-10-27'de okundu: [SO-101](https://huggingface.co/docs/lerobot/so101), [gerçek robotlarda imitation learning](https://huggingface.co/docs/lerobot/il_robots), [ACT](https://huggingface.co/docs/lerobot/act), [hesaplama donanımı rehberi](https://huggingface.co/docs/lerobot/hardware_guide).
- LeRobot kodu (Apache-2.0): [`so_follower.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/so_follower.py), [`config_so_follower.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/config_so_follower.py), [`motors_bus.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/motors/motors_bus.py).
- [SO-ARM100 deposu](https://github.com/TheRobotStudio/SO-ARM100) (malzeme listesi, 3D baskı dosyaları).
- [MuJoCo Menagerie · robotstudio_so101](https://github.com/google-deepmind/mujoco_menagerie/tree/f3b6013/robotstudio_so101) (Apache-2.0).
- Ölçülen bütün sayılar: [`experiments/40-real-so101/output.txt`](../../experiments/40-real-so101/output.txt).
