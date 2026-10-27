# Table of contents

* [Giriş](README.md)

## Başlarken

* [Kurulum: bir Mac, 0 TL](getting-started/setup.md)
* [Bir ders nasıl okunur](getting-started/how-to-read-a-lesson.md)
* [Sayı ve kaynak kuralları](getting-started/numbers-and-sources.md)
* [Okuma sırası](getting-started/reading-order.md)

## Faz 0 · Dünya kare kare

* [Giriş](phase-0-world-frame-by-frame/README.md)
  * [01 · Dünya bir metin dosyası](phase-0-world-frame-by-frame/01-world-as-text.md)
  * [02 · Timestep: küp masanın içinden neden geçti?](phase-0-world-frame-by-frame/02-timestep.md)
  * [03 · Simülasyonda masa aslında bir sünger: soft contact](phase-0-world-frame-by-frame/03-soft-contact.md)
  * [04 · Friction: ağır küp daha geç kayar mı?](phase-0-world-frame-by-frame/04-friction.md)

## Faz 1 · Uzay ve beden

* [Giriş](phase-1-space-and-body/README.md)
  * [05 · Robot için "sol" diye bir şey yok: coordinate frames](phase-1-space-and-body/05-coordinate-frames.md)
  * [06 · 6 motor var, 6 degrees of freedom yok](phase-1-space-and-body/06-degrees-of-freedom.md)
  * [07 · İki açı, bir nokta: forward kinematics](phase-1-space-and-body/07-forward-kinematics.md)
  * [08 · Bu kol kendi tabanına dokunabilir mi? Workspace](phase-1-space-and-body/08-workspace.md)
  * [09 · Aynı noktaya iki farklı dirsekle: inverse kinematics](phase-1-space-and-body/09-inverse-kinematics.md)

## Faz 2 · Kas

* [Giriş](phase-2-muscle/README.md)
  * [10 · Motora kuvvet değil, açı söylüyorsun: position servo](phase-2-muscle/10-position-servo.md)
  * [11 · 12 eklem, iki sayı: 20 ve 0,5](phase-2-muscle/11-pd-control.md)
  * [12 · Kol zayıf değil, matematik onu sarkıtıyor: gravity compensation](phase-2-muscle/12-gravity.md)
  * [13 · Eklemleri düz oynattım, uç eğri çizdi: trajectory](phase-2-muscle/13-trajectory.md)
  * [14 · Robotun ilk kuralı: kendini durdurabilmek](phase-2-muscle/14-safety.md)

## Faz 3 · Duyular

* [Giriş](phase-3-senses/README.md)
  * [15 · Noise'u temizledim, robot geç kaldı: filtering ve latency](phase-3-senses/15-noise-and-latency.md)
  * [16 · Kameran düz çizgiyi eğri görüyor: camera calibration](phase-3-senses/16-camera-calibration.md)
  * [17 · Ekrana tıkladım, kol oraya dokundu: homography](phase-3-senses/17-pixel-to-table.md)
  * [18 · Lambayı açtım, robot kırmızıyı göremedi: color thresholding ve open-vocabulary detection](phase-3-senses/18-color-vs-open-vocabulary.md)

## Faz 4 · Göstererek öğretmek

* [Giriş](phase-4-learning-from-demonstration/README.md)
  * [19 · Replay öğrenme değildir](phase-4-learning-from-demonstration/19-replay.md)
  * [20 · Bir robot dataset'i neye benzer?](phase-4-learning-from-demonstration/20-datasets.md)
  * [21 · Kimse kod yazmadı: behavior cloning](phase-4-learning-from-demonstration/21-behavior-cloning.md)
  * [22 · %90 success rate dedim, aslında ne kadar? Evaluation](phase-4-learning-from-demonstration/22-evaluation.md)
  * [23 · Mükemmel taklitçi ilk hatada batar mı? Distribution shift](phase-4-learning-from-demonstration/23-distribution-shift.md)
  * [24 · İki doğru cevabın ortalaması yanlıştır](phase-4-learning-from-demonstration/24-two-right-answers.md)
  * [25 · Robot her frame'de fikir değiştiriyordu: action chunking](phase-4-learning-from-demonstration/25-action-chunking.md)
  * [26 · Gerçek kolda end-to-end imitation learning](phase-4-learning-from-demonstration/26-end-to-end-imitation.md)

## Faz 5 · Ajan

* [Giriş](phase-5-agents/README.md)
  * [27 · Agent loop'u 25 satırda yazmak](phase-5-agents/27-agent-loop.md)
  * [28 · Büyük model kolu oynatmıyor, emir veriyor](phase-5-agents/28-two-brains.md)
  * [29 · Agent'a "masanın altına uzan" dedim: tool call validation](phase-5-agents/29-agent-safety.md)
  * [30 · Aynı kolu üç farklı agent kullansın: MCP](phase-5-agents/30-mcp.md)
  * [31 · Robota konuşarak iş yaptırmak: Strands](phase-5-agents/31-strands-agent.md)

## Faz 6 · Deneyerek öğrenmek

* [Giriş](phase-6-learning-by-trial/README.md)
  * [32 · Yol göstermedim, sadece reward verdim: Q-learning](phase-6-learning-by-trial/32-q-learning.md)
  * [33 · Binlerce robot aynı anda düşerse: PPO sezgisi](phase-6-learning-by-trial/33-parallel-rl.md)
  * [34 · Agent reward'u kandırırsa: reward hacking](phase-6-learning-by-trial/34-reward-hacking.md)

## Faz 7 · Sim'den masaya

* [Giriş](phase-7-sim-to-real/README.md)
  * [35 · Robotun beyni kaç milisaniye geride? Latency ve control rate](phase-7-sim-to-real/35-latency-and-control-rate.md)
  * [36 · Simdeki motor gerçeğinden farklıydı: system identification](phase-7-sim-to-real/36-system-identification.md)
  * [37 · Sim'i bilerek bozmak: domain randomization](phase-7-sim-to-real/37-domain-randomization.md)

## Faz 8 · Büyük beyinler

* [Giriş](phase-8-big-models/README.md)
  * [38 · Bir VLA'nın içine bakmak: image + text → action](phase-8-big-models/38-vla.md)
  * [39 · Noise'dan yol çıkarmak: diffusion ve flow matching](phase-8-big-models/39-diffusion-and-flow.md)

## Faz 9 · Bitirme

* [Giriş](phase-9-capstone/README.md)
  * [40 · "Mavi küpü kutuya koy"](phase-9-capstone/40-capstone.md)

## Ekler

* [Ek dersler](appendix/README.md)
  * [EK A · Singularity: kol dümdüz uzanınca neden çıldırır?](appendix/a-singularities.md)
  * [EK B · Gimbal lock ve quaternion](appendix/b-gimbal-lock.md)
  * [EK C · Kalman sezgisi: iki noisy ölçümü birleştirmek](appendix/c-kalman.md)
  * [EK D · SAM ile grasp point](appendix/d-sam-grasp-points.md)
  * [EK E · SmolVLA'yı kendi verinde fine-tune etmek](appendix/e-smolvla-finetuning.md)
  * [EK F · World model: "sonra ne olur"u öğrenen küçük model](appendix/f-world-models.md)
  * [EK G · π0, GR00T, Gemini Robotics: aynı fikrin üç ölçeği](appendix/g-big-robot-models.md)
  * [EK H · Mac'te kaç Hz? CPU ve MPS'de inference hızı](appendix/h-inference-speed-on-mac.md)

## Başvuru

* [Sözlük](glossary.md)
* [Kaynakça](references.md)
* [Notebook'lar](notebooks/README.md)
* [Katkı](contributing.md)
