# Sözlük

Teknik terimler orijinal adlarıyla kullanılır. Bu sözlük her terimin kitapta ilk geçtiği derse bağlanır. Dersler yazıldıkça genişler.

| Terim | Kısaca | İlk geçtiği ders |
|---|---|---|
| ACT (Action Chunking with Transformers) | Bir sonraki tek action yerine bir action chunk tahmin eden imitation policy | [24](phase-4-learning-from-demonstration/24-two-right-answers.md), [25](phase-4-learning-from-demonstration/25-action-chunking.md) |
| Open-loop / closed-loop | Çıktıyı ölçmeden komut uygulamak / ölçtüğüne göre komutu değiştirmek | [19](phase-4-learning-from-demonstration/19-replay.md) |
| Overshoot | Hedefin (ya da oturulan değerin) ne kadar ötesine geçildiği | [11](phase-2-muscle/11-pd-control.md) |
| `body` / `geom` | MJCF'te hareket eden çerçeve / ona bağlı çarpışma ve kütle şekli | [01](phase-0-world-frame-by-frame/01-world-as-text.md) |
| Behavior cloning (BC) | Demonstration'lardan state → action mapping'ini supervised learning ile öğrenmek | [21](phase-4-learning-from-demonstration/21-behavior-cloning.md) |
| Decimation | RL training'inde policy'nin kaç physics step'te bir karar verdiği | [02](phase-0-world-frame-by-frame/02-timestep.md) |
| Domain randomization | Sim parametrelerini training'de rastgele değiştirerek policy'yi robust yapmak | [04](phase-0-world-frame-by-frame/04-friction.md), [37](phase-7-sim-to-real/37-domain-randomization.md) |
| `freejoint` | Cismi 6 degrees of freedom ile serbest bırakan eklem | [01](phase-0-world-frame-by-frame/01-world-as-text.md) |
| IK (inverse kinematics) | İstenen uç konumuna götüren eklem açılarını bulmak | [09](phase-1-space-and-body/09-inverse-kinematics.md) |
| FK (forward kinematics) | Eklem açılarından uç konumunu hesaplamak | [07](phase-1-space-and-body/07-forward-kinematics.md) |
| Kp / Kd | PD control'de konum farkını ve hızı torka çeviren katsayılar | [11](phase-2-muscle/11-pd-control.md) |
| MDN (mixture density network) | Çıkışı birden çok olası cevap ve ağırlıkları olan ağ | [24](phase-4-learning-from-demonstration/24-two-right-answers.md) |
| MJCF | MuJoCo'nun dünyayı tarif eden XML biçimi | [01](phase-0-world-frame-by-frame/01-world-as-text.md) |
| μ (friction coefficient) | Sürtünme tavanının bastırma kuvvetine oranı; kayma açısının tanjantı | [04](phase-0-world-frame-by-frame/04-friction.md) |
| Replay | Kaydedilmiş action'ları değiştirmeden tekrar göndermek | [19](phase-4-learning-from-demonstration/19-replay.md) |
| Creep | Soft contact yüzünden eşiğin altında görülen yavaş, sabit kayma | [04](phase-0-world-frame-by-frame/04-friction.md) |
| Friction cone | Contact kuvvetinin sürtünmeyle karşılanabilecek yönleri | [04](phase-0-world-frame-by-frame/04-friction.md) |
| Contact normal | Engine'in iki cismi ayırmak için seçtiği yön | [02](phase-0-world-frame-by-frame/02-timestep.md) |
| `timestep` (dt) | Physics engine'in bir adımda zamanı ne kadar ileri aldığı | [02](phase-0-world-frame-by-frame/02-timestep.md) |
| Tool calling | LLM'in bir fonksiyonu adı ve argümanlarıyla çağırması | [31](phase-5-agents/31-strands-agent.md) |
| Tunneling | Hızlı bir cismin tek adımda ince bir engelin öbür yanına geçmesi | [02](phase-0-world-frame-by-frame/02-timestep.md) |
| URDF | ROS dünyasının robot tarif biçimi | [01](phase-0-world-frame-by-frame/01-world-as-text.md) |
| VLA (vision-language-action) | Image ve cümleden action üreten model | [38](phase-8-big-models/38-vla.md) |
| Semi-implicit Euler | Önce hızı, sonra konumu yeni hızla güncelleyen integrator | [02](phase-0-world-frame-by-frame/02-timestep.md) |
