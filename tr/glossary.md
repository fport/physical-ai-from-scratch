# Sözlük

Teknik terimler orijinal adlarıyla kullanılır. Her terim onu öğreten derse bağlanır.

| Terim | Kısaca | Ders |
|---|---|---|
| ACT (Action Chunking with Transformers) | CVAE ile eğitilmiş bir transformer'la gelecekteki action'lardan bir chunk tahmin eden imitation policy | [31](part-7-learning-from-demonstration/31-act.md) |
| Action chunking | Birkaç gelecek action'ı birden tahmin edip yeniden sormadan önce uygulamak | [31](part-7-learning-from-demonstration/31-act.md) |
| Antipodal grasp | Normal'leri birbirine bakan, iki friction cone'un da içinde kalan iki contact | [16](part-4-dynamics-and-control/16-grasping-and-pick-place.md) |
| Armature | Joint'e eklenen yansıyan rotor ataleti; sert position control'ü stabilize eder | [05](part-1-simulation/05-joints-and-actuators.md) |
| Behavior cloning (BC) | Demonstration'lardan observation → action eşlemesini supervised learning ile öğrenmek | [28](part-7-learning-from-demonstration/28-behavior-cloning.md) |
| `body` / `geom` | MJCF'te hareket eden frame / ona bağlı collision ve kütle şekli | [01](part-1-simulation/01-world-as-text.md) |
| Collocation | State ve control'lerin dynamics constraint'leriyle bağlı karar değişkenleri olduğu trajectory optimization | [23](part-6-planning-and-optimal-control/23-trajectory-optimization.md) |
| Configuration space (C-space) | Bütün joint konfigürasyonlarının uzayı; engeller burada yasak bölgelere dönüşür | [22](part-6-planning-and-optimal-control/22-motion-planning.md) |
| Contact normal | Engine'in iki cismi ayırmak için kullandığı yön | [02](part-1-simulation/02-timestep.md) |
| Creep | Soft contact yüzünden kayma eşiğinin altında görülen yavaş, sabit kayma | [04](part-1-simulation/04-friction.md) |
| DAgger | Dataset aggregation: öğrenenin gittiği state'leri bir expert'e etiketlet, yeniden eğit | [29](part-7-learning-from-demonstration/29-dagger.md) |
| Damped least squares (DLS) | Singularity yakınında sınırlı kalan IK güncellemesi $(J^\top J + \lambda^2 I)^{-1} J^\top e$ | [10](part-3-kinematics/10-inverse-kinematics.md) |
| Decimation | İki policy kararı arasındaki physics step sayısı | [34](part-8-rl-for-robots/34-observation-action-design.md) |
| Domain randomization | Policy gerçek sistemi de kapsasın diye rastgele physics'ler üzerinde eğitmek | [39](part-9-sim-to-real/39-domain-randomization.md) |
| Flow matching | Noise'u veriye taşıyan bir velocity field öğrenen generative model; action üretmek için kullanılır | [30](part-7-learning-from-demonstration/30-diffusion-and-flow-policies.md) |
| Forward kinematics (FK) | Eklem açıları → robottaki bir frame'in pose'u | [08](part-3-kinematics/08-forward-kinematics.md) |
| Friction cone | Sürtünmenin kaydırmadan taşıyabileceği contact kuvvetlerinin kümesi | [04](part-1-simulation/04-friction.md) |
| Gimbal lock | Euler açısı parametrizasyonunda bir dönme serbestliğinin kaybolması | [06](part-2-spatial-math/06-rotations.md) |
| Gravity feedforward | Sarkmayı gidermek için modelin yerçekimi torkunu controller çıkışına eklemek | [12](part-4-dynamics-and-control/12-joint-control.md) |
| Homogeneous transform | Rotation ve translation'ı birleştiren 4×4 matrix; frame değişimlerini zincirler | [07](part-2-spatial-math/07-rigid-transforms.md) |
| Homography | Bir görüntü ile bir düzlem arasındaki 3×3 projektif eşleme | [20](part-5-perception-and-estimation/20-pixels-to-table.md) |
| Impedance control | End effector'ın bir hedef etrafında yay–sönümleyici gibi davranmasını sağlamak | [14](part-4-dynamics-and-control/14-impedance-control.md) |
| Intrinsics | Kamera matrix'i $K$: focal length'ler ve principal point | [19](part-5-perception-and-estimation/19-camera-calibration.md) |
| Inverse kinematics (IK) | İstenen pose'a ulaşan eklem açılarını bulmak | [10](part-3-kinematics/10-inverse-kinematics.md) |
| Jacobian | Eklem hızlarını end effector hızına eşleyen matrix | [09](part-3-kinematics/09-jacobians.md) |
| Kalman filter | Linear-Gaussian sistemler için recursive optimal state estimator | [18](part-5-perception-and-estimation/18-kalman-filter.md) |
| Kp / Kd | Bir PD controller'ın proportional ve derivative gain'leri | [12](part-4-dynamics-and-control/12-joint-control.md) |
| LQR | Linear dynamics ve quadratic cost için optimal linear state feedback | [24](part-6-planning-and-optimal-control/24-lqr-and-mpc.md) |
| Manipulability | Bir kolun bir konfigürasyonda her yöne ne kadar iyi hareket edebildiğinin ölçüsü | [10](part-3-kinematics/10-inverse-kinematics.md) |
| MJCF | MuJoCo'nun dünyayı tarif eden XML biçimi | [01](part-1-simulation/01-world-as-text.md) |
| MPC | Her adımda sonlu horizon'lu bir optimal control problemini yeniden çözmek | [24](part-6-planning-and-optimal-control/24-lqr-and-mpc.md) |
| Operational-space control | End effector ataleti $\Lambda = (J M^{-1} J^\top)^{-1}$ ile task-space control | [14](part-4-dynamics-and-control/14-impedance-control.md) |
| Overshoot | Cevabın son değerinin ne kadar ötesine geçtiği | [12](part-4-dynamics-and-control/12-joint-control.md) |
| PPO | Clipped surrogate objective kullanan policy-gradient algoritması | [33](part-8-rl-for-robots/33-ppo.md) |
| Quaternion | Bir rotation'ı temsil eden birim 4-vektör; MuJoCo w-first saklar | [06](part-2-spatial-math/06-rotations.md) |
| Replay | Kaydedilmiş action'ları dünyaya bakmadan yeniden göndermek (open loop) | [26](part-7-learning-from-demonstration/26-teleop-replay-data.md) |
| Reward shaping | Öğrenmeyi yönlendirmek için reward'a terim eklemek; potential-based shaping optimumu korur | [35](part-8-rl-for-robots/35-reward-design.md) |
| RRT / RRT-Connect | C-space'te rastgele ağaçlar büyüten sampling-based motion planner'lar | [22](part-6-planning-and-optimal-control/22-motion-planning.md) |
| Semi-implicit Euler | Önce hızı, sonra konumu yeni hızla güncelleyen integrator | [02](part-1-simulation/02-timestep.md) |
| Singularity | Jacobian'ın rank kaybettiği konfigürasyon | [10](part-3-kinematics/10-inverse-kinematics.md) |
| `solref` / `solimp` | MuJoCo contact stiffness/damping ve impedance parametreleri | [03](part-1-simulation/03-contact.md) |
| System identification | Model parametrelerini ölçülmüş veriye fit etmek | [38](part-9-sim-to-real/38-system-identification.md) |
| Temporal ensembling | Üst üste binen action chunk'larını üstel ağırlıklarla ortalamak | [31](part-7-learning-from-demonstration/31-act.md) |
| `timestep` (dt) | Physics engine'in bir adımda zamanı ne kadar ilerlettiği | [02](part-1-simulation/02-timestep.md) |
| Tunneling | Hızlı bir cismin iki adım arasında ince bir engelin öbür yanına geçmesi | [02](part-1-simulation/02-timestep.md) |
| URDF | ROS'un robot tarif biçimi | [01](part-1-simulation/01-world-as-text.md) |
| VLA | Vision-language-action model: görüntü + dil + state → action | [41](part-10-foundation-models-and-agents/41-vla.md) |
| Watchdog | Komutlar gelmeyi kesince robotu durduran zamanlayıcı | [15](part-4-dynamics-and-control/15-safety.md) |
| Wilson interval | n denemede k başarıdan success rate için confidence interval | [25](part-7-learning-from-demonstration/25-evaluating-policies.md) |
| World model | State ve action'dan sonraki state'i tahmin eden öğrenilmiş model | [43](part-10-foundation-models-and-agents/43-world-models.md) |
