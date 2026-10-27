# Physical AI stack'i

> Bir robot iç içe loop'lardan oluşan bir stack'tir: hızlı olanlar eklemi olması gereken yerde tutar, yavaş olanlar nerede olması gerektiğine karar verir.

Bir küpü alan ya da odada yürüyen bir robot aynı anda birkaç yazılım katmanı çalıştırır. Her katmanın kendi işi, kendi hızı ve kendi hata biçimleri vardır. Bu kitap onları aşağıdan yukarıya, ihtiyaç duyduğun sırayla öğretir.

| Katman | İşi | Tipik araçlar | Kitapta |
|---|---|---|---|
| Physics ve simulation | Cisimlerin nasıl hareket edip çarpıştığını tahmin etmek | MuJoCo, MJCF/URDF modelleri | [Bölüm 1](../part-1-simulation/README.md) |
| Geometri | Nesnelerin nerede olduğunu ve orientation'ını tarif etmek | Rotation matrix, quaternion, transform | [Bölüm 2](../part-2-spatial-math/README.md) |
| Kinematics | Eklem açılarını gripper pose'una ve geri çevirmek | FK, Jacobian, IK | [Bölüm 3](../part-3-kinematics/README.md) |
| Low-level control | Bir hedefi güvenli şekilde motor torklarına çevirmek | PD/PID, servo, impedance control, limit'ler | [Bölüm 4](../part-4-dynamics-and-control/README.md) |
| State estimation ve perception | Noisy sensor'leri ve pixel'leri bir state'e çevirmek | Filter, Kalman filter, camera calibration, detection | [Bölüm 5](../part-5-perception-and-estimation/README.md) |
| Planning ve optimal control | Uygulanabilir hareket bulmak, control'ü optimize etmek | RRT, trajectory optimization, LQR, MPC | [Bölüm 6](../part-6-planning-and-optimal-control/README.md) |
| Öğrenilmiş policy'ler | Veriden ya da reward'dan observation → action eşlemesi öğrenmek | Behavior cloning, ACT, diffusion policy, PPO | [Bölüm 7–8](../part-7-learning-from-demonstration/README.md) |
| Deployment | Simde eğitilen davranışı donanımda çalıştırmak | Latency, system identification, domain randomization, LeRobot | [Bölüm 9](../part-9-sim-to-real/README.md) |
| Görev katmanı | Dilden ve bağlamdan ne yapılacağına karar vermek | VLA, robot tool'ları olan LLM agent'ları, world model | [Bölüm 10](../part-10-foundation-models-and-agents/README.md) |

## Loop'lar farklı hızlarda çalışır

Alt katmanlar üst katmanlardan hızlı çalışır. Kitapta kullanılan iki somut örnek:

- Unitree'nin Go2 için açık training kodunda öğrenilmiş policy her 20 ms'de (50 Hz) yeni eklem hedefleri seçer, bir PD kuralı bunları her 5 ms'de torka çevirir ([`legged_robot_config.py`](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/base/legged_robot_config.py) içinde `sim.dt = 0.005`, `decimation = 4`). Ders [12](../part-4-dynamics-and-control/12-joint-control.md) bu sayıları kullanır.
- LeRobot, SO-101'de demonstration'ları saniyede yaklaşık 30 frame ile kaydeder ([LeRobot belgeleri](https://huggingface.co/docs/lerobot/il_robots)); servo'lar kendi position loop'larını motorun içinde kapatır (ders [12](../part-4-dynamics-and-control/12-joint-control.md)).

## ROS 2 nereye oturur

Gerçek robotlarda katmanlar genellikle mesajlaşan ayrı process'ler olarak çalışır; bunun için en yaygın middleware [ROS 2](https://docs.ros.org/)'dir (node, topic, service, action). Bu kitap algoritmalar görünür kalsın diye her deneyi MuJoCo ile tek bir Python process'inde tutar; aynı algoritmalar ROS 2 node'larının içine koyacağın şeylerdir. Bölüm 7 ve 9'da kullanılan LeRobot, SO-101 ile doğrudan seri bus üzerinden konuşur.

## Kitaptaki robotlar

| Robot | Ne | Model |
|---|---|---|
| SO-101 | LeRobot ile kullanılan düşük maliyetli 6 motorlu kol (5 eklem + gripper) | Menagerie `robotstudio_so101` |
| Franka Panda | Redundancy ya da torque control gereken yerlerde kullanılan 7-DoF araştırma kolu | Menagerie `franka_emika_panda` |
| Unitree Go2 | Control ve locomotion için kullanılan 12 eklemli dört ayaklı robot | Menagerie `unitree_go2` |

Üç model de [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)'den gelir ve `experiments/common/menagerie.py` tarafından indirilir.
