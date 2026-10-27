# 35 · Reward tasarımı ve reward hacking

<!-- meta -->
| | |
|---|---|
| **Bölüm** | [8 · Robotlar için reinforcement learning](README.md) |
| **Çalışma türü** | CPU · Mac ya da Colab CPU |
| **Önkoşul** | [33](33-ppo.md) |
| **Önce öğren** | [MDPs, States, Actions & Rewards](https://aiengineeringfromscratch.com/lesson?path=phases/09-reinforcement-learning/01-mdps-states-actions-rewards) |
| **Süre** | ~60 dk |
| **Kod** | [`experiments/35-reward-design/experiment.py`](../../experiments/35-reward-design/experiment.py) |
| **Durum** | ✅ tam |
<!-- /meta -->

> Agent kastettiğiniz görevi değil, yazdığınız reward'u optimize eder. Her terim ve bir episode'un bitebileceği her yol, yazdığınızın parçasıdır.

## Problem

Go2 training config'i ağırlığı sıfır olmayan on reward terimini toplar; `tracking_lin_vel = 1.0`'dan `action_rate = -0.01` ve `torques = -0.0002`'ye kadar. Ve yalnızca bir agent'ın reward'ları nasıl istismar ettiği yüzünden var olan bir satır taşır: `only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)` ([`legged_robot_config.py`](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/base/legged_robot_config.py), [`go2_config.py`](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/go2/go2_config.py)). Bu ders SO-101 uzanmasında, sabit bir PPO ve bütçeyle farklı reward şekillerinin ne yaptığını ölçüyor, bilerek bir reward hack'i üretiyor ve bir yumuşaklık cezasını sweep ediyor.

## Ana fikir

MDP reward'unun kendisi AIEFS'te var ([MDPs, States, Actions & Rewards](https://aiengineeringfromscratch.com/lesson?path=phases/09-reinforcement-learning/01-mdps-states-actions-rewards)). Robot reward'ları dört tür terimden kurulur:

| Terim | Örnek (uzanma, $d$ metre cinsinden mesafe) | Rolü |
|---|---|---|
| sparse görev | $\mathbb 1[d < 0{,}02]$ | başarının tam olarak ne olduğunu söyler; başarı bulunana kadar hiç gradient vermez |
| dense görev | $-d$ | her adımda hedefi gösterir; "iyi"yi bir vekille tanımlar |
| potential-based shaping | $F = \gamma\Phi(s') - \Phi(s)$ | optimal policy'yi değiştirmeden yönlendirme ekler |
| düzenleyiciler | $-w\lVert a_t - a_{t-1}\rVert^2$, $-w\lVert\tau\rVert^2$ | görev performansını yumuşaklık ve enerjiyle takas eder |

**Potential-based shaping.** Ng, Harada ve Russell, herhangi bir $\Phi$ potansiyeli için $F(s,s') = \gamma\Phi(s') - \Phi(s)$ eklemenin optimal policy'yi değiştirmediğini ve MDP hakkında başka bilgi yoksa bu biçimin o garanti için gerekli de olduğunu gösterdi ([ICML 1999, s. 278–287](https://mlanthology.org/icml/1999/ng1999icml-policy/)). Bir trajectory boyunca shaping terimleri teleskop gibi sadeleşir, $\sum_t \gamma^t F_t = \gamma^T \Phi(s_T) - \Phi(s_0)$; yani onlardan reward toplanamaz. PPO discount'u $\gamma = 0{,}95$ ile $\Phi = -20\,d$ kullanıyoruz.

**Termination, işaretin anlamını değiştirir.** Terminal bir durumdan sonra gelecekteki her reward 0'dır. Adım başına reward'lar negatifse episode'u bitirmek caziptir; pozitifse bitirmek bir kayıptır. Agent, yazılan toplam için hangisi daha iyiyse oraya yönelir. Locomotion'da düşmek episode'u bitirir; bu yüzden çoğu negatif ceza terimlerinden oluşan bir toplam erken düşmeyi cazip kılar; Go2 config'i toplamı sıfırda keser ve sebebini "early termination problems" diye adlandırır. Ters durum aşağıda kurduğumuz hack'tir.

## Deney

[`experiment.py`](../../experiments/35-reward-design/experiment.py) ders 33'teki her şeyi sabit tutar (mutlak joint hedefleri, 50 Hz, PPO referans config'i, 300k adım, 3 seed × 100 evaluation episode'u) ve env'e verilen reward fonksiyonunu değiştirir:

```python
def r_sparse(info):
    return float(info["dist"] < 0.02)

def r_pbrs(info):                       # sparse + gamma * Phi(s') - Phi(s), Phi = -20 d
    return r_sparse(info) + 0.95 * (-20 * info["dist"]) - (-20 * info["dist_prev"])

def r_closeness(info):                  # positive every step, 1 at the target
    return 1.0 - np.tanh(info["dist"] / 0.05)

def r_action_rate(w):
    return lambda info: -info["dist"] - w * np.sum((info["action"] - info["prev_action"]) ** 2)
```

**Tahmin edin:** closeness reward'u ve gripper 2 cm'ye girdiği anda biten bir episode ile eğitilen kol ne yapacak?

MuJoCo 3.17.0, NumPy 2.5.4, PyTorch 2.14.1 (CPU), Apple Silicon Mac (Apple M5). Seed başına deterministik. Script'in tamamı (33 run, altısı paralel) yoğun yüklü bir makinede 1904 s (yaklaşık 32 dakika) sürdü. `python experiment.py --only close+end,pbrs` yalnızca o hücreleri üç seed'le eğitir; bu tekrar çalıştırma aşağıdaki tablolardaki satırlarını aynen üretti. `--quick` yalnızca seed 0'ı eğitir: her satırın ilk "seed başına başarı" girdisini yeniden üretir, toplam sütunlarını değil.

**A. Reward şekli** (termination yok; başarı = 1,6 s'lik zaman sınırında 2 cm içinde):

| reward | seed başına başarı | toplam | Wilson %95 | medyan son mesafe |
|---|---|---|---|---|
| dense $-d$ | 100 / 98 / 100 | 298/300 | %97,6–99,8 | 7,5 mm |
| sparse $\mathbb 1[d < 2\,\text{cm}]$ | 8 / 0 / 0 | 8/300 | %1,4–5,2 | 117,8 mm |
| sparse + $0{,}95\,\Phi(s') - \Phi(s)$, $\Phi = -20d$ | 99 / 97 / 99 | 295/300 | %96,2–99,3 | 7,8 mm |

(Medyan son mesafe: üç seed üzerinden ortalama.) Sparse reward görevin dürüst tanımıdır ve başarısız olur: başlangıç policy'si şans eseri nadiren 2 cm'ye yaklaşır (ders 32: rastgele action'lı 200 episode'un 1'i orada bitti), bu yüzden training'in çoğunda her reward sıfırdır ve advantage'lar bilgi taşımaz. Potential-based bir terim eklemek öğrenmeyi dense reward düzeyine geri getirir (iki aralık örtüşüyor); Ng vd.'nin teoremine göre de sparse görev için hangi policy'nin optimal olduğunu değiştirmez.

**B. Reward hacking.** Episode artık $d < 2$ cm olduğu anda biter (`terminated`); "ulaştı"yı tanımlamanın doğal bir yolu gibi görünür. Reward pozitif closeness terimidir, $1 - \tanh(d / 5\,\text{cm})$:

| reward, termination | seed başına başarı | toplam | Wilson %95 | yazılan return | episode uzunluğu | medyan son mesafe |
|---|---|---|---|---|---|---|
| closeness, başarıda biter | 23 / 43 / 29 | 95/300 | %26,7–37,1 | **22,67** | 62,3 adım | 28,8 mm |
| $-d$, başarıda biter | 99 / 100 / 98 | 297/300 | %97,1–99,7 | −0,72 | 9,2 adım | 14,7 mm |
| closeness, termination yok | 98 / 100 / 100 | 298/300 | %97,6–99,8 | 64,00 | 80,0 adım | 5,1 mm |

İki kontrol ölçümü hack'i açık hâle getirir:

- Her hedefe ulaşan scripted IK controller, closeness-artı-termination reward'u altında yalnızca **2,28** yazılan return alır. Hack yapan policy 22,67 alır: kendisine verilen amaçta, görevi daha az yaparak doğru controller'ı yener.
- Termination olmadan çalıştırıldığında aynı hack yapan policy hedeften medyan **30,8 mm** uzakta bitirir: gripper'ı yaklaşık 3 cm uzağa, başarı küresinin hemen dışına park eder ve episode'un geri kalanında adım başına kabaca 0,45 toplar.

Mekanizma Ana fikir'dekiyle aynı: küreye girmek episode'u ve onunla birlikte gelecekteki tüm pozitif reward'u bitirir; agent yaklaşıp kısa durmayı öğrenir. 95 başarı, yine de kürenin içine düştüğü episode'lardır. Aynı termination kuralı negatif bir reward'la ($-d$) ters teşvik verir: her adım maliyetli olduğu için agent içeri acele eder (ortalama 9,2 adım). Termination olmadan pozitif reward da çalışır (298/300). Hack kombinasyonda yaşar; bu yüzden kazara yazmak kolaydır.

## Boz

Yumuşaklık cezaları, takasın bir uçuruma dönüştüğü yerdir. Action-rate teriminin $w$ ağırlığını sweep edin (Go2 config'i aynı terimi kullanır, `action_rate = -0.01`, [`legged_robot.py`](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/base/legged_robot.py)'de $\sum (a_{t-1} - a_t)^2$ olarak hesaplanır; oradaki `_prepare_reward_function` her ağırlığı policy periyoduyla, 0,02 s ile çarpar, yani adım başına ağırlık $0{,}01 \times 0{,}02 = 0{,}0002$'dir, hız takibi terimi ise adım başına en fazla $1{,}0 \times 0{,}02$'dir):

| $w$ | seed başına başarı | toplam | Wilson %95 | medyan son mesafe (mm) | tork hızı (N·m/s) | güç (W) |
|---|---|---|---|---|---|---|
| 0 | 100 / 98 / 100 | 298/300 | %97,6–99,8 | 7,5 | 1002 | 9,66 |
| 0,03 | 81 / 99 / 99 | 279/300 | %89,5–95,4 | 8,8 | 933 | 9,44 |
| 0,1 | 80 / 87 / 96 | 263/300 | %83,5–90,9 | 11,4 | 901 | 9,45 |
| 0,3 | 41 / 9 / 94 | 144/300 | %42,4–53,6 | 25,6 | 850 | 8,20 |
| 1,0 | 7 / 35 / 14 | 56/300 | %14,7–23,5 | 39,2 | 674 | 6,54 |
| 3,0 | 27 / 6 / 26 | 59/300 | %15,6–24,5 | 31,9 | 743 | 5,64 |

$w = 0{,}1$'e kadar policy başarıdan 6 ila 12 yüzde puanı kaybeder (298/300'e karşı 279/300 ve 263/300) ve çok az yumuşaklık kazanır (tork hızı 1002 → 901 N·m/s). $w = 0{,}3$'ten itibaren başarı çöker ve seed'ler arasında büyük fark oluşur (100 üzerinden 9'a karşı 94): setpoint'i kaydırmanın cezası artık mesafe reward'uyla kıyaslanabilir düzeydedir ve sonuç seed'e bağlıdır. Güç düşer (9,66 → 5,64 W), büyük ölçüde kol daha az hareket ettiği için: medyan son mesafe 25,6–39,2 mm'ye çıkar. İki ders: bir düzenleyicinin ağırlığı görev teriminin ölçeğine göre ayarlanmalı (burada $-d$ adım başına birkaç santimetre) ve sert bir servoyla policy'nin hedeflerini yumuşatmak torku pek yumuşatmaz. Amaç yumuşak torksa, $w$'yi büyütmeden önce arayüzü ya da gain'leri değiştirin (ders 34).

## Gerçek robotta

- Ceza ağırlıkları simülasyonda görev başarısını daha nazik hareketle takas eder; ölçülen tork hızı azalmasının gerçeğe taşınıp taşınmadığı gerçek actuator'e bağlıdır. SO-101'de ölçülmedi.
- Training için yazılan bir termination kuralı donanımda aynı zamanda bir güvenlik kuralıdır: gerçek robotta "episode biter", bir denetleyicinin hareketi durdurması demektir ([ders 15](../part-4-dynamics-and-control/15-safety.md)). Reward, bu durmayı policy'nin kendi başına aradığı ya da kaçındığı bir şey hâline getirmemelidir.
- Go2 config'inin reward ağırlıkları onun simülatörü ve PD gain'leri için ayarlandı; ikisinden birini değiştirmek o ağırlıkların anlamını değiştirir.

## Kendini dene

1. **Kolay.** Closeness reward'unda hedeften 3 cm uzağa park etmiş bir gripper adım başına ne kadar reward toplar, başarı küresine girdikten sonra ne toplar?
2. **Orta.** Termination'ı kaldırmadan hack'i düzeltin: episode başarıyla bittiğinde bir terminal bonus $B$ ekleyin. Ulaşmayı, kalan ~60 adım boyunca havada beklemek kadar iyi yapan $B$'yi tahmin edin, sonra üç seed eğitip ölçün.
3. **Zor.** Action-rate cezasını env içinde fizik hızında hesaplanan bir tork hızı cezasıyla değiştirin ve sweep'i tekrarlayın. Kaybedilen başarı başına action-rate teriminden daha fazla yumuşaklık sağlıyor mu?

### Cevaplar

1. Adım başına $1 - \tanh(0{,}03/0{,}05) = 1 - \tanh(0{,}6) \approx 0{,}46$; başarıdan sonra episode biter, sonraki her reward 0'dır.
2. Adım başına ~0,46 ile $n$ adım daha beklemek yaklaşık $0{,}46 \sum_{t<n} 0{,}95^t$ eder; uzun kalan sürelerde bu $0{,}46/(1-0{,}95) \approx 9{,}2$'ye yaklaşır; bunun belirgin biçimde üstünde bir bonus teşviki ortadan kaldırır. Training sonucunu siz ölçeceksiniz; biz çalıştırmadık.
3. Açık; çalıştırmadık.

## Akılda kalanlar ve kaynaklar

- Sparse reward'lar görevi tam olarak tanımlar ama bütçe içinde 8/300 verdi; potential-based shaping optimal policy'yi değiştirmeden 295/300'ü geri getirdi.
- Termination artı pozitif adım reward'u ölçülmüş bir reward hack'idir: 95/300 başarı, ~3 cm uzağa park edilmiş gripper ve her hedefe ulaşan bir controller'ın (2,28) yaklaşık on katı yazılan return (22,67). Bir episode'un bitebileceği her yerde reward'un işaretini kontrol edin.
- Düzenleyici ağırlıklarında uçurum var: action-rate ağırlığı 0,1 12 yüzde puanına mal oldu (263/300), 0,3 başarıyı yarıya indirdi; ağırlıkları görev terimine göre ayarlayın ve gerçekte elde ettiğiniz yumuşaklığı ölçün.

Sıradaki: [ders 36](36-legged-locomotion.md), legged locomotion ve massively parallel training.

**Kaynaklar**

- A. Y. Ng, D. Harada, S. Russell, "Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping", ICML 1999, s. 278–287. [mlanthology](https://mlanthology.org/icml/1999/ng1999icml-policy/)
- [`unitree_rl_gym` · `legged_robot_config.py`](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/base/legged_robot_config.py), [`go2_config.py`](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/go2/go2_config.py), [`legged_robot.py`](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/base/legged_robot.py) (`_reward_action_rate`, `_reward_torques`) (BSD-3-Clause).
- J. Schulman vd., "Proximal Policy Optimization Algorithms", 2017. [arXiv 1707.06347](https://arxiv.org/abs/1707.06347)
- [Gymnasium · `Env` API](https://gymnasium.farama.org/api/env/) (terminated ve truncated).
- Kod: [`experiment.py`](../../experiments/35-reward-design/experiment.py), [`output.txt`](../../experiments/35-reward-design/output.txt); env ve PPO [ders 32](../../experiments/32-rl-environment/reach_env.py) ve [ders 33](../../experiments/33-ppo/ppo.py)'ten.
