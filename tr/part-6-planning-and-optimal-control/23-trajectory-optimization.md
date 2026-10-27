# 23 · Trajectory optimization

<!-- meta -->
| | |
|---|---|
| **Bölüm** | [6 · Planning ve optimal control](README.md) |
| **Çalışma türü** | CPU · Mac ya da Colab CPU |
| **Önkoşul** | [13](../part-4-dynamics-and-control/13-trajectory-generation.md), [22](22-motion-planning.md) |
| **Önce öğren** | [Optimization](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/08-optimization) |
| **Süre** | ~60 dk |
| **Kod** | [`experiments/23-trajectory-optimization/experiment.py`](../../experiments/23-trajectory-optimization/experiment.py) |
| **Durum** | ✅ tam |
<!-- /meta -->

> Trajectory optimization "bir hareket bul"u "dinamiğe uyarak bir maliyeti minimize et"e çevirir. Dinamiği nasıl dayattığın, "yakınsadı"nın ne anlama geldiğini belirler.

## Problem

Bir sarkaç (1 m'de 1 kg, $mgl = 9{,}81$ N·m, joint damping 0,1 N·m·s) aşağı sarkıyor ve 4 s sonunda dik ve durgun olmalı. Motor en fazla 5 N·m veriyor, yatayda yerçekiminin istediğinin yaklaşık yarısı; yani sarkacı doğrudan kaldıramaz, enerji pompalamak zorunda. [Ders 13](../part-4-dynamics-and-control/13-trajectory-generation.md)'teki zaman polinomu ne dinamiği ne torque limitini bilir; [ders 22](22-motion-planning.md)'deki path ise yalnızca geometridir. İstediğimiz şey girdinin kendisi, $u(t)$: dinamik ve limit altında uygulanabilir ve olabildiğince ucuz.

## Ana fikir

Sürekli problem (kısıtlı optimization temelleri için bkz. AIEFS: [Optimization](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/08-optimization)):

$$\min_{u(\cdot)} \int_0^T u^2\,dt + w_s \int_0^T \dot u^2\,dt \quad \text{s.t.}\quad \dot x = f(x,u),\ \ \lvert u\rvert \le u_{max},\ \ x(0)=x_0,\ \ x(T)=x_{goal}.$$

Efor ve yumuşaklık maliyet terimleridir; hedef burada sert bir terminal kısıt olarak dayatılıyor (yumuşak alternatif terminal ceza $w_g\lVert x(T)-x_{goal}\rVert^2$). $u(t)$'yi $N+1$ knot'tan geçen parçalı doğrusal bir fonksiyon olarak ayrıklaştırıyoruz, aralık $h = T/N$, ve SciPy'ın SLSQP'siyle (sequential quadratic programming) çözüyoruz. $\dot x = f(x,u)$'yu dayatmanın iki standart yolu var ([Tedrake, böl. 10](https://underactuated.csail.mit.edu/trajopt.html)):

| | Direct shooting | Direct collocation |
|---|---|---|
| karar değişkenleri | knot girdileri $u_0..u_N$ | knot state'leri $x_0..x_N$ ve girdiler $u_0..u_N$ |
| dinamik | ileri doğru integre edilir (burada RK4, aralık başına 20 alt adım) | yalnızca knot'larda eşitlik "defect"leri |
| kısıtlar | $x_N(u) = x_{goal}$ (2 denklem) | $2N$ defect + sınır state'leri |
| kesin olan | simüle edilen trajectory | ayrıklaştırılmış denklemler |
| ilk tahmin | yalnızca girdiler | state'ler ve girdiler |

Bizim collocation'ımız trapez defect'ini kullanıyor:

$$x_{k+1} - x_k - \tfrac{h}{2}\bigl(f(x_k,u_k) + f(x_{k+1},u_{k+1})\bigr) = 0.$$

Tedrake'in bölümü kübik state'li, doğrusal girdili ve üçüncü dereceden doğru bir şema kullanır; trapez onun daha basit, ikinci dereceden hâlidir. Collocation'ın Jacobian'ı seyrek ve bantlıdır, her kısıt yalnızca komşu knot'ları bağlar; shooting'in değişkeni azdır ama her kısıt tüm rollout boyunca her girdiye bağlıdır ve erken girdiler terminal state'i geç girdilerden çok daha fazla oynatır (Tedrake'in "tail wagging the dog" dediği şey).

## Deney

**Tahmin.** Yeterli knot ile iki yöntem de yakınsamalı, yaklaşık aynı maliyete. "Yakınsadı" ikisi için aynı şeyi ifade etmeyebilir.

Tam script: [`experiments/23-trajectory-optimization/experiment.py`](../../experiments/23-trajectory-optimization/experiment.py). Collocation sürümünün özü:

```python
def defects(z):                                   # z = [x_0..x_N, u_0..u_N]
    X, U = unpack(z)
    F = f(X, U)
    d = X[1:] - X[:-1] - h / 2 * (F[1:] + F[:-1])  # trapezoidal rule
    return np.concatenate([d.ravel(), X[0] - X0, X[-1] - XG])

r = minimize(obj, z0, jac=obj_grad, method="SLSQP",
             bounds=[(None, None)] * (2 * n) + [(-UMAX, UMAX)] * n,
             constraints=[{"type": "eq", "fun": defects, "jac": defects_jac}])
```

Maliyet: $J = \int u^2 dt + 0{,}05 \int \dot u^2 dt$. Collocation analitik bir kısıt Jacobian'ı alıyor; shooting'in terminal kısıt Jacobian'ı toplu ileri farklarla hesaplanıyor (maliyet gradient'i ikisinde de analitik). İkisi de sıfır torque ve dipte duran state ile başlıyor. Çözümden sonra her $u(t)$ **bir MuJoCo sarkacında open-loop tekrar oynatılıyor** (aynı parametreler, RK4, 1 ms adım): replay hatası, gerçekte elde edeceğin hareketin hedef hatasıdır. MuJoCo 3.17.0, NumPy 2.5.4, SciPy 1.18.1, Apple Silicon Mac (Apple M5), deterministik.

| yöntem | N | yakınsadı | iterasyon | maliyet | solver residual | replay hatası θ (derece) | replay hatası ω (derece/s) |
|---|---|---|---|---|---|---|---|
| shooting | 5 | False | 438 | 104,44 | 3,9e-01 | 21,33 | 22,41 |
| collocation | 5 | False | 500 | 89,74 | 3,5e+00 | 174,73 | 527,46 |
| shooting | 10 | True | 158 | 43,51 | 1,1e-13 | 0,00 | 0,00 |
| collocation | 10 | True | 22 | 45,56 | 5,9e-12 | 23,82 | 71,61 |
| shooting | 20 | True | 123 | 39,33 | 5,5e-11 | 0,00 | 0,00 |
| collocation | 20 | True | 32 | 39,63 | 3,2e-12 | 6,73 | 20,45 |
| shooting | 40 | True | 147 | 38,34 | 5,1e-12 | 0,00 | 0,00 |
| collocation | 40 | True | 54 | 38,41 | 2,9e-12 | 2,27 | 6,92 |
| shooting | 80 | True | 158 | 38,09 | 1,4e-11 | 0,00 | 0,00 |
| collocation | 80 | True | 91 | 38,11 | 3,4e-13 | 0,61 | 1,87 |

1. **Aynı maliyet, farklı gerçek.** $N=10$'dan itibaren ikisi de $10^{-10}$ veya daha küçük residual ile yakınsadığını bildiriyor ve maliyetler buluşuyor ($N=80$'de 38,09 ve 38,11). Ama collocation'ın $N=10$'daki $u(t)$'si gerçekten simüle edilince dik konumun 23,82° gerisinde bitiyor. Solver ayrıklaştırılmış denklemleri tam çözdü; yanlış olan, knot'lar arasındaki trapez kuralı. Knot ekledikçe hata düşüyor: 23,82 → 6,73 → 2,27 → 0,61°.
2. **Shooting'in kısıtı simülasyonun kendisi.** Terminal kısıtı aynı parçalı doğrusal girdinin RK4 rollout'u üzerinde hesaplanıyor; bu yüzden replay, yazdırılan hassasiyette hedefe iniyor.
3. **Collocation'ı çözmek daha ucuz.** 123–158'e karşı 22–91 iterasyon; üstelik her shooting iterasyonu, finite-difference Jacobian'ı için $N+1$ ek rollout istiyor.
4. **$N=5$ ikisinde de başarısız.** Altı knot torque'u ile iki solver da uygulanabilir bir noktaya ulaşmadı. Tedrake'in belirttiği gibi, yerel bir solver bu problemin uygulanamaz mı olduğunu yoksa solver'ın sadece takılıp mı kaldığını söyleyemez.

Buradan çıkan pratik reçete: collocation ile çöz, sonra simülasyonla kontrol et (ya da collocation cevabından başlayıp shooting ile iyileştir) ve asla tek başına residual'a güvenme.

## Boz

Aynı problem, $N = 40$ ile collocation, farklı ilk tahminler. Sekiz gürültülü tahmin tek bir generator'dan (seed 0) sırayla çekiliyor: düz çizgi artı her knot state'ine Gauss gürültüsü ($\sigma = 1$) ve $\pm 5$ N·m içinde uniform torque'lar.

| ilk tahmin | yakınsadı | iterasyon | maliyet | replay hatası θ (derece) | torque işaret değişimi | θ aralığı (derece) |
|---|---|---|---|---|---|---|
| dipte duruyor, u = 0 | True | 54 | 38,41 | 2,27 | 3 | −77 .. 180 |
| düz çizgi θ 0 → π, u = 0 | True | 33 | 50,63 | 58,03 | 1 | −56 .. 180 |
| çizgi + gürültü, çekiliş 0, 1, 4, 5, 7 | True | 33–41 | 50,63 | 58,03 | 1 | −56 .. 180 |
| çizgi + gürültü, çekiliş 2, 3 | True | 50, 58 | 38,41 | 2,27 | 3 | −77 .. 180 |
| çizgi + gürültü, çekiliş 6 | False | 76 | 134,04 | 160,58 | 2 | −29 .. 197 |

Akla gelen ilk tahmin, aşağıdan yukarıya düz bir çizgi, temiz yakınsıyor ve başarı bildiriyor; ama farklı bir local minimum'a: üç dönüşlü bir pompalama yerine tek büyük bir geri salınım (θ −56°'ye kadar), maliyet 38,41 yerine 50,63. Sekiz gürültülü tahminin beşi aynı havzaya, ikisi daha iyisine düşüyor, biri hiç yakınsamıyor. Defect'ler $\sin\theta$ içeriyor, yani uygulanabilir küme convex değil ve SQP optimallik (KKT) koşullarını sağlayan yerel bir noktaya yakınsıyor; hangisine, başladığı yere bağlı. Tedrake, kendi acrobot ve cart-pole swing-up örneklerinde düz çizgi tahmininin gerekli olduğunu yazıyor; bizim sarkaçta işe yarıyor ama kötü minimum'a iniyor. Başarı bayrağı hangi minimum'u bulduğun hakkında hiçbir şey söylemez. Çareler: multi-start yapıp en iyisini tut, fiziksel olarak makul bir rollout'tan (ya da bir MPC koşusundan, ders 24) başlat veya continuation kullan (önce kolay bir problemi çöz, ör. daha büyük torque limiti, sonra sıkılaştır). Tek salınımlı çözümün aynı $N$'de neden çok daha kötü replay ettiğini ayrıştırmadık; her durumda, kararsız bir dengede biten open-loop bir planı izlemek için feedback gerekir.

## Gerçek robotta

- **Torque limitleri model verisidir.** Menagerie Panda, actuator kuvvetini joint 1–4'te ±87 N·m, joint 5–7'de ±12 N·m ile sınırlar ([`panda.xml`](https://github.com/google-deepmind/mujoco_menagerie/blob/f3b6013/franka_emika_panda/panda.xml)); SO-101 modeli STS3215 servo sınıfı için ±2,94 N·m kullanır ([`so101.xml`](https://github.com/google-deepmind/mujoco_menagerie/blob/f3b6013/robotstudio_so101/so101.xml)). Bu sayılar doğrudan sınırlara girer; modeldeki limit ölçülmüş bir donanım limiti olmadığı için altında pay bırak.
- **SO-101 prosedürü.** Modelde actuator'lar position servo'dur (`<position kp=...>`); yani optimize edilen hareket joint position trajectory'si olarak uygulanır ve torque yalnızca dolaylı olarak gözetilir. Model limitinin altında torque sınırlarıyla bir uzanma hareketi optimize et, position actuator'larla MuJoCo'da tekrar oynat, actuator kuvvetlerini sınırla karşılaştır, sonra pozisyonları düşük hızda gönder. Gerçek kolda henüz ölçülmedi.
- **Open-loop yetmez.** Simülasyonda bile tam yakınsamış bir collocation planı 0,61° ile 23,82° arasında ıskaladı. Donanımda model hatası ayrıklaştırma hatasının üstüne eklenir; plan bir feedback controller ile izlenmeli (ders 12 PD ya da trajectory etrafında LQR, ders 24).

## Kendini dene

1. **Kolay.** Yakınsamış her shooting çözümü neden 0,00° hatayla replay ediyor da collocation etmiyor?
2. **Orta.** Trapez defect'ini Hermite–Simpson ile değiştir (kübik state, aralık ortasında defect). $N=10$ replay hatasının nasıl değişeceğini tahmin et, sonra ölç.
3. **Zor.** $N=40$'ta 20 rastgele ilk tahmin çalıştır ve son maliyetleri kümele. Kaç farklı local minimum buluyorsun ve en iyisine en sık hangi tahmin ailesi ulaşıyor?

### Cevaplar

1. Shooting'in kısıtı, replay'in integre ettiği girdinin aynısının RK4 rollout'u üzerinde hesaplanıyor (MuJoCo'nun 1 ms'lik RK4'ü aynı girdinin bir başka ince adımlı integrasyonu); residual, integrasyon hatası dışında gerçek terminal hatayı ölçüyor. Collocation'ın residual'ı yalnızca trapez denklemlerini ölçer; ayrıklaştırma hataları kalır ($N=10$'da 23,82°).
2. Burada hesaplanmadı. Üçüncü dereceden bir şema aynı $N$'de daha küçük hata vermeli; ölç.
3. Bizim 10 tahminimizle: iki minimum (38,41 ve 50,63) ve yakınsamayan bir koşu (134,04). 20 tahmin için çalıştır.

## Akılda kalanlar ve kaynaklar

- Collocation ve shooting aynı maliyete ulaştı ($N=80$'de 38,11 ve 38,09), ama collocation'ın $N=10$'daki "yakınsamış" planı simüle edilince dik konumu 23,82° ıskaladı. Her zaman replay et.
- Collocation daha az iterasyon istedi (22–91'e karşı 123–158) ve state ilk tahmini kabul ediyor; shooting kullandığı integrator için kesin.
- Convex olmayan problemler ilk tahminin havzasına yakınsar: düz çizgi tahmini 50,63, dipte duran tahmin 38,41 maliyet verdi.

Sonraki: [ders 24](24-lqr-and-mpc.md), optimization'dan feedback: bir denge etrafında LQR ve her adımda yeniden çözen MPC.

**Kaynaklar**

- R. Tedrake, *Underactuated Robotics*, böl. 10 "Trajectory Optimization" (direct transcription, direct shooting, direct collocation, local minimum'lar). [underactuated.csail.mit.edu/trajopt.html](https://underactuated.csail.mit.edu/trajopt.html).
- SciPy, [`scipy.optimize.minimize` (SLSQP)](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html).
- MuJoCo Menagerie, [`franka_emika_panda/panda.xml`](https://github.com/google-deepmind/mujoco_menagerie/blob/f3b6013/franka_emika_panda/panda.xml), [`robotstudio_so101/so101.xml`](https://github.com/google-deepmind/mujoco_menagerie/blob/f3b6013/robotstudio_so101/so101.xml).
- Tüm tablolar: [`experiments/23-trajectory-optimization/output.txt`](../../experiments/23-trajectory-optimization/output.txt) (MuJoCo 3.17.0, SciPy 1.18.1, 2026-10-27).
