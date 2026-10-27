# 19 · Replay öğrenme değildir

> Aynı hareketi birebir oynattım. Küpü 3 cm kaydırınca ıskaladı.

> **Durum: taslak.** Deney simülasyonda koşturuldu, notebook hazır ve sayılar doğrulandı. Gerçek SO-101 ile `lerobot-replay` sweep'i henüz yapılmadı.

| | |
|---|---|
| **Faz** | [4 · Göstererek öğretmek](README.md) |
| **Tür** | Deney |
| **Önkoşul** | [13 · Yörünge](../phase-2-muscle/13-trajectory.md), [14 · Güvenlik](../phase-2-muscle/14-safety.md) (sim deneyi için şart değil) |
| **Süre** | ~45 dk |
| **Araç** | MuJoCo 3.17 · Menagerie `robotstudio_so101` · Colab ya da Mac |
| **Kod** | [notebook](../../notebooks/19-replay.ipynb) |
| **Durum** | 📝 taslak · notebook hazır |

## Neden önemli

LeRobot iş akışında bir kaydı tekrar oynatmanın resmî bir komutu var: `lerobot-replay`, kayıttaki action'ları robota olduğu gibi tekrar gönderir ([LeRobot · gerçek robotla imitation learning](https://huggingface.co/docs/lerobot/il_robots)). Kol kayıttaki hareketi kusursuz tekrar ettiğinde "robot öğrendi" sanmak kolay. Bu ders o yanılgıyı ölçüyor: kayıt ile dünya arasındaki küçük bir fark yeterli.

## Kavram

**Open-loop** kontrol, çıktıyı ölçmeden komut uygular; **closed-loop** (feedback) kontrol ölçtüğüne göre komutu değiştirir (Lynch & Park, *Modern Robotics* §11.1; Åström & Murray, *Feedback Systems* §1.1). Replay saf open-loop'tur: listeyi sırayla gönderir, küpe hiç bakmaz.

## Deney (notebook'ta)

- Kol bir küpü **bir kez** alıp kaldırır (IK ile yazılmış hareket, 4,5 sn). Her 2 ms'de gönderilen 6 komut (5 eklem + gripper) kaydedilir: **2250 adım × 6**.
- **Replay:** aynı liste, hiç değiştirilmeden. Küp 0,5–5 cm (0,5 cm adımla 10 mesafe) × 8 yön kaydırılır, artı kayıttaki yer: **81 deneme**.
- **Feedback'li kural:** aynı zaman çizelgesi, ama hedef pozlar küpün o denemedeki yerinden IK ile yeniden hesaplanır. Öğrenilmiş parametre yok.
- **Başarı fiziksel:** küp yalnız sürtünmeyle tutuluyor (yapıştırma/weld yok). Ölçüt: t = 4,5 sn'de küpün altı zeminden ≥ 4 cm yukarıda ve küp iki çeneye de değiyor.

## Sayılarla sonuç

MuJoCo 3.17.0, deterministik (iki koşu aynı sonuç).

| Küpün kayması | Replay success rate (8 yönde) |
|---|---|
| 0 cm (kayıttaki yer) | tuttu |
| 0,5 cm | 8/8 |
| 1 cm | 7/8 |
| 1,5 cm | 4/8 |
| 2 cm | 3/8 |
| 2,5 cm | 1/8 |
| 3 cm | 1/8 |
| 4 cm | 1/8 |
| 4,5 ve 5 cm | 0/8 |

- **Replay toplam: 27/81.** Feedback'li kural: **81/81**.
- 3 cm ileri kayınca gripper küpün eski yerinde kapanıyor, küpe hiç dokunulmuyor.
- Yön önemli: küp hareketli çene tarafına (+y) kayınca 4 cm'ye kadar tuttu; sabit çene tarafına (−y) kayınca 1 cm'de kaçırdı. İleri/geri tolerans 1–1,5 cm.
- 8 denemelik oranların güven aralığı (confidence interval) geniş: ör. 1/8 için Wilson %95 aralığı 0,02–0,47. Bu konu [ders 22](22-evaluation.md)'de.

Tam tablo (3,5 cm dahil) notebook'un sonuç dosyasında.

## Sınırlar

- **Feedback'li kural da öğrenme değil.** Küpün yeri simülasyondan **mükemmel** okunuyor (kamera yok, gürültü yok) ve hareket başlamadan bir kez okunuyor. Daha kesin terim "algıya göre yeniden planlama" (replanning from perception).
- Kayıt simülasyonda 2 ms'de bir. Gerçek SO-101'de LeRobot kaydı ~30 fps ([il_robots](https://huggingface.co/docs/lerobot/il_robots)).
- Menagerie modelinin gripper çarpışmaları (collision) manipülasyon için ayarlanmış ilkel şekiller; aktüatör kazançları (gain) gerçek servonunkiyle birebir değil. Simdeki tolerans sayıları gerçek kola aynen taşınmaz.
- Her (mesafe, yön) tek koşu.
- İlk home pozunda feedback'li koşulardan biri self-collision ile başarısız oldu (80/81); home pozu değiştirildi. Üç farklı home pozunda replay hep 27/81, feedback'li kural hep 81/81 çıktı.

## Alıştırmalar

1. **Kolay.** Notebook'un 6. bölümünde gripper'ı daha az aç (`G_OPEN`). Replay toleransı daralıyor mu?
2. **Orta.** Küpü büyüt (`CUBE_HALF`). Hangi yönlerde tolerans artıyor?
3. **Zor.** Feedback'li kurala küp konumu okumasında ±5 mm gürültü ekle. 81/81 kalıyor mu?

## Kaynaklar

- [LeRobot · gerçek robotla imitation learning](https://huggingface.co/docs/lerobot/il_robots): kayıt, replay, fps.
- [MuJoCo Menagerie · `robotstudio_so101`](https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101) (Apache-2.0).
- Lynch & Park, [*Modern Robotics*](https://hades.mech.northwestern.edu/index.php/Modern_Robotics), Bölüm 11.
- Åström & Murray, [*Feedback Systems*](https://fbswiki.org), Bölüm 1.
- Sayılar Koddan Kasa 19 deneyinin sonuç dosyasından ve iç doğrulama notundan (27.10.2026).
