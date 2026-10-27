# Sayı ve kaynak kuralları

Bu kitabın tek vaadi: **her sayı gerçek.** Bunu tutmak için yazarken uyulan kurallar şunlar.

## Sayılar

1. **Kendi deneyimizden gelen sayı** yalnız koşturulmuş bir deneyden yazılır; koşunun tam çıktısı script'in yanında `output.txt` olarak saklanır. Yanında MuJoCo (ya da ilgili kütüphane) sürümü, donanım ve deneyin deterministik olup olmadığı belirtilir.
2. **Başkasından gelen sayı** (bir config satırı, bir belge, bir makale) bağlantısıyla verilir. Birincil kaynak tercih edilir: kodun kendisi, resmî belge, makale.
3. **Henüz koşturulmamış bir deneyin sonucu yazılmaz.** "Yakında" işaretli sayfalarda yalnız plan durur.
4. **Success rate** söylenen her yerde deneme sayısı da söylenir. [Ders 25](../part-7-learning-from-demonstration/25-evaluating-policies.md)'ten itibaren confidence interval da verilir.
5. **Doğrulanamayan iddia** ya metinden çıkarılır ya da açıkça "doğrulanmadı" diye işaretlenir.

## Simülasyon ve gerçek

- Simülasyon sonucu "simülasyonda" diye yazılır. Gerçek dünyadaki karşılığı ölçülmediyse "gerçekte de böyledir" denmez.
- Bir modelin (ör. MuJoCo'nun hava direnci modeli) sonucu, ölçüm gibi sunulmaz.
- Gerçek bir robotun eğitim sayıları ile gerçek robotta kullanılan sayılar ayrı tutulur (bkz. [ders 12](../part-4-dynamics-and-control/12-joint-control.md): eğitimde 20 / 0,5, gerçek Go2 örneğinde 60 / 5).

## Kaynak türleri

| İşaret | Anlamı |
|---|---|
| Birincil | Kodun, belgenin ya da makalenin kendisi okundu |
| İkincil | Üçüncü taraf özet, haber, arama sonucu. Yalnız birincil bulunamazsa ve öyle belirtilerek |
| Doğrulanmadı | Kaynağa erişilemedi ya da kaynak bunu söylemiyor. Metinde iddia olarak kullanılmaz |

## Telif

- Robot modelleri MuJoCo Menagerie'den, kendi lisanslarıyla (ör. `robotstudio_so101` Apache-2.0, `unitree_go2` BSD-3-Clause). Görüntüler kendi render'ımız.
- Üretici adları olgusal olarak kullanılır; hiçbir üretici bu kitabı desteklemez ya da onaylamaz.
- Başka kaynaklardan uzun alıntı yapılmaz; kısa alıntılar kaynağıyla verilir.
