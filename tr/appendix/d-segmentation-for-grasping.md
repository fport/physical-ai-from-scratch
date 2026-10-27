# Ek D · Grasp point için segmentation

> Bir mask hangi pikselin nesneye ait olduğunu söyler; bir grasp ise iki contact noktası, bir yaklaşma yönü ve bir çene açıklığı ister. Aradaki adım geometridir ve hataların çoğu orada olur.

**Mask'ler.** Segment Anything (SAM), SA-1B üzerinde (11 milyon görüntüde yaklaşık 1 milyar mask) eğitilmiş, prompt'lanabilir bir segmentation modelidir ve yeni görüntü dağılımlarına zero-shot aktarılır ([arXiv 2304.02643](https://arxiv.org/abs/2304.02643)). SAM 2 onu streaming bir bellekle videoya genişletiyor ve görüntü segmentation'ında SAM'den 6× hızlı olduğunu raporluyor ([arXiv 2408.00714](https://arxiv.org/abs/2408.00714)). Bir prompt (bir nokta, bir kutu ya da dil koşullu bir detector'dan bir tespit) nesneyi seçer; çıkış ikili bir mask'tir.

**Mask'ten grasp adaylarına (paralel gripper).**

1. Mask'lenmiş depth piksellerini [ders 19–20](../part-5-perception-and-estimation/20-pixels-to-table.md)'nin kamera intrinsics ve extrinsics'iyle robot frame'inde bir point cloud'a geri projekte et. Önce mask'i birkaç piksel aşındır: kenar pikselleri nesne ve arka plan derinliğini karıştırır.
2. Noktaların ana eksenleri (PCA): yukarıdan bir grasp için çeneleri küçük yatay eksen boyunca kapat, −z boyunca yaklaş, merkezi centroid'e koy. Kapanma eksenindeki genişlik çene açıklığını aşıyorsa reddet.
3. Antipodal kontrol: iki contact'taki yüzey normalleri friction cone'ların içinde kalmalı, [ders 16](../part-4-dynamics-and-control/16-grasping-and-pick-place.md)'nın koşulu.
4. Erişilebilirlik: pre-grasp, grasp ve kaldırma pozları için joint limitleri içinde IK ([ders 10](../part-3-kinematics/10-inverse-kinematics.md)); açık çenelerin komşu mask'lere karşı çarpışma kontrolü.

**Öğrenilmiş grasp üreticileri.** Contact-GraspNet, 17 milyon simüle grasp ile eğitilmiş olarak, 6-DoF paralel çene grasp'lerini doğrudan bir depth point cloud'undan tahmin ediyor ve yapılandırılmış dağınıklıkta %90'ın üzerinde başarı raporluyor ([arXiv 2103.14127](https://arxiv.org/abs/2103.14127)). Kodu, her nesnenin çevresindeki bölgeleri kırpmak (`--local_regions`) ve yalnızca nesnenin yüzeyindeki contact'ları tutmak (`--filter_grasps`) için isteğe bağlı bir segmentation haritası kullanıyor ([repository](https://github.com/NVlabs/contact_graspnet)): mask *hangi* nesneyi seçer, ağ *nasıl* tutulacağını önerir. Gemini Robotics-ER, embodied reasoning çıktıları arasında grasp tahminini sayıyor ([arXiv 2503.20020](https://arxiv.org/abs/2503.20020)).

**Nerede kırılır.** Masaya taşan mask'ler centroid'i kaydırır ve genişliği şişirir; ince, saydam ya da parlak nesneler eksik depth verir; yanlış nesnenin doğru bir mask'i (benzer bir nesneye çözülen bir dil prompt'u) yanlış şeyin mükemmel bir grasp'ini üretir; bu, [ders 42](../part-10-foundation-models-and-agents/42-llm-agents.md)'de ölçülen hedef ikamesinin perception karşılığıdır. Grasp'i mask skoruyla değil fiziksel olarak doğrula (çene contact'ı ve kaldırma).

**SO-101'de.** [Ders 21](../part-5-perception-and-estimation/21-object-perception.md)'in renk eşiği detector'ı tek renkli küpler için mask adımıdır; keyfi nesneler için yerine SAM tarzı bir model geçer. Bu kitapta SAM'i ya da Contact-GraspNet'i koşturmadık; buradaki hiçbir sayı bize ait değil.

Kaynaklar: [SAM, arXiv 2304.02643](https://arxiv.org/abs/2304.02643) · [SAM 2, arXiv 2408.00714](https://arxiv.org/abs/2408.00714) · [Contact-GraspNet, arXiv 2103.14127](https://arxiv.org/abs/2103.14127) ve [kod](https://github.com/NVlabs/contact_graspnet) · [Gemini Robotics, arXiv 2503.20020](https://arxiv.org/abs/2503.20020).
