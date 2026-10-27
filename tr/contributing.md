# Katkı

Bu kitap katkıya açık: GitHub'da bir issue ya da pull request aç.

## Bir sayıyı düzeltmek

Bir dersteki her sayı yeniden üretilebilir olmalı. Tutmayan bir sayı varsa dersi, kütüphane sürümlerini ve çıktını yaz. Yeniden üretilemeyen bir sayı bir hatadır.

## Bir ders yazmak ya da iyileştirmek

1. Yazım kılavuzunu oku: [`tools/AUTHORING.md`](../tools/AUTHORING.md) (şablon, dürüstlük kuralları, Türkçe kuralları).
2. Ders yapısı [`curriculum.yaml`](../curriculum.yaml)'dan gelir; onu değiştirdikten sonra `uv run python tools/build_book.py` çalıştır. Bu komut `SUMMARY.md`'yi, bölüm sayfalarını ve her dersin künye bloğunu yeniden üretir. Üretilen kısımları elle düzenleme.
3. Deneyi `experiments/NN-slug/experiment.py` olarak yaz (İngilizce kod, sabit seed'ler, tablo yazdırır) ve tam stdout'unu `output.txt` olarak kaydet.
4. Dersi İngilizce (`part-…/NN-slug.md`) ve Türkçe (`tr/part-…/NN-slug.md`) yaz. Türkçede teknik terimler İngilizce kalır.
5. Her göreli bağlantının çalıştığını ve sitenin derlendiğini kontrol et: `cd site && npm run build`.

## Kod kuralları

- İngilizce identifier, yorum ve çıktı.
- Bağımlılıklar `pyproject.toml`'da, sürümler `uv.lock`'ta sabit (`uv add <paket>`, ardından `uv export --no-hashes --no-dev --format requirements-txt > requirements.txt`).
- Pull request öncesi: `uv run ruff check .` ve `uv run python tools/check_book.py`.
- Robot modelleri `experiments/common/menagerie.py` üzerinden.
