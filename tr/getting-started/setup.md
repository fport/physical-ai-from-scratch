# Kurulum: bir Mac, 0 TL

İlk dört fazın bütün deneyleri bir dizüstü bilgisayarda ya da ücretsiz Colab'de koşar. GPU, robot ya da ücretli hesap gerekmez.

## Seçenek A · Kendi Mac'in (önerilen)

Kitaptaki sayılar Apple Silicon bir Mac'te, Python 3.12, MuJoCo 3.17.0 ve NumPy 2.5 ile üretildi.

```bash
git clone https://github.com/fport/physical-ai-from-scratch.git
cd physical-ai-from-scratch
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python experiments/02-timestep/experiment.py
```

Çıktının ilk satırı `mujoco 3.17.0` olmalı. Başka bir sürümde son basamaklar biraz kayabilir; kitap bunu her deneyde belirtir.

## Seçenek B · Colab

Her dersin notebook'u `notebooks/` altında. Notebook'un ilk hücresi kendi bağımlılıklarını kurar (`pip install mujoco==3.17.0`). Menagerie modeli gereken notebook'lar (Go2, SO-101) ilgili klasörü kendisi klonlar.

> "Open in Colab" rozetleri, repo herkese açık olunca eklenecek.

## Hangi derste ne gerekir?

| Etiket | Anlamı | Nerede |
|---|---|---|
| **MJ** | MuJoCo, Mac'te yerel ya da Colab CPU | Faz 0–3'ün çoğu |
| **LR** | LeRobot | Faz 4'ten itibaren |
| **KOL** | Gerçek SO-101 | 06'dan itibaren insan katmanı; 10, 14, 19, 26, 36, 40'ta asıl deney |
| **CO-GPU** | Colab GPU ya da kiralık GPU | 26 (eğitim), 33–34, EK E |
| **OL** | Ollama ile yerel dil modeli | Faz 5 |

Mac'te yapılamayanlar dürüstçe işaretli: Isaac Sim/Lab macOS'ta desteklenmiyor ([gereksinimler](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)); π0 inference'ı > 8 GB NVIDIA GPU istiyor ([openpi](https://github.com/Physical-Intelligence/openpi)). Bunlar kitapta yalnız kaynak gösterilerek anlatılır ([EK G](../appendix/g-big-robot-models.md)).

## Sorun çıkarsa

- `ModuleNotFoundError: mujoco` → virtual environment aktif değil: `source .venv/bin/activate`.
- Ders 31 (Strands) MuJoCo **3.16.0** ister; ayrı bir virtual environment'ta kur (sebebi derste).
