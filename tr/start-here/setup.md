# Kurulum

Bölüm 1–7 ve 8–10'un çoğu bir dizüstünde koşar. Kitaptaki sayılar Apple Silicon bir Mac'te (Apple M5), Python 3.14 ve `uv.lock` içinde kilitlenmiş tam sürümlerle (MuJoCo 3.17.0, NumPy 2.5.4, SciPy, PyTorch 2.14, Gymnasium, OpenCV, Matplotlib) üretildi.

## Kur

Repo bir [uv](https://docs.astral.sh/uv/) projesi: `pyproject.toml` bağımlılıkları listeler, `uv.lock` her sürümü sabitler, `.python-version` Python 3.14'ü sabitler (sende yoksa uv indirir).

```bash
git clone https://github.com/fport/physical-ai-from-scratch.git
cd physical-ai-from-scratch
uv sync
uv run python experiments/02-timestep/experiment.py
```

uv olmadan: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt` (dosya `uv.lock`'tan üretilir).

Her deneyin ilk satırı kütüphane sürümlerini yazdırır. Başka sürümlerde bazı sonuçların son basamakları farklı çıkabilir.

## Robot modelleri

SO-101, Franka Panda ya da Unitree Go2 kullanan deneyler modelleri `experiments/common/menagerie.py` üzerinden [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)'den yükler; bu modül üç model klasörünü sabit bir commit'te bir kez `.cache/mujoco_menagerie` altına klonlar.

## Colab

`notebooks/` altındaki notebook'lar bağımlılıklarını ilk hücrede kendileri kurar. Çalışma türü **GPU** olan dersler hangi Colab runtime'ını istediklerini yazar.

## Mac'in yapamadıkları

Bazı araçlar macOS'ta yok ve yalnız kaynaklarından anlatılır: Isaac Sim/Lab Linux ya da Windows ister ([gereksinimler](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)); π0 inference'ı 8 GB'tan büyük bir NVIDIA GPU ister ([openpi](https://github.com/Physical-Intelligence/openpi)).

## Kitabı web sitesi olarak oku

```bash
cd site
npm install
npm run dev
```
