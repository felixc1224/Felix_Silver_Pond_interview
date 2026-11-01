# CIFAR-10 image classifier — CPU-only, step by step

A small convolutional neural network trained on CIFAR-10 (60,000 colour images, 32×32 px,
10 classes) using Python + PyTorch on **CPU only** inside Ubuntu WSL.

The project is built deliberately one piece at a time, with an AI assistant (Claude) writing
code and explaining every decision, and me reviewing, questioning and verifying each piece
before the next one starts. The full conversation is kept verbatim in `doc/ai-chat-history/`.

## Status

| Step | What | State |
|------|------|-------|
| 1 | Folder structure, WSL environment, data cleanup | done |
| 2 | Data loader + stratified train/val split + verification script | done — **v1, AI-written, human review pending** |
| 3 | Config + TinyCNN model + layer visualisation page | done — reviewed |
| 3b | Test suite (pytest) for dataset + model | done |
| 4 | Training loop (`src/train.py`) + transforms/augmentation | done — AI-written, review pending |
| 5 | Evaluation (`src/evaluate.py`): curves, confusion matrix, per-class, calibration, worst mistakes | done — AI-written, review pending |
| 6 | Predict on my own images (`src/predict.py`, `data/challenge/`) | done — AI-written, review pending |
| 7 | Experiments: lr, epochs, scheduler (configs `exp01`–`exp04`) | done — see `doc/experiments.md` |
| 8 | Architecture experiments: BatchNorm, extra conv block | next |

## Folder structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── configs/            # one YAML per experiment: baseline.yaml, exp01_… exp04_…
├── src/                # the actual code, imported as a package
│   ├── dataset.py      # load CIFAR-10 pickles -> numpy -> stratified split -> torch Dataset/DataLoader
│   ├── model.py        # TinyCNN + build_model / layer_table helpers
│   ├── train.py        # training loop: epochs, logging, best checkpoint, early stopping
│   ├── evaluate.py     # metrics (numpy) + all evaluation figures as PNG
│   └── predict.py      # your own photos -> ranked predictions + contact sheet
├── scripts/            # runnable entry points
│   ├── verify_data.py  # sanity-check the data pipeline (counts, overlap, shapes, sample grid)
│   ├── visualize_layers.py          # real images through every layer -> interactive HTML
│   ├── layer_walkthrough_template.html
│   └── run_experiment.sh            # train -> evaluate -> predict for one config
├── tests/              # pytest: conftest.py (fixtures), test_dataset.py, test_model.py, test_metrics.py (todo)
├── data/
│   ├── raw/            # cifar-10-python.tar.gz  (NOT in git — see data/raw/README.md)
│   ├── splits/         # train_idx.csv, val_idx.csv — the fixed train/val split (in git)
│   ├── output/         # visual checks & evaluation plots (sample grid, curves, confusion matrix)
│   └── challenge/      # my own images for the final test of the model
├── outputs/
│   ├── models/         # saved weights (NOT in git)
│   └── logs/           # training logs (NOT in git)
├── doc/
│   ├── ai-chat-history/  # verbatim AI conversation log, one file per session
│   ├── decision-log.md   # decisions + alternatives considered
│   ├── experiments.md    # what was tried, results
│   ├── experiments/      # saved figures + logs per run (e.g. baseline_tinycnn_cloud_run/)
│   ├── pipeline.md       # diagrams (pipeline, training, prediction, network) + baseline results
│   ├── diagrams/         # the diagrams as PNG + editable Mermaid .mmd
│   ├── REVIEW_GUIDE.md   # review checklist + standing reminders / end-of-project changes
│   └── learning-notes.md # my own notes / explanations in simple English
└── app/                # (later) small demo app
```

## Setup (Ubuntu WSL, CPU only)

```bash
cd /mnt/c/uni/2026_sem2/SilverPond_Interview_Test

# 1. virtual environment — on the LINUX side of WSL, not under /mnt/c.
#    Importing torch opens thousands of small files; through the /mnt/c bridge that takes
#    minutes instead of seconds. The project itself can stay on C:.
python3 -m venv ~/.venvs/silverpond
source ~/.venvs/silverpond/bin/activate

# 2. CPU build of PyTorch first (the default pip build pulls ~2 GB of CUDA libraries we don't need)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. everything else
pip install -r requirements.txt

# 4. data: download cifar-10-python.tar.gz from https://www.cs.toronto.edu/~kriz/cifar.html
#    and place it at data/raw/cifar-10-python.tar.gz, then check it:
md5sum data/raw/cifar-10-python.tar.gz     # expect c58f30108f718f92721af3b95e74349a
```

## How to run

Always run from the repo root so `src/` is importable.

```bash
# Step 2 — verify the data pipeline (extracts the archive on first run, ~5 s)
python scripts/verify_data.py          # writes data/output/verify_random_samples.png
python scripts/verify_data.py --show   # also opens a window (needs WSLg)
python scripts/verify_data.py --seed 7 # pick a different set of 8 random images
```

Expected: `train: 45000`, `val: 5000` (exactly 500 per class), `test: 10000`,
`overlap: 0`, batches of shape `(128, 3, 32, 32)` in range [0, 1], and `ALL CHECKS PASSED`.

# Step 3 — model and visualisation
python -m src.model                     # print layer table + parameter count
python scripts/visualize_layers.py      # writes data/output/layer_walkthrough.html (open in a browser)
python scripts/visualize_layers.py --checkpoint outputs/models/best.pt   # after training

# Step 4 — train (CPU; ~10 s/epoch on 2 cores, so the 15-epoch baseline is a few minutes)
python -m src.train --smoke             # 2 epochs x 5 batches: check the plumbing first (~3 s)
python -m src.train                     # full run from configs/baseline.yaml
python -m src.train --config configs/other.yaml --epochs 20
#   -> outputs/logs/<experiment>.csv, outputs/models/<experiment>_best.pt / _last.pt

# Step 5 — evaluate (on the VALIDATION set by default)
python -m src.evaluate                  # uses outputs/models/baseline_tinycnn_best.pt
python -m src.evaluate --checkpoint outputs/models/<experiment>_best.pt
python -m src.evaluate --split test     # the sealed test set: run ONCE, for the final model only
#   -> data/output/<experiment>/<split>/  learning_curves.png, confusion_matrix.png,
#      per_class_accuracy.png, calibration.png, worst_mistakes.png, metrics.json, summary.md

# Step 6 — predict on your own photos (drop JPG/PNG files into data/challenge/)
python -m src.predict                               # every image in data/challenge/, top-3
python -m src.predict --input data/challenge/cat.jpg --top 5
#   -> data/output/<experiment>/challenge/predictions.csv + contact_sheet.png
#      (original | the 32x32 the model actually saw | top-k bars)
# Pre-processing: EXIF rotate -> RGB -> centre-crop square -> 32x32 -> normalise.
# Expect lower accuracy than on val: phone photos are a different domain from CIFAR.
# The four _synthetic_*.{jpg,png} files are CIFAR test images disguised as phone photos
# (borders, EXIF rotation, alpha, greyscale) to prove the pre-processing paths work.

# Step 7 — experiments: one config per run, train -> evaluate -> predict in one command
bash scripts/run_experiment.sh exp01_lr_third        # any name in configs/<name>.yaml
#   -> outputs/logs/<name>.csv + <name>_config.md (run card), outputs/models/<name>_best.pt,
#      data/output/<name>/val/ and /challenge/
# New experiment: copy configs/baseline.yaml, change `experiment:` and the knobs, run, add a row to doc/experiments.md

## Tests

```bash
pytest -v          # 49 tests, a few seconds
```

`tests/test_dataset.py` checks the split (sizes, stratification, no train/val overlap, determinism,
the committed CSVs are unchanged) and the Dataset class. `tests/test_model.py` checks TinyCNN's
shapes against the docstring table, the exact parameter count, config loading and its error
messages, gradient flow to every parameter, dropout behaviour, and a regression test for the
in-place-ReLU recorder bug found during review. Dataset tests skip (not fail) if the archive is
missing. `tests/test_metrics.py` checks the numpy metric functions (confusion matrix, per-class
precision/recall/F1, top-k, calibration/ECE) against hand-computed answers. `tests/test_inference.py`
checks the photo pre-processing (mode conversion, EXIF rotation, centre crop, normalisation, ranked
output). 49 tests total.

## Experiments so far

| config | what changed vs baseline |
|---|---|
| `configs/baseline.yaml` | — (lr 0.05, 15 epochs, cosine) |
| `configs/exp01_lr_third.yaml` | lr 0.05 → 0.0167 |
| `configs/exp02_lr_third_45ep.yaml` | lr 0.0167 and epochs 15 → 45 |
| `configs/exp03_sched_step.yaml` | scheduler cosine → step |
| `configs/exp04_sched_none.yaml` | scheduler cosine → none (constant lr) |

Results and my observations: `doc/experiments.md`. Pipeline / training / prediction / network diagrams
and the baseline numbers: `doc/pipeline.md` (sources in `doc/diagrams/*.mmd`).

## Data

- **Source:** the original CIFAR-10 python release (Krizhevsky, 2009), 50,000 train + 10,000 test
  images, perfectly balanced (5,000 / 1,000 per class). Not committed; see `data/raw/README.md`.
- **Split:** the 50,000 training images are split **45,000 train / 5,000 val**, stratified
  (500 per class in val), with a fixed random seed (42). The chosen indices are saved to
  `data/splits/*.csv` and reused on every run, so the split never changes. Regenerating them
  from scratch produces byte-identical files (`md5`: train `cc0bef88…`, val `a57d306e…`).
- **Test set** (10,000, labelled) is kept sealed: it is only used once, for the final number.
- Images are kept **in memory as numpy arrays** (~180 MB) rather than written out as 60k PNG
  files — faster on CPU and keeps the repo clean.

## Workflow / AI usage

Each step: the assistant proposes a plan and the alternatives → I approve → it writes and
tests the code → it explains the code block by block → I review line by line, ask questions,
and try to break it → only then the next step. Every exchange is in `doc/ai-chat-history/`.
Bugs found along the way (and who found them) are recorded there too.
