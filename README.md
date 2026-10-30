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
| 3 | Model (small CNN) | next |
| 4 | Training loop, learning curves | todo |
| 5 | Evaluation: confusion matrix, per-class accuracy, test set | todo |
| 6 | Predict on my own images (`data/challenge/`) | todo |

## Folder structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── configs/            # experiment settings (YAML), e.g. baseline.yaml
├── src/                # the actual code, imported as a package
│   ├── dataset.py      # load CIFAR-10 pickles -> numpy -> stratified split -> torch Dataset/DataLoader
│   ├── model.py        # (next) CNN definition
│   ├── train.py        # (todo) training loop
│   ├── evaluate.py     # (todo) metrics, confusion matrix
│   └── predict.py      # (todo) run the trained model on new images
├── scripts/            # runnable entry points
│   └── verify_data.py  # sanity-check the data pipeline (counts, overlap, shapes, sample grid)
├── tests/              # pytest unit tests (todo)
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
│   └── learning-notes.md # my own notes / explanations in simple English
└── app/                # (later) small demo app
```

## Setup (Ubuntu WSL, CPU only)

```bash
cd /mnt/c/uni/2026_sem2/SilverPond_Interview_Test

# 1. virtual environment
python3 -m venv .venv
source .venv/bin/activate

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

Later steps (train / evaluate / predict) will be added here as they are built.

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
