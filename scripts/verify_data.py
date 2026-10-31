"""
Sanity-check the dataset pipeline BEFORE building any model.

Run from the repo root:
    python scripts/verify_data.py            # saves the image grid to data/output/
    python scripts/verify_data.py --show     # also pops up the matplotlib window

Checks:
  1. image / label counts per split and per class
  2. train and val indices do not overlap, and together cover all 50k images
  3. the split CSVs on disk match what the loader uses
  4. one batch from each DataLoader has the right shape / dtype / value range
  5. a grid of 8 random training images with their labels (visual check)
"""

import argparse 
import sys 
from collections import Counter #help counting and verify the data number for each class"
from pathlib import Path #help access the folder for our structure"

import matplotlib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1] #helping from dataset to access the parent folder "
sys.path.insert(0, str(REPO_ROOT)) #setup the reporoot to become easy to access like having index"                # so `from src...` works

from src.dataset import SPLIT_DIR, build_dataloaders  # noqa: E402

OUT_DIR = REPO_ROOT / "data" / "output" 

"checking the image number that the data.py say is correct perclass"
def check_counts(d):
    names = d["class_names"]
    print(f"classes ({len(names)}): {names}")
    for split in ("train", "val", "test"):
        counts = Counter(d[split].labels.tolist())
        per_class = [counts[c] for c in range(len(names))]
        print(f"{split:5s}: {len(d[split]):6d} images | per class: {per_class}")
    assert len(d["train"]) == 45_000 and len(d["val"]) == 5_000 and len(d["test"]) == 10_000
    assert all(v == 500 for v in Counter(d["val"].labels.tolist()).values()), "val not stratified"
    print("OK counts")

"checking is the train and validation is overlap or have extra copy important so data leaking not happen"
def check_no_overlap(d):
    train_set, val_set = set(d["train_idx"].tolist()), set(d["val_idx"].tolist())
    overlap = train_set & val_set
    print(f"train idx: {len(train_set)}, val idx: {len(val_set)}, overlap: {len(overlap)}")
    assert not overlap, f"LEAK: {len(overlap)} indices in both train and val"
    assert train_set | val_set == set(range(50_000)), "train+val do not cover all 50k"
    print(f"split CSVs: {SPLIT_DIR / 'train_idx.csv'} , {SPLIT_DIR / 'val_idx.csv'}")
    print("OK no overlap, full coverage")

"check the actually datat checking the size,pixel value and other format"
def check_batch_shapes(d):
    for split in ("train", "val", "test"):
        x, y = next(iter(d[f"{split}_loader"]))
        print(f"{split:5s} batch: x {tuple(x.shape)} {x.dtype} range [{x.min():.3f}, {x.max():.3f}] | y {tuple(y.shape)} {y.dtype}")
        assert x.shape[1:] == (3, 32, 32) and x.dtype.is_floating_point
        assert 0.0 <= x.min() and x.max() <= 1.0, "expected [0,1] pixel range with no transform"
    print("OK batch shapes")

"make human can visuall verify the label and picture is correct"
def show_random_samples(d, n_show=8, seed=None, show=False):
    import matplotlib.pyplot as plt

    ds, names = d["train"], d["class_names"]
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(ds), n_show, replace=False)

    plt.figure(figsize=(8, 3))
    for i, idx in enumerate(idxs):
        img, label = ds[int(idx)]                 # img: [3,32,32] float in [0,1]
        img_disp = img.permute(1, 2, 0).numpy()  # -> [32,32,3] for imshow
        plt.subplot(2, 4, i + 1)
        plt.imshow(img_disp)
        plt.title(f"{names[label]} ({label})", fontsize=9)
        plt.axis("off")
    plt.suptitle("Random training samples from CIFAR-10", fontsize=12, y=1.02)
    plt.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "verify_random_samples.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved sample grid -> {out}")
    if show:
        plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true", help="open the matplotlib window")
    ap.add_argument("--seed", type=int, default=None, help="seed for which 8 images to show")
    args = ap.parse_args()
    if not args.show:
        matplotlib.use("Agg")                    # headless: just write the PNG

    d = build_dataloaders(batch_size=128)
    check_counts(d)
    check_no_overlap(d)
    check_batch_shapes(d)
    show_random_samples(d, seed=args.seed, show=args.show)
    print("\nALL CHECKS PASSED")
