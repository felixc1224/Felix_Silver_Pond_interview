"""
CIFAR-10 data loading for this project.

Data flow:
    data/raw/cifar-10-python.tar.gz          (untouched source, verified by MD5)
        └─ extract once ─▶ data/raw/cifar-10-batches-py/   (6 pickle files)
        └─ load into memory ─▶ numpy arrays  (50k train + 10k test, uint8 32x32x3)
        └─ stratified split ─▶ data/splits/train_idx.csv, val_idx.csv  (fixed seed)
        └─ wrap ─▶ torch Dataset / DataLoader

Design choice (Option 1 from our discussion): we never write 60k PNGs.
The whole dataset is ~180 MB of uint8, which fits comfortably in RAM.
"""

from __future__ import annotations

import csv
import pickle
import tarfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# ----------------------------------------------------------------------------
# Paths  (resolved relative to the repo root so the code works from anywhere)
# ----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
ARCHIVE = RAW_DIR / "cifar-10-python.tar.gz"
BATCHES_DIR = RAW_DIR / "cifar-10-batches-py"
SPLIT_DIR = REPO_ROOT / "data" / "splits"

TRAIN_BATCHES = [f"data_batch_{i}" for i in range(1, 6)]
TEST_BATCH = "test_batch"

# Per-channel mean / std of the CIFAR-10 training set (standard published values).
# Used to normalise pixels so every colour channel is centred around 0.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


# ----------------------------------------------------------------------------
# Step 1: extract + read the raw pickle batches
# ----------------------------------------------------------------------------
def extract_if_needed() -> Path:
    """Unpack the tar.gz once. Later runs skip this because the folder exists."""
    if not BATCHES_DIR.exists():
        if not ARCHIVE.exists():
            raise FileNotFoundError(f"Missing dataset archive: {ARCHIVE}")
        with tarfile.open(ARCHIVE, "r:gz") as tar:
            tar.extractall(RAW_DIR)
    return BATCHES_DIR


def _read_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """One pickle -> (images uint8 [N,32,32,3], labels int64 [N])."""
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    # Raw layout is [N, 3072] = 1024 red, 1024 green, 1024 blue values per image.
    # Reshape to [N, 3, 32, 32] then move channels last -> [N, 32, 32, 3] (HWC).
    images = d["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.asarray(d["labels"], dtype=np.int64)
    return images, labels


def load_cifar10() -> dict:
    """
    Returns a dict with everything in memory:
        x_train [50000,32,32,3] uint8, y_train [50000]
        x_test  [10000,32,32,3] uint8, y_test  [10000]
        class_names: list of 10 strings
    """
    root = extract_if_needed()
    xs, ys = zip(*(_read_batch(root / b) for b in TRAIN_BATCHES))
    x_train, y_train = np.concatenate(xs), np.concatenate(ys)
    x_test, y_test = _read_batch(root / TEST_BATCH)
    with open(root / "batches.meta", "rb") as f:
        class_names = pickle.load(f, encoding="latin1")["label_names"]
    return dict(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                class_names=class_names)


# ----------------------------------------------------------------------------
# Step 2: stratified train/val split, saved to CSV so it is fixed forever
# ----------------------------------------------------------------------------
def make_stratified_split(labels: np.ndarray, val_per_class: int = 500,
                          seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    For each class, shuffle its indices with a seeded RNG and take `val_per_class`
    for validation; the rest go to training. 10 classes x 500 = 5000 val images.
    """
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)      # all positions where label == c
        rng.shuffle(idx)
        val_idx.append(idx[:val_per_class])
        train_idx.append(idx[val_per_class:])
    train_idx = np.sort(np.concatenate(train_idx))
    val_idx = np.sort(np.concatenate(val_idx))
    return train_idx, val_idx


def _write_split_csv(path: Path, indices: np.ndarray, labels: np.ndarray,
                     class_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "label", "class_name"])
        for i in indices:
            w.writerow([int(i), int(labels[i]), class_names[labels[i]]])


def _read_split_csv(path: Path) -> np.ndarray:
    with open(path, newline="") as f:
        return np.array([int(row["index"]) for row in csv.DictReader(f)], dtype=np.int64)


def get_or_create_split(y_train: np.ndarray, class_names: list[str],
                        val_per_class: int = 500, seed: int = 42,
                        force: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Load the split CSVs if they exist, otherwise create and save them."""
    train_csv, val_csv = SPLIT_DIR / "train_idx.csv", SPLIT_DIR / "val_idx.csv"
    if train_csv.exists() and val_csv.exists() and not force:
        return _read_split_csv(train_csv), _read_split_csv(val_csv)
    train_idx, val_idx = make_stratified_split(y_train, val_per_class, seed)
    _write_split_csv(train_csv, train_idx, y_train, class_names)
    _write_split_csv(val_csv, val_idx, y_train, class_names)
    return train_idx, val_idx


# ----------------------------------------------------------------------------
# Step 3: PyTorch Dataset / DataLoader
# ----------------------------------------------------------------------------
class CIFAR10Arrays(Dataset):
    """
    A Dataset backed by in-memory numpy arrays.
    __getitem__ returns (image tensor [3,32,32] float32, label int).
    `transform` is an optional torchvision transform applied to a uint8 CHW tensor.
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        assert len(images) == len(labels), "images and labels must align"
        self.images = images          # uint8 [N,32,32,3]
        self.labels = labels          # int64 [N]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        # HWC uint8 numpy -> CHW uint8 tensor (what torchvision transforms expect)
        img = torch.from_numpy(self.images[i]).permute(2, 0, 1)
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = img.float() / 255.0  # plain [0,1] scaling if no transform given
        return img, int(self.labels[i])


def build_datasets(train_transform=None, eval_transform=None,
                   val_per_class: int = 500, seed: int = 42) -> dict:
    """Assemble train / val / test Datasets plus the metadata we need elsewhere."""
    data = load_cifar10()
    train_idx, val_idx = get_or_create_split(data["y_train"], data["class_names"],
                                             val_per_class, seed)
    return dict(
        train=CIFAR10Arrays(data["x_train"][train_idx], data["y_train"][train_idx], train_transform),
        val=CIFAR10Arrays(data["x_train"][val_idx], data["y_train"][val_idx], eval_transform),
        test=CIFAR10Arrays(data["x_test"], data["y_test"], eval_transform),
        class_names=data["class_names"],
        train_idx=train_idx,
        val_idx=val_idx,
    )


def build_dataloaders(batch_size: int = 128, num_workers: int = 0, **kwargs) -> dict:
    """
    DataLoaders for CPU training.
    num_workers=0 is deliberate: the data is already in RAM, so extra worker
    processes would mostly add overhead (and WSL multiprocessing can be flaky).
    """
    ds = build_datasets(**kwargs)
    loaders = dict(
        train_loader=DataLoader(ds["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers),
        val_loader=DataLoader(ds["val"], batch_size=batch_size, shuffle=False, num_workers=num_workers),
        test_loader=DataLoader(ds["test"], batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )
    return {**ds, **loaders}   # datasets under 'train'/'val'/'test', loaders under '*_loader'
