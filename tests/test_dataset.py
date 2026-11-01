"""Tests for src/dataset.py — the data pipeline must be exactly what we designed."""

import hashlib
from collections import Counter

import numpy as np
import pytest
import torch

from src.dataset import (CIFAR10Arrays, SPLIT_DIR, _read_split_csv, _write_split_csv,
                         make_stratified_split)

# md5 of the committed split files (data/splits/*.csv). If these ever change,
# the validation set has moved and every past result is no longer comparable.
TRAIN_CSV_MD5 = "cc0bef88b177537af19ab6000d976dfe"
VAL_CSV_MD5 = "a57d306eea5ae07809c7410020bb571f"


# ---------------------------------------------------------------- counts
def test_split_sizes(data):
    assert len(data["train"]) == 45_000
    assert len(data["val"]) == 5_000
    assert len(data["test"]) == 10_000


def test_ten_classes_with_names(data):
    assert data["class_names"] == ["airplane", "automobile", "bird", "cat", "deer",
                                   "dog", "frog", "horse", "ship", "truck"]


def test_val_is_stratified(data):
    counts = Counter(data["val"].labels.tolist())
    assert sorted(counts) == list(range(10))
    assert all(v == 500 for v in counts.values())


def test_train_is_balanced(data):
    counts = Counter(data["train"].labels.tolist())
    assert all(v == 4_500 for v in counts.values())


def test_test_is_balanced(data):
    counts = Counter(data["test"].labels.tolist())
    assert all(v == 1_000 for v in counts.values())


# ---------------------------------------------------------------- split integrity
def test_no_overlap_and_full_coverage(data):
    train, val = set(data["train_idx"].tolist()), set(data["val_idx"].tolist())
    assert not (train & val), "train/val leak"
    assert train | val == set(range(50_000))


def test_indices_are_sorted_and_unique(data):
    for key in ("train_idx", "val_idx"):
        idx = data[key]
        assert np.all(np.diff(idx) > 0), f"{key} not strictly increasing"


def test_split_is_deterministic():
    labels = np.repeat(np.arange(10), 5_000)
    a = make_stratified_split(labels, 500, seed=42)
    b = make_stratified_split(labels, 500, seed=42)
    c = make_stratified_split(labels, 500, seed=43)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert not np.array_equal(a[1], c[1]), "different seed should give a different split"


def test_split_is_stratified_on_synthetic_labels():
    labels = np.repeat(np.arange(10), 5_000)
    train_idx, val_idx = make_stratified_split(labels, 500, seed=0)
    assert len(train_idx) == 45_000 and len(val_idx) == 5_000
    assert all(v == 500 for v in Counter(labels[val_idx].tolist()).values())


def test_committed_split_csvs_unchanged():
    """The split on disk is the one every experiment has used."""
    def md5(p):
        return hashlib.md5(p.read_bytes()).hexdigest()
    train_csv, val_csv = SPLIT_DIR / "train_idx.csv", SPLIT_DIR / "val_idx.csv"
    if not train_csv.exists():
        pytest.skip("split CSVs not present")
    assert md5(train_csv) == TRAIN_CSV_MD5
    assert md5(val_csv) == VAL_CSV_MD5


def test_csv_roundtrip(tmp_path):
    labels = np.repeat(np.arange(10), 5)
    idx = np.array([0, 7, 12, 49])
    path = tmp_path / "x.csv"
    _write_split_csv(path, idx, labels, [f"c{i}" for i in range(10)])
    back = _read_split_csv(path)
    assert np.array_equal(back, idx)
    assert path.read_text().splitlines()[0] == "index,label,class_name"


# ---------------------------------------------------------------- Dataset class
def test_getitem_shape_dtype_range(data):
    img, label = data["train"][0]
    assert isinstance(img, torch.Tensor) and img.shape == (3, 32, 32)
    assert img.dtype == torch.float32
    assert 0.0 <= img.min() and img.max() <= 1.0
    assert isinstance(label, int) and 0 <= label <= 9


def test_getitem_matches_source_pixels(data):
    """__getitem__ must be a pure HWC->CHW /255 of the stored uint8 image."""
    ds = data["train"]
    img, _ = ds[123]
    expected = torch.from_numpy(ds.images[123]).permute(2, 0, 1).float() / 255.0
    assert torch.equal(img, expected)


def test_index_out_of_range_raises(data):
    with pytest.raises(IndexError):
        _ = data["train"][len(data["train"])]


def test_misaligned_images_labels_rejected():
    images = np.zeros((10, 32, 32, 3), dtype=np.uint8)
    labels = np.zeros(9, dtype=np.int64)
    with pytest.raises(AssertionError):
        CIFAR10Arrays(images, labels)


def test_transform_is_applied():
    images = np.full((2, 32, 32, 3), 255, dtype=np.uint8)
    labels = np.zeros(2, dtype=np.int64)
    ds = CIFAR10Arrays(images, labels, transform=lambda t: t.float() * 0 + 7)
    img, _ = ds[0]
    assert float(img.min()) == 7.0 and float(img.max()) == 7.0
