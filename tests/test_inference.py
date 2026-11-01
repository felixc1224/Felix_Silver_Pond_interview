"""Tests for src/predict.py pre-processing: any photo must become an upright 32x32 RGB tensor."""

import numpy as np
import pytest
import torch
from PIL import Image

from src.dataset import build_transforms
from src.model import TinyCNN
from src.predict import load_image, predict_image, to_cifar_size, to_model_tensor


def _save(tmp_path, name, img, **kw):
    p = tmp_path / name
    img.save(p, **kw)
    return p


def test_load_image_converts_modes(tmp_path):
    rgba = Image.new("RGBA", (50, 40), (255, 0, 0, 128))
    grey = Image.new("L", (50, 40), 77)
    for p in (_save(tmp_path, "a.png", rgba), _save(tmp_path, "g.png", grey)):
        img = load_image(p)
        assert img.mode == "RGB" and img.size == (50, 40)


def test_load_image_applies_exif_rotation(tmp_path):
    # 60 wide x 20 tall, stored rotated with orientation tag 6 -> must come back as 60x20 upright
    img = Image.new("RGB", (60, 20), (0, 255, 0))
    rot = img.rotate(90, expand=True)                 # now 20x60 on disk
    exif = rot.getexif(); exif[0x0112] = 6
    p = _save(tmp_path, "r.jpg", rot, exif=exif)
    assert load_image(p).size == (60, 20)


def test_to_cifar_size_crops_centre_square():
    # left third red, middle green, right third blue; centre crop of 300x100 keeps the green
    arr = np.zeros((100, 300, 3), dtype=np.uint8)
    arr[:, :100] = (255, 0, 0); arr[:, 100:200] = (0, 255, 0); arr[:, 200:] = (0, 0, 255)
    small = to_cifar_size(Image.fromarray(arr))
    assert small.size == (32, 32)
    assert np.array(small)[16, 16].tolist() == [0, 255, 0]


def test_to_model_tensor_shape_and_normalisation():
    cfg = {"data": {"normalize": True, "augment": {}}}
    _, eval_tf = build_transforms(cfg)
    small = Image.new("RGB", (32, 32), (255, 255, 255))
    t = to_model_tensor(small, eval_tf)
    assert t.shape == (1, 3, 32, 32) and t.dtype == torch.float32
    assert t.max() > 1.0, "white should map above 1 after mean/std normalisation"


def test_predict_image_returns_ranked_probabilities(tmp_path):
    torch.manual_seed(0)
    model = TinyCNN().eval()
    cfg = {"data": {"normalize": True, "augment": {}}}
    _, eval_tf = build_transforms(cfg)
    names = [f"c{i}" for i in range(10)]
    p = _save(tmp_path, "x.jpg", Image.new("RGB", (200, 150), (10, 200, 30)))
    r = predict_image(model, p, eval_tf, names, top=3)
    assert len(r["ranked"]) == 3
    confs = [c for _, c in r["ranked"]]
    assert confs == sorted(confs, reverse=True), "ranked must be descending"
    assert r["probs"].sum() == pytest.approx(1.0, abs=1e-5)
    assert r["ranked"][0][1] == pytest.approx(r["probs"].max())
