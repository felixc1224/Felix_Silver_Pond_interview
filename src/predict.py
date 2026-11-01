"""
Predict on your own images (phone photos, downloads) with a trained checkpoint.

    python -m src.predict                                    # every image in data/challenge/
    python -m src.predict --input data/challenge/cat.jpg     # one file
    python -m src.predict --checkpoint outputs/models/<experiment>_best.pt --top 5

Outputs -> data/output/<experiment>/challenge/
    predictions.csv     one row per image: top-k classes and confidences
    contact_sheet.png   original thumbnail | the 32x32 the model actually saw | top-k bars

Pre-processing (the model only knows 32x32 CIFAR-style images, so we make the photo look like one):
    EXIF rotate -> RGB -> centre-crop to square -> resize to 32x32 (antialiased) -> normalise
Expect lower accuracy than on the validation set: real photos are a different "domain"
(different cameras, framing, object size). That gap is a finding, not a bug.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps

from src.dataset import REPO_ROOT, build_transforms
from src.model import build_model

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# ----------------------------------------------------------------------------
# pre-processing
# ----------------------------------------------------------------------------
def load_image(path: Path) -> Image.Image:
    """Open any common image as an upright RGB PIL image."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)     # apply the phone's stored rotation
    return img.convert("RGB")              # drop alpha / greyscale / palette modes


def to_cifar_size(img: Image.Image, size: int = 32) -> Image.Image:
    """Centre-crop to a square, then shrink to size x size with antialiasing."""
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return img.crop((left, top, left + s, top + s)).resize((size, size), Image.LANCZOS)


def to_model_tensor(small: Image.Image, eval_tf) -> torch.Tensor:
    """32x32 RGB PIL -> normalised [1,3,32,32] tensor via the same eval transform as validation."""
    arr = np.array(small, dtype=np.uint8)                   # [32,32,3] (a copy, so torch may write to it)
    chw = torch.from_numpy(arr).permute(2, 0, 1)            # [3,32,32] uint8
    return eval_tf(chw).unsqueeze(0)                        # [1,3,32,32] float, normalised


# ----------------------------------------------------------------------------
# prediction
# ----------------------------------------------------------------------------
@torch.no_grad()
def predict_image(model, path: Path, eval_tf, names: list[str], top: int = 3) -> dict:
    img = load_image(path)
    small = to_cifar_size(img)
    probs = torch.softmax(model(to_model_tensor(small, eval_tf)), dim=1)[0].numpy()
    order = np.argsort(-probs)[:top]
    return dict(file=path.name, original_size=img.size, thumb=img.copy(), small=small,
                ranked=[(names[i], float(probs[i])) for i in order], probs=probs)


def contact_sheet(results: list[dict], out: Path, top: int) -> None:
    n = len(results)
    fig, axes = plt.subplots(n, 3, figsize=(9, 2.3 * n), squeeze=False,
                             gridspec_kw=dict(width_ratios=[1.2, 1, 1.8]))
    for row, r in zip(axes, results):
        t = r["thumb"].copy(); t.thumbnail((160, 160))
        row[0].imshow(t); row[0].set_title(f"{r['file']}\n{r['original_size'][0]}x{r['original_size'][1]}", fontsize=8); row[0].axis("off")
        row[1].imshow(r["small"], interpolation="nearest"); row[1].set_title("what the model saw (32x32)", fontsize=8); row[1].axis("off")
        labels = [f"{c} {100*p:.0f}%" for c, p in r["ranked"]]
        row[2].barh(range(top), [p for _, p in r["ranked"]], color="#0b7a5a")
        row[2].set_yticks(range(top), labels, fontsize=8); row[2].invert_yaxis()
        row[2].set_xlim(0, 1); row[2].set_xticks([0, .5, 1], ["0", "50%", "100%"], fontsize=7)
        row[2].set_title(f"top-{top}", fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def predict(checkpoint: Path, inputs: list[Path], top: int = 3, out_root: Path | None = None) -> list[dict]:
    state = torch.load(checkpoint, map_location="cpu")
    cfg = state["cfg"]
    model = build_model(cfg); model.load_state_dict(state["model"]); model.eval()
    _, eval_tf = build_transforms(cfg)
    names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    results = [predict_image(model, p, eval_tf, names, top) for p in inputs]

    out = (out_root or REPO_ROOT / cfg["paths"]["figures_dir"]) / cfg["experiment"] / "challenge"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "width", "height"] + [x for i in range(1, top + 1) for x in (f"top{i}", f"conf{i}")])
        for r in results:
            w.writerow([r["file"], *r["original_size"]] + [x for c, p in r["ranked"] for x in (c, f"{p:.4f}")])
    contact_sheet(results, out / "contact_sheet.png", top)

    width = max(len(r["file"]) for r in results)
    print(f"checkpoint: {checkpoint.name} (epoch {state.get('epoch')}, val acc {state.get('val_acc', 0):.3f})\n")
    for r in results:
        ranked = "  |  ".join(f"{c} {100*p:5.1f}%" for c, p in r["ranked"])
        print(f"{r['file']:<{width}}  {ranked}")
    print(f"\nsaved -> {out / 'predictions.csv'}, {out / 'contact_sheet.png'}")
    return results


def collect_inputs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise SystemExit(f"no images found in {path} (looked for {sorted(IMAGE_EXTS)})")
    return files


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="outputs/models/baseline_tinycnn_best.pt")
    ap.add_argument("--input", default="data/challenge", help="an image file or a folder of images")
    ap.add_argument("--top", type=int, default=3)
    args = ap.parse_args()
    predict(REPO_ROOT / args.checkpoint, collect_inputs(REPO_ROOT / args.input), args.top)
