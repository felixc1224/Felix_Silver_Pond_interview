"""
Build an interactive HTML walk-through of TinyCNN: push a few REAL CIFAR-10
images through the network and show what every layer outputs (feature maps,
shapes, parameter counts, activation stats, final class scores).

Run from the repo root:
    python scripts/visualize_layers.py                        # untrained (random) weights
    python scripts/visualize_layers.py --checkpoint outputs/models/best.pt
    python scripts/visualize_layers.py --indices 0 7 42 99    # choose which train images

Output: data/output/layer_walkthrough.html  (self-contained, open in any browser)
"""

import argparse
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.dataset import build_datasets  # noqa: E402
from src.model import build_model, count_parameters, layer_table  # noqa: E402

OUT_HTML = REPO_ROOT / "data" / "output" / "layer_walkthrough.html"
TEMPLATE = REPO_ROOT / "scripts" / "layer_walkthrough_template.html"


# ----------------------------------------------------------------------------
# image helpers: numpy -> PNG data URI (nearest-neighbour upscaled so pixels are visible)
# ----------------------------------------------------------------------------
def to_data_uri(arr_uint8: np.ndarray, scale: int) -> str:
    img = Image.fromarray(arr_uint8)
    img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def fmap_to_uri(fmap: np.ndarray, size: int = 64) -> str:
    """One feature map [H,W] float -> grayscale PNG, normalised to its own min/max."""
    lo, hi = float(fmap.min()), float(fmap.max())
    norm = (fmap - lo) / (hi - lo) if hi > lo else np.zeros_like(fmap)
    return to_data_uri((norm * 255).astype(np.uint8), scale=max(1, size // fmap.shape[0]))


def kernel_to_uri(k: np.ndarray, size: int = 48) -> str:
    """A conv1 kernel [3,3,3] (RGB in, 3x3) -> small RGB image."""
    lo, hi = float(k.min()), float(k.max())
    norm = (k - lo) / (hi - lo) if hi > lo else np.zeros_like(k)
    rgb = (norm.transpose(1, 2, 0) * 255).astype(np.uint8)  # [3,3,3] CHW -> HWC
    return to_data_uri(rgb, scale=size // 3)


# ----------------------------------------------------------------------------
# collect everything for one image
# ----------------------------------------------------------------------------
def describe_image(model, ds, class_names, idx: int) -> dict:
    img_t, label = ds[idx]                                  # [3,32,32] float in [0,1]
    acts = model.forward_with_activations(img_t.unsqueeze(0))
    raw = ds.images[idx]                                    # uint8 HWC for display

    layers = []
    for name, t in acts.items():
        t = t[0]                                            # drop batch dim
        a = t.numpy()
        entry = dict(name=name, shape=list(t.shape),
                     min=float(a.min()), max=float(a.max()), mean=float(a.mean()),
                     zero_frac=float((a == 0).mean()))
        if a.ndim == 3:                                     # [C,H,W] -> one image per channel
            entry["maps"] = [fmap_to_uri(a[c]) for c in range(a.shape[0])]
        else:                                               # 1-D vector -> bar values
            entry["values"] = [float(v) for v in a]
        layers.append(entry)

    logits = acts["fc2"][0]
    probs = torch.softmax(logits, dim=0).tolist()
    return dict(
        index=int(idx), label=int(label), label_name=class_names[label],
        image=to_data_uri(raw, scale=4),
        layers=layers,
        probs=probs,
        predicted=int(np.argmax(probs)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--checkpoint", default=None, help="optional .pt with model weights")
    ap.add_argument("--indices", type=int, nargs="+", default=None,
                    help="train-set positions to visualise (default: one each of 4 classes)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(REPO_ROOT / args.config))
    torch.manual_seed(cfg["seed"])
    model = build_model(cfg)
    trained = False
    if args.checkpoint:
        state = torch.load(REPO_ROOT / args.checkpoint, map_location="cpu")
        model.load_state_dict(state["model"] if "model" in state else state)
        trained = True
    model.eval()

    d = build_datasets()
    ds, names = d["train"], d["class_names"]

    if args.indices is None:
        # first image of each of four visually different classes
        want = [names.index(c) for c in ("airplane", "cat", "ship", "frog")]
        args.indices = [int(np.flatnonzero(ds.labels == c)[0]) for c in want]

    payload = dict(
        trained=trained,
        checkpoint=args.checkpoint,
        total_params=count_parameters(model),
        table=[dict(r, out_shape=list(r["out_shape"])) for r in layer_table(model)],
        conv1_kernels=[kernel_to_uri(k) for k in model.features.conv1.weight.detach().numpy()],
        class_names=names,
        images=[describe_image(model, ds, names, i) for i in args.indices],
    )

    fragment = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/", json.dumps(payload))
    html = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '</head><body style="margin:0">' + fragment + '</body></html>')
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html, encoding="utf-8")
    OUT_HTML.with_suffix(".fragment.html").write_text(fragment, encoding="utf-8")
    print(f"wrote {OUT_HTML}  ({OUT_HTML.stat().st_size / 1e6:.2f} MB, "
          f"{len(args.indices)} images, weights={'trained' if trained else 'random init'})")


if __name__ == "__main__":
    main()
