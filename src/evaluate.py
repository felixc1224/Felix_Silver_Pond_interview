"""
Evaluate a trained checkpoint and save every result as PNG / JSON / Markdown.

    python -m src.evaluate                                   # best checkpoint of baseline, on VAL
    python -m src.evaluate --checkpoint outputs/models/baseline_tinycnn_best.pt
    python -m src.evaluate --split test                      # the sealed test set: run ONCE, at the end

Outputs -> data/output/<experiment>/<split>/
    learning_curves.png        train/val loss and accuracy per epoch (from outputs/logs/<experiment>.csv)
    confusion_matrix.png       10x10, rows = true class, columns = predicted
    per_class_accuracy.png     bar per class, with precision / recall / F1 table in metrics.json
    calibration.png            reliability diagram: confidence vs. actual accuracy
    worst_mistakes.png         the most confident wrong predictions, with images
    metrics.json               every number
    summary.md                 human-readable summary for the report

Metric functions are plain numpy (no scikit-learn) and unit-tested in tests/test_metrics.py.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # file output only; no window needed on WSL
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.dataset import REPO_ROOT, dataloaders_from_config, denormalize
from src.model import build_model


# ----------------------------------------------------------------------------
# metrics (pure numpy, testable)
# ----------------------------------------------------------------------------
def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """cm[i, j] = number of samples whose true class is i and predicted class is j."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def per_class_metrics(cm: np.ndarray) -> dict:
    """Per class: accuracy (= recall), precision, F1, support. Zero-safe."""
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1)                       # true count per class
    predicted = cm.sum(axis=0)                     # predicted count per class
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return dict(recall=recall, precision=precision, f1=f1, support=support)


def top_k_accuracy(probs: np.ndarray, y_true: np.ndarray, k: int = 2) -> float:
    topk = np.argsort(-probs, axis=1)[:, :k]
    return float((topk == y_true[:, None]).any(axis=1).mean())


def calibration_bins(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> dict:
    """Group predictions by confidence; compare mean confidence to actual accuracy per bin.
    Returns bin accuracy / confidence / count and ECE (expected calibration error)."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    acc, avg_conf, count = np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        m = (conf > edges[b]) & (conf <= edges[b + 1]) if b else (conf >= edges[b]) & (conf <= edges[b + 1])
        if m.any():
            acc[b], avg_conf[b], count[b] = correct[m].mean(), conf[m].mean(), m.sum()
    ece = float(np.sum(count / max(1, count.sum()) * np.abs(acc - avg_conf)))
    return dict(edges=edges, acc=acc, conf=avg_conf, count=count, ece=ece)


# ----------------------------------------------------------------------------
# inference
# ----------------------------------------------------------------------------
@torch.no_grad()
def predict_all(model, loader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the whole loader. Returns (probs [N,10], y_true [N], images [N,3,32,32] normalised)."""
    model.eval()
    probs, ys, xs = [], [], []
    for x, y in loader:
        probs.append(torch.softmax(model(x), dim=1))
        ys.append(y)
        xs.append(x)
    return torch.cat(probs).numpy(), torch.cat(ys).numpy(), torch.cat(xs)


# ----------------------------------------------------------------------------
# plots
# ----------------------------------------------------------------------------
def plot_learning_curves(log_csv: Path, out: Path, title: str) -> bool:
    if not log_csv.exists():
        return False
    rows = list(csv.DictReader(open(log_csv)))
    ep = [int(r["epoch"]) for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    a1.plot(ep, [float(r["train_loss"]) for r in rows], marker="o", label="train")
    a1.plot(ep, [float(r["val_loss"]) for r in rows], marker="o", label="val")
    a1.set(title="Loss", xlabel="epoch", ylabel="cross-entropy"); a1.grid(alpha=0.3); a1.legend()
    a2.plot(ep, [100 * float(r["train_acc"]) for r in rows], marker="o", label="train")
    a2.plot(ep, [100 * float(r["val_acc"]) for r in rows], marker="o", label="val")
    a2.set(title="Accuracy", xlabel="epoch", ylabel="%"); a2.grid(alpha=0.3); a2.legend()
    best = max(rows, key=lambda r: float(r["val_acc"]))
    a2.annotate(f"best val {100*float(best['val_acc']):.1f}% @ ep {best['epoch']}",
                xy=(int(best["epoch"]), 100 * float(best["val_acc"])), xytext=(5, -25),
                textcoords="offset points", fontsize=8, arrowprops=dict(arrowstyle="->"))
    fig.suptitle(title); fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return True


def plot_confusion_matrix(cm: np.ndarray, names: list[str], out: Path, title: str) -> None:
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set(xlabel="predicted", ylabel="true", title=title)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, label="fraction of true class")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def plot_per_class(pc: dict, names: list[str], overall: float, out: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(names))
    ax.bar(x, 100 * pc["recall"], color="#0b7a5a")
    ax.axhline(100 * overall, color="#b4571b", ls="--", lw=1, label=f"overall {100*overall:.1f}%")
    ax.set_xticks(x, names, rotation=45, ha="right"); ax.set(ylabel="accuracy (%)", ylim=(0, 100), title=title)
    for i, v in enumerate(pc["recall"]):
        ax.text(i, 100 * v + 1, f"{100*v:.0f}", ha="center", fontsize=8)
    ax.legend(loc="lower right"); fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def plot_calibration(cal: dict, out: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    centers = (cal["edges"][:-1] + cal["edges"][1:]) / 2
    has = cal["count"] > 0
    ax.bar(centers[has], cal["acc"][has], width=0.1, edgecolor="black", color="#0b7a5a", alpha=0.8, label="actual accuracy")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    ax.set(xlabel="predicted confidence", ylabel="actual accuracy", xlim=(0, 1), ylim=(0, 1),
           title=f"{title}\nECE = {cal['ece']:.3f}")
    ax.legend(loc="upper left", fontsize=8); fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def plot_worst_mistakes(images: torch.Tensor, probs: np.ndarray, y_true: np.ndarray,
                        names: list[str], out: Path, n: int = 16) -> None:
    pred, conf = probs.argmax(axis=1), probs.max(axis=1)
    wrong = np.flatnonzero(pred != y_true)
    worst = wrong[np.argsort(-conf[wrong])][:n]
    cols = 8; rows = int(np.ceil(len(worst) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 2.4))
    for ax, i in zip(np.ravel(axes), worst):
        ax.imshow(denormalize(images[i]).permute(1, 2, 0).numpy())
        ax.set_title(f"true {names[y_true[i]]}\npred {names[pred[i]]} {100*conf[i]:.0f}%", fontsize=7)
        ax.axis("off")
    for ax in np.ravel(axes)[len(worst):]:
        ax.axis("off")
    fig.suptitle(f"{len(worst)} most confident mistakes", fontsize=10)
    fig.tight_layout(h_pad=2.0); fig.savefig(out, dpi=150); plt.close(fig)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def evaluate(checkpoint: Path, split: str = "val", out_root: Path | None = None) -> dict:
    state = torch.load(checkpoint, map_location="cpu")
    cfg = state["cfg"]
    name = cfg["experiment"]
    model = build_model(cfg)
    model.load_state_dict(state["model"])

    d = dataloaders_from_config(cfg)
    names = d["class_names"]
    loader = {"val": d["val_loader"], "test": d["test_loader"]}[split]

    out = (out_root or REPO_ROOT / cfg["paths"]["figures_dir"]) / name / split
    out.mkdir(parents=True, exist_ok=True)

    probs, y_true, images = predict_all(model, loader)
    y_pred = probs.argmax(axis=1)
    cm = confusion_matrix(y_true, y_pred, len(names))
    pc = per_class_metrics(cm)
    cal = calibration_bins(probs, y_true)
    acc = accuracy(y_true, y_pred)
    title = f"{name} — {split} ({len(y_true)} images)"

    curves = plot_learning_curves(REPO_ROOT / cfg["paths"]["logs_dir"] / f"{name}.csv", out / "learning_curves.png", name)
    plot_confusion_matrix(cm, names, out / "confusion_matrix.png", title)
    plot_per_class(pc, names, acc, out / "per_class_accuracy.png", title)
    plot_calibration(cal, out / "calibration.png", title)
    plot_worst_mistakes(images, probs, y_true, names, out / "worst_mistakes.png")

    # the two most confused pairs (off-diagonal), for the summary
    off = cm.copy(); np.fill_diagonal(off, 0)
    pairs = sorted(((off[i, j], names[i], names[j]) for i in range(10) for j in range(10)), reverse=True)[:3]

    metrics = dict(
        experiment=name, split=split, checkpoint=str(checkpoint), epoch=state.get("epoch"),
        n=int(len(y_true)), accuracy=acc, top2_accuracy=top_k_accuracy(probs, y_true, 2),
        macro_f1=float(pc["f1"].mean()), ece=cal["ece"],
        per_class={n_: dict(accuracy=float(pc["recall"][i]), precision=float(pc["precision"][i]),
                            f1=float(pc["f1"][i]), support=int(pc["support"][i])) for i, n_ in enumerate(names)},
        confusion_matrix=cm.tolist(),
        most_confused=[dict(count=int(c), true=t, predicted=p) for c, t, p in pairs],
    )
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    lines = [f"# {name} — {split} set", "",
             f"- checkpoint: `{checkpoint.name}` (epoch {state.get('epoch')})",
             f"- images: {len(y_true)}",
             f"- **accuracy: {100*acc:.2f}%**  |  top-2: {100*metrics['top2_accuracy']:.2f}%  |  macro-F1: {metrics['macro_f1']:.3f}  |  ECE: {cal['ece']:.3f}",
             "", "| class | accuracy | precision | F1 | n |", "|---|---|---|---|---|"]
    lines += [f"| {n_} | {100*pc['recall'][i]:.1f}% | {100*pc['precision'][i]:.1f}% | {pc['f1'][i]:.3f} | {int(pc['support'][i])} |"
              for i, n_ in enumerate(names)]
    lines += ["", "Most confused pairs (true → predicted): " +
              ", ".join(f"{t}→{p} ({c})" for c, t, p in pairs), "",
              "Figures: " + ", ".join(p.name for p in sorted(out.glob("*.png")))]
    if not curves:
        lines.append("\n(no learning_curves.png: training log not found)")
    # append the run card (architecture + full config) so the figures and the
    # settings that produced them are never separated
    from src.train import run_card_text
    lines += ["", "---", "", run_card_text(cfg, model, state.get("epoch", cfg["train"]["epochs"]))]
    (out / "summary.md").write_text("\n".join(lines))
    metrics["config"] = cfg

    print("\n".join(lines))
    print(f"\nsaved -> {out}")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="outputs/models/baseline_tinycnn_best.pt")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    args = ap.parse_args()
    if args.split == "test":
        print("WARNING: evaluating on the TEST set. This should happen once, for the final model only.\n")
    evaluate(REPO_ROOT / args.checkpoint, args.split)
