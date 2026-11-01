"""
Training loop — CPU only.

    python -m src.train                                  # full run from configs/baseline.yaml
    python -m src.train --config configs/exp02.yaml
    python -m src.train --epochs 2                       # override epochs
    python -m src.train --smoke                          # 5 batches/epoch, 2 epochs: plumbing check

Per epoch:  train (forward, loss, backward, step) -> validate -> log a CSV row
            -> save checkpoint if val accuracy improved -> early stop if stuck.

Outputs (paths from cfg['paths']):
    outputs/logs/<experiment>.csv            one row per epoch
    outputs/models/<experiment>_best.pt      best-val-accuracy weights + config + epoch
    outputs/models/<experiment>_last.pt      weights at the final epoch (for resuming / comparison)
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from src.dataset import REPO_ROOT, dataloaders_from_config
from src.model import build_model, count_parameters, layer_table


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Same seed -> same weight init, same shuffle order, same augmentation draws."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_optimizer(cfg: dict, model: nn.Module) -> torch.optim.Optimizer:
    t = cfg["train"]
    if t["optimizer"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=t["lr"], momentum=t.get("momentum", 0.9),
                               weight_decay=t.get("weight_decay", 0.0), nesterov=True)
    if t["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=t["lr"], weight_decay=t.get("weight_decay", 0.0))
    raise ValueError(f"unknown optimizer {t['optimizer']!r}, choose sgd | adam")


def build_scheduler(cfg: dict, optimizer, epochs: int):
    """Learning-rate schedule, stepped once per epoch. 'none' returns None."""
    s = cfg["train"].get("scheduler", "none")
    if s == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if s == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.1)
    if s == "none":
        return None
    raise ValueError(f"unknown scheduler {s!r}, choose cosine | step | none")


def run_epoch(model, loader, criterion, optimizer=None, max_batches: int | None = None):
    """
    One pass over `loader`. If `optimizer` is given we train, otherwise we only evaluate.
    Returns (mean loss, accuracy) for the pass.
    """
    training = optimizer is not None
    model.train(training)                      # dropout on/off
    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(training):     # no gradient graph when evaluating
        for b, (x, y) in enumerate(loader):
            if max_batches is not None and b >= max_batches:
                break
            logits = model(x)                  # [B,10]
            loss = criterion(logits, y)        # scalar
            if training:
                optimizer.zero_grad()          # clear last step's gradients
                loss.backward()                # compute new gradients
                optimizer.step()               # move the weights
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(dim=1) == y).sum().item()
            seen += len(y)
    return total_loss / seen, correct / seen


def run_card_text(cfg: dict, model: nn.Module, epochs: int) -> str:
    """Markdown: the full config + the measured layer table for one run."""
    rows = layer_table(model)
    lines = [f"# Run card — {cfg['experiment']}", "",
             f"- date: {time.strftime('%Y-%m-%d %H:%M')}",
             f"- model: `{cfg['model']['name']}`  |  parameters: {count_parameters(model):,}",
             f"- epochs: {epochs}  |  optimizer: {cfg['train']['optimizer']}  lr {cfg['train']['lr']}  "
             f"momentum {cfg['train'].get('momentum', '-')}  wd {cfg['train'].get('weight_decay', 0)}  "
             f"scheduler {cfg['train'].get('scheduler', 'none')}",
             f"- data: batch {cfg['data']['batch_size']}  normalize {cfg['data'].get('normalize', True)}  "
             f"augment {cfg['data'].get('augment', {})}",
             "", "## Architecture (measured)", "",
             "| layer | type | output shape | params |", "|---|---|---|---|"]
    lines += [f"| {r['name']} | {r['type']} | {'×'.join(map(str, r['out_shape']))} | {r['params']:,} |" for r in rows]
    lines += ["", "## Full config", "", "```yaml", yaml.safe_dump(cfg, sort_keys=False).strip(), "```"]
    return "\n".join(lines)


def write_run_card(cfg: dict, model: nn.Module, epochs: int, path: Path) -> None:
    path.write_text(run_card_text(cfg, model, epochs), encoding="utf-8")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def train(cfg: dict, epochs: int | None = None, max_batches: int | None = None) -> dict:
    set_seed(cfg["seed"])
    torch.set_num_threads(max(1, torch.get_num_threads()))   # use all CPU cores

    name = cfg["experiment"]
    epochs = epochs or cfg["train"]["epochs"]
    patience = cfg["train"].get("early_stopping_patience", 0)
    logs_dir = REPO_ROOT / cfg["paths"]["logs_dir"]
    models_dir = REPO_ROOT / cfg["paths"]["models_dir"]
    logs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{name}.csv"

    d = dataloaders_from_config(cfg)
    model = build_model(cfg)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer, epochs)

    print(f"experiment: {name} | params: {count_parameters(model):,} | "
          f"train {len(d['train'])} / val {len(d['val'])} | epochs {epochs} | "
          f"threads {torch.get_num_threads()}")

    # Snapshot the exact configuration + architecture used for this run, so the
    # log / figures can always be traced back to the settings that produced them.
    write_run_card(cfg, model, epochs, logs_dir / f"{name}_config.md")

    best_acc, best_epoch, bad_epochs = 0.0, 0, 0
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "seconds"])

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            lr = optimizer.param_groups[0]["lr"]
            tr_loss, tr_acc = run_epoch(model, d["train_loader"], criterion, optimizer, max_batches)
            va_loss, va_acc = run_epoch(model, d["val_loader"], criterion, None, max_batches)
            if scheduler is not None:
                scheduler.step()
            secs = time.time() - t0

            w.writerow([epoch, f"{lr:.6f}", f"{tr_loss:.4f}", f"{tr_acc:.4f}", f"{va_loss:.4f}", f"{va_acc:.4f}", f"{secs:.1f}"])
            f.flush()
            improved = va_acc > best_acc
            print(f"epoch {epoch:3d}/{epochs} | lr {lr:.4f} | train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
                  f"val loss {va_loss:.3f} acc {va_acc:.3f} | {secs:5.1f}s" + ("  *best*" if improved else ""))

            state = dict(model=model.state_dict(), cfg=cfg, epoch=epoch, val_acc=va_acc)
            torch.save(state, models_dir / f"{name}_last.pt")
            if improved:
                best_acc, best_epoch, bad_epochs = va_acc, epoch, 0
                torch.save(state, models_dir / f"{name}_best.pt")
            else:
                bad_epochs += 1
                if patience and bad_epochs >= patience:
                    print(f"early stopping: no val improvement for {patience} epochs")
                    break

            if not math.isfinite(tr_loss):
                print("loss is NaN/inf — stopping. Usually the learning rate is too high.")
                break

    print(f"done. best val acc {best_acc:.4f} at epoch {best_epoch} -> {models_dir / f'{name}_best.pt'}")
    print(f"log -> {log_path}")
    return dict(best_val_acc=best_acc, best_epoch=best_epoch, log=log_path,
                best_ckpt=models_dir / f"{name}_best.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--epochs", type=int, default=None, help="override cfg['train']['epochs']")
    ap.add_argument("--smoke", action="store_true", help="2 epochs x 5 batches: check the plumbing only")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(REPO_ROOT / args.config))
    if args.smoke:
        cfg["experiment"] += "_smoke"
        train(cfg, epochs=2, max_batches=5)
    else:
        train(cfg, epochs=args.epochs)
