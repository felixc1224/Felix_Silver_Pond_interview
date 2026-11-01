"""
Model definitions.

v1: TinyCNN — two conv blocks + one hidden FC layer. ~270k parameters.

Shape walk-through for one 32x32 RGB image (batch dimension omitted):

    input          [3, 32, 32]
    conv1 3x3      [16, 32, 32]   16 filters, padding=1 keeps the size
    relu           [16, 32, 32]
    maxpool 2x2    [16, 16, 16]   halves height and width
    conv2 3x3      [32, 16, 16]
    relu           [32, 16, 16]
    maxpool 2x2    [32, 8, 8]
    flatten        [2048]         32 * 8 * 8
    fc1            [128]
    relu           [128]
    (dropout)      [128]
    fc2            [10]           one score per class ("logits")

Design choice: 3x3 convs with padding=1 so that only the pooling layers change
the spatial size. That makes the dimension story easy to follow and to draw.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn


class TinyCNN(nn.Module):
    def __init__(self, conv_channels=(16, 32), fc_hidden: int = 128,
                 dropout: float = 0.0, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        c1, c2 = conv_channels

        # ---- feature extractor: learns WHAT is in the image ----
        self.features = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)),
            ("relu1", nn.ReLU(inplace=True)),
            ("pool1", nn.MaxPool2d(kernel_size=2)),
            ("conv2", nn.Conv2d(c1, c2, kernel_size=3, padding=1)),
            ("relu2", nn.ReLU(inplace=True)),
            ("pool2", nn.MaxPool2d(kernel_size=2)),
        ]))

        # after two 2x2 pools a 32x32 input is 8x8
        flat_dim = c2 * 8 * 8

        # ---- classifier: turns features into 10 class scores ----
        self.classifier = nn.Sequential(OrderedDict([
            ("flatten", nn.Flatten()),
            ("fc1", nn.Linear(flat_dim, fc_hidden)),
            ("relu3", nn.ReLU(inplace=True)),
            ("dropout", nn.Dropout(dropout)),
            ("fc2", nn.Linear(fc_hidden, num_classes)),
        ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    @torch.no_grad()
    def forward_with_activations(self, x: torch.Tensor) -> "OrderedDict[str, torch.Tensor]":
        """
        Same as forward() but also returns every intermediate tensor, keyed by
        layer name. Used by scripts/visualize_layers.py to show feature maps.
        """
        # .clone() matters: ReLU(inplace=True) overwrites its input tensor, so without
        # a copy the stored conv output would already be post-ReLU (bug found in review).
        acts: "OrderedDict[str, torch.Tensor]" = OrderedDict(input=x.clone())
        for name, layer in list(self.features.named_children()) + list(self.classifier.named_children()):
            x = layer(x)
            acts[name] = x.clone()
        return acts


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
MODELS = {"tiny_cnn": TinyCNN}


def build_model(cfg: dict) -> nn.Module:
    """Construct the model named in cfg['model']['name'] with its settings."""
    m = dict(cfg["model"])
    name = m.pop("name")
    if name not in MODELS:
        raise ValueError(f"unknown model '{name}', choose from {list(MODELS)}")
    return MODELS[name](**m)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def layer_table(model: TinyCNN, input_shape=(3, 32, 32)) -> list[dict]:
    """
    One row per layer: name, type, output shape, #params.
    Runs a dummy forward pass so the shapes are measured, not assumed.
    """
    model.eval()
    x = torch.zeros(1, *input_shape)
    acts = model.forward_with_activations(x)
    layers = dict(list(model.features.named_children()) + list(model.classifier.named_children()))
    rows = [dict(name="input", type="Input", out_shape=tuple(acts["input"].shape[1:]), params=0)]
    for name, t in acts.items():
        if name == "input":
            continue
        layer = layers[name]
        rows.append(dict(name=name, type=type(layer).__name__,
                         out_shape=tuple(t.shape[1:]),
                         params=sum(p.numel() for p in layer.parameters())))
    return rows


if __name__ == "__main__":
    # quick self-check: python -m src.model
    model = TinyCNN()
    print(model)
    print(f"\ntrainable parameters: {count_parameters(model):,}")
    print(f"\n{'layer':10s} {'type':12s} {'output shape':16s} {'params':>9s}")
    for r in layer_table(model):
        print(f"{r['name']:10s} {r['type']:12s} {str(r['out_shape']):16s} {r['params']:9,d}")
