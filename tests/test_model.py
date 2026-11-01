"""Tests for src/model.py — TinyCNN must have exactly the shape we designed."""

import pytest
import torch

from src.model import MODELS, TinyCNN, build_model, count_parameters, layer_table

EXPECTED_PARAMS = 268_650

# the table from the model.py docstring, as (name, output shape without batch)
EXPECTED_SHAPES = [
    ("input", (3, 32, 32)),
    ("conv1", (16, 32, 32)), ("relu1", (16, 32, 32)), ("pool1", (16, 16, 16)),
    ("conv2", (32, 16, 16)), ("relu2", (32, 16, 16)), ("pool2", (32, 8, 8)),
    ("flatten", (2048,)), ("fc1", (128,)), ("relu3", (128,)), ("dropout", (128,)),
    ("fc2", (10,)),
]


@pytest.fixture
def model():
    torch.manual_seed(0)
    return TinyCNN()


# ---------------------------------------------------------------- shapes
@pytest.mark.parametrize("batch", [1, 4, 128])
def test_output_shape(model, batch):
    x = torch.randn(batch, 3, 32, 32)
    assert model(x).shape == (batch, 10)


def test_layer_table_matches_docstring(model):
    rows = layer_table(model)
    got = [(r["name"], r["out_shape"]) for r in rows]
    assert got == EXPECTED_SHAPES


def test_parameter_count(model):
    assert count_parameters(model) == EXPECTED_PARAMS
    rows = {r["name"]: r["params"] for r in layer_table(model)}
    assert rows["conv1"] == 3 * 16 * 9 + 16          # 448
    assert rows["conv2"] == 16 * 32 * 9 + 32         # 4,640
    assert rows["fc1"] == 2048 * 128 + 128           # 262,272
    assert rows["fc2"] == 128 * 10 + 10              # 1,290
    assert sum(rows.values()) == EXPECTED_PARAMS


def test_conv_channels_change_shapes():
    m = TinyCNN(conv_channels=(8, 16), fc_hidden=64)
    rows = {r["name"]: r for r in layer_table(m)}
    assert rows["conv1"]["out_shape"] == (8, 32, 32)
    assert rows["pool2"]["out_shape"] == (16, 8, 8)
    assert rows["flatten"]["out_shape"] == (1024,)
    assert rows["fc1"]["params"] == 1024 * 64 + 64


# ---------------------------------------------------------------- build_model / config
def test_build_model_from_config(cfg):
    m = build_model(cfg)
    assert isinstance(m, TinyCNN)
    assert count_parameters(m) == EXPECTED_PARAMS


def test_build_model_does_not_mutate_config(cfg):
    before = dict(cfg["model"])
    build_model(cfg)
    assert cfg["model"] == before, "build_model must copy, not pop from, the config"


def test_build_model_unknown_name(cfg):
    bad = {"model": {**cfg["model"], "name": "tiny_cn"}}
    with pytest.raises(ValueError, match="unknown model"):
        build_model(bad)


def test_build_model_misspelt_key(cfg):
    bad = {"model": {**cfg["model"], "fc_hiden": 64}}
    with pytest.raises(TypeError):
        build_model(bad)


def test_registry_contains_tiny_cnn():
    assert MODELS["tiny_cnn"] is TinyCNN


# ---------------------------------------------------------------- forward_with_activations
def test_activations_order_and_keys(model):
    acts = model.forward_with_activations(torch.randn(1, 3, 32, 32))
    assert list(acts) == [n for n, _ in EXPECTED_SHAPES]


def test_conv_output_is_pre_relu(model):
    """Regression test: ReLU(inplace=True) must not overwrite the stored conv output."""
    acts = model.forward_with_activations(torch.randn(1, 3, 32, 32))
    assert acts["conv1"].min() < 0, "conv1 activation has no negatives -> recorder captured post-ReLU"
    assert acts["relu1"].min() == 0
    assert acts["conv2"].min() < 0


def test_activations_logits_match_forward(model):
    model.eval()
    x = torch.randn(2, 3, 32, 32)
    acts = model.forward_with_activations(x)
    assert torch.allclose(acts["fc2"], model(x))


def test_activations_do_not_require_grad(model):
    acts = model.forward_with_activations(torch.randn(1, 3, 32, 32))
    assert not acts["fc2"].requires_grad


# ---------------------------------------------------------------- training mechanics
def test_gradients_flow_to_every_parameter(model):
    model.train()
    x, y = torch.randn(8, 3, 32, 32), torch.randint(0, 10, (8,))
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, f"no gradient reached {name}"


def test_deterministic_under_seed():
    torch.manual_seed(123); a = TinyCNN()
    torch.manual_seed(123); b = TinyCNN()
    x = torch.randn(3, 3, 32, 32)
    assert torch.equal(a(x), b(x))


def test_dropout_only_active_in_train_mode():
    m = TinyCNN(dropout=0.5)
    x = torch.randn(4, 3, 32, 32)
    m.eval()
    assert torch.equal(m(x), m(x)), "eval mode must be deterministic"
    m.train()
    assert not torch.equal(m(x), m(x)), "train mode with dropout should vary"


def test_eval_logits_are_finite(model):
    model.eval()
    out = model(torch.randn(4, 3, 32, 32))
    assert torch.isfinite(out).all()
