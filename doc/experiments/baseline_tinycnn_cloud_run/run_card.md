# Run card — baseline_tinycnn

- date: 2026-11-01 18:24
- model: `tiny_cnn`  |  parameters: 268,650
- epochs: 15  |  optimizer: sgd  lr 0.05  momentum 0.9  wd 0.0005  scheduler cosine
- data: batch 128  normalize True  augment {'random_crop_pad': 4, 'horizontal_flip': True}

## Architecture (measured)

| layer | type | output shape | params |
|---|---|---|---|
| input | Input | 3×32×32 | 0 |
| conv1 | Conv2d | 16×32×32 | 448 |
| relu1 | ReLU | 16×32×32 | 0 |
| pool1 | MaxPool2d | 16×16×16 | 0 |
| conv2 | Conv2d | 32×16×16 | 4,640 |
| relu2 | ReLU | 32×16×16 | 0 |
| pool2 | MaxPool2d | 32×8×8 | 0 |
| flatten | Flatten | 2048 | 0 |
| fc1 | Linear | 128 | 262,272 |
| relu3 | ReLU | 128 | 0 |
| dropout | Dropout | 128 | 0 |
| fc2 | Linear | 10 | 1,290 |

## Full config

```yaml
experiment: baseline_tinycnn
seed: 42
data:
  val_per_class: 500
  batch_size: 128
  num_workers: 0
  normalize: true
  augment:
    random_crop_pad: 4
    horizontal_flip: true
model:
  name: tiny_cnn
  conv_channels:
  - 16
  - 32
  fc_hidden: 128
  dropout: 0.0
train:
  epochs: 15
  optimizer: sgd
  lr: 0.05
  momentum: 0.9
  weight_decay: 0.0005
  scheduler: cosine
  early_stopping_patience: 5
paths:
  models_dir: outputs/models
  logs_dir: outputs/logs
  figures_dir: data/output
```