# baseline_tinycnn — val set

- checkpoint: `baseline_tinycnn_best.pt` (epoch 15)
- images: 5000
- **accuracy: 76.92%**  |  top-2: 89.94%  |  macro-F1: 0.768  |  ECE: 0.015

| class | accuracy | precision | F1 | n |
|---|---|---|---|---|
| airplane | 80.8% | 77.5% | 0.791 | 500 |
| automobile | 87.4% | 87.8% | 0.876 | 500 |
| bird | 64.4% | 67.5% | 0.659 | 500 |
| cat | 56.8% | 61.7% | 0.592 | 500 |
| deer | 73.6% | 75.7% | 0.746 | 500 |
| dog | 64.8% | 63.3% | 0.640 | 500 |
| frog | 82.2% | 80.9% | 0.815 | 500 |
| horse | 84.0% | 82.2% | 0.831 | 500 |
| ship | 87.6% | 85.7% | 0.866 | 500 |
| truck | 87.6% | 84.9% | 0.862 | 500 |

Most confused pairs (true → predicted): cat→dog (99), dog→cat (75), automobile→truck (42)

Figures: calibration.png, confusion_matrix.png, learning_curves.png, per_class_accuracy.png, worst_mistakes.png