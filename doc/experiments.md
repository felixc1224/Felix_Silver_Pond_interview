# Experiments log

One row per training run. The exact settings of every run are in its run card
(`outputs/logs/<experiment>_config.md`, also appended to `data/output/<experiment>/<split>/summary.md`),
so this table only needs the headline and *what changed vs. the previous row*.

Workflow for a new experiment:
1. copy `configs/baseline.yaml` → `configs/<name>.yaml`, change `experiment:` and the knobs
2. `python -m src.train --config configs/<name>.yaml`
3. `python -m src.evaluate --checkpoint outputs/models/<name>_best.pt`
4. copy the figures you want to keep into `doc/experiments/<name>/`, add a row below
5. if the architecture changed: update the shape table at the top of `src/model.py` (see REVIEW_GUIDE reminders)

| # | experiment | what changed | params | epochs | best val acc | top-2 | macro-F1 | epoch time | notes / figures |
|---|---|---|---|---|---|---|---|---|---|
| 0 | baseline_tinycnn (cloud reference run by Claude) | — | 268,650 | 15 | 76.9% | 89.9% | 0.768 | ~9 s (2 cores) | `doc/experiments/baseline_tinycnn_cloud_run/`; cat↔dog and auto→truck dominate the errors; curves still rising at ep 15 |
| 1 | baseline_tinycnn (my laptop) | — | 268,650 | 15 | | | | | |
