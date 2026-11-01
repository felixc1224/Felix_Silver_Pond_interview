#!/usr/bin/env bash
# Train -> evaluate -> predict for one experiment, from the repo root.
#   bash scripts/run_experiment.sh exp03_sched_step
# Expects configs/<name>.yaml whose `experiment:` field is also <name>.
set -euo pipefail
name="${1:?usage: bash scripts/run_experiment.sh <experiment-name>}"
cfg="configs/${name}.yaml"
[ -f "$cfg" ] || { echo "no such config: $cfg"; exit 1; }
mkdir -p outputs/logs
python -u -m src.train --config "$cfg" 2>&1 | tee "outputs/logs/${name}_stdout.txt"
python -u -m src.evaluate --checkpoint "outputs/models/${name}_best.pt"
python -u -m src.predict  --checkpoint "outputs/models/${name}_best.pt"
echo
echo "done: figures in data/output/${name}/  log in outputs/logs/${name}.csv  run card in outputs/logs/${name}_config.md"
