# Scripts

Unified experiment entry points.

Single dataset:

    python scripts/evaluate.py SKAB
    python scripts/evaluate.py PUMP
    python scripts/evaluate.py MSL

Short shell wrapper:

    bash scripts/eval.sh SKAB

All datasets:

    python scripts/evaluate.py all
    bash scripts/eval_all.sh

Default output path:

    results/FIXED_COMBINED/<DATASET>/

Dataset-specific parameters are stored in:

    configs/datasets.json
    configs/experiment.json
