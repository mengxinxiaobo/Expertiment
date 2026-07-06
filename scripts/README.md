# Scripts

Recommended entrypoint:

    python scripts/run.py SKAB

Default behavior:

    train missing checkpoint, then evaluate

Common commands:

    python scripts/run.py SKAB
    python scripts/run.py SKAB --mode eval
    python scripts/run.py SKAB --mode train-eval --force-train
    python scripts/run.py SKAB,PUMP,PSM --mode eval
    python scripts/run.py all --mode eval

Evaluation-only compatibility entrypoint:

    python scripts/evaluate.py SKAB

Short shell wrappers:

    bash scripts/eval.sh SKAB
    bash scripts/eval_all.sh

Outputs:

    results/FIXED_COMBINED/<DATASET>/
    results/PIPELINE_RUNS/
