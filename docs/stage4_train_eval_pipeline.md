# Stage 4: single-script train and evaluate pipeline

## Main command

Run one dataset:

    python scripts/run.py SKAB

Run evaluation only:

    python scripts/run.py SKAB --mode eval

Train missing checkpoint and then evaluate:

    python scripts/run.py SKAB --mode train-eval

Force retraining and then evaluate:

    python scripts/run.py SKAB --mode train-eval --force-train

Run multiple datasets:

    python scripts/run.py SKAB,PUMP,PSM --mode eval

Run all datasets:

    python scripts/run.py all --mode eval

## Code structure

The command-line layer is thin:

    scripts/run.py

The pipeline logic is in:

    asca_ad/pipeline.py

Training logic is in:

    asca_ad/trainer.py

Evaluation logic is in:

    asca_ad/evaluator.py

Dataset and experiment configuration is in:

    configs/datasets.json
    configs/experiment.json

Model architecture remains in:

    asca_ad/model.py

This stage does not change the ASCA-AD model algorithm. It only improves the
project structure and provides a single entrypoint for training and evaluation.
