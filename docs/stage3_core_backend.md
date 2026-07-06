# Stage 3 core backend

The public CLI is now thin:

    python scripts/evaluate.py SKAB

The evaluation logic lives in:

    asca_ad/evaluator.py

Dataset configuration helpers live in:

    asca_ad/config.py
    asca_ad/runner.py
    data_factory/registry.py

The current backend still uses the validated fixed-combined engine:

    scripts/run_all_fixed_combined.py
    scripts/dataset_runners/

This is intentional for compatibility. After all datasets are verified, the legacy
dataset runners can be merged into a fully unified backend.
