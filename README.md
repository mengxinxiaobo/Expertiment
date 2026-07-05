# Adaptive Sparse Temporal Anchor Model V4

An ultra-lightweight multivariate time-series anomaly detection model based on
adaptive sparse temporal anchors, differentiable Top-k selection, and a shared
Gaussian fitting architecture.

## Repository status

The verified V4 implementation is currently stored in:

```text
src/legacy/main_adaptive_sparse_anchor_v4.py
```

Use the unified entry point:

```bash
python main.py --help
```

or:

```bash
bash scripts/run_v4.sh --help
```

## Main directories

- `src/`: model, training, evaluation, data, and utility code
- `preprocess/`: dataset conversion and validation
- `configs/`: experiment configurations
- `scripts/`: launch scripts
- `dataset/`: local processed data
- `results/`: compact experiment summaries
- `checkpoints/`: local model weights
- `logs/`: local training and evaluation logs
- `tests/`: unit and regression tests
- `docs/`: model and protocol documentation
