# ASCA-AD V4-IO

This directory contains an independent, checkpoint-compatible inference
experiment. It does not replace or patch `asca_ad/`.

Phase-1 changes are deliberately limited to mathematically equivalent inference:

- compute only the frozen `total` score;
- omit Normal-CDF/area, gap, combined, probability and full-details work;
- omit the unused eval softmax in straight-through Top-k;
- normalize only `total`;
- use `torch.inference_mode()` in the optimized adapter/runners;
- cache lag/time-derived indexes as non-persistent buffers.

The normal `forward` remains training-compatible. Existing checkpoints are loaded
with `strict=True`; cached buffers are excluded from `state_dict`.

## Gated execution

The default entry first runs Validation and the complete SKAB score/Detection/
Efficiency A/B gate. It proceeds to the remaining five datasets only if every
equivalence and source-integrity gate passes:

```bash
PYTHONUNBUFFERED=1 bash scripts/benchmarks/run_asca_inference_optimization.sh
```

Useful restricted modes are `--validate-only`, `--run-skab`, `--run-all`,
`--dataset DATASET`, `--resume`, `--retry-failed`, `--old-only`, and
`--optimized-only`. No command trains or writes a checkpoint.
