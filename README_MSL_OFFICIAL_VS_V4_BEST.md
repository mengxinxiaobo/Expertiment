# MSL Original Official vs V4 Best

## Protocol

- Original PPLAD: official `scripts/MSL.sh`.
- V4: current `official_k6` structure, 10 training epochs.
- V4 selection: search `gap`, `total`, and `combined` over
  `anormly_ratio=0.10..3.00` with step `0.01`, selecting maximum PA-F1.
- Resource benchmark: same MSL data, `win_size=90`, `batch_size=256`,
  same hardware and benchmark procedure.

## Run

```bash
cd /mnt/c/Users/DING/Desktop/Experiment/CODE
chmod +x run_msl_official_vs_v4_best_full.sh
bash run_msl_official_vs_v4_best_full.sh
```

## Main outputs

- `results/MSL_OFFICIAL_VS_V4_BEST/original_official_metrics.json`
- `results/MSL_OFFICIAL_VS_V4_BEST/v4_threshold_sweep.csv`
- `results/MSL_OFFICIAL_VS_V4_BEST/v4_best_run.json`
- `results/MSL_OFFICIAL_VS_V4_BEST/BENCHMARK/summary.md`
