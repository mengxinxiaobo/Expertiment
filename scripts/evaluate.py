#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_CONFIG = ROOT / "configs" / "datasets.json"
EXPERIMENT_CONFIG = ROOT / "configs" / "experiment.json"
ENGINE_PATH = ROOT / "scripts" / "run_all_fixed_combined.py"

def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))

def load_engine():
    if not ENGINE_PATH.exists():
        raise FileNotFoundError(f"Missing evaluation engine: {ENGINE_PATH}")
    spec = importlib.util.spec_from_file_location("asca_ad_eval_engine", str(ENGINE_PATH))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load engine: {ENGINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["asca_ad_eval_engine"] = module
    spec.loader.exec_module(module)
    return module

def select_datasets(name: str, config: dict) -> list[str]:
    name = name.strip().upper()
    if name == "ALL":
        return list(config.keys())
    items = [x.strip().upper() for x in name.split(",") if x.strip()]
    bad = [x for x in items if x not in config]
    if bad:
        raise ValueError(f"Unknown dataset(s): {bad}. Valid: {', '.join(config.keys())}")
    return items

def make_output_dir(raw: str, custom: str | None) -> Path:
    if custom:
        path = Path(custom)
        return path if path.is_absolute() else ROOT / path
    name = raw.strip().upper().replace(",", "_")
    return ROOT / "results" / "FIXED_COMBINED" / name

def main() -> None:
    parser = argparse.ArgumentParser(description="Unified ASCA-AD evaluator.")
    parser.add_argument("dataset", help="SKAB, PUMP, PSM, MSL, SMAP, HAI, SMD, WADI, or all")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train-if-missing", action="store_true")
    args = parser.parse_args()

    dataset_cfg = read_json(DATASET_CONFIG)
    exp_cfg = read_json(EXPERIMENT_CONFIG)
    engine = load_engine()

    datasets = select_datasets(args.dataset, dataset_cfg)
    out_dir = make_output_dir(args.dataset, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = int(args.seed) if args.seed is not None else int(exp_cfg.get("default_seed", 42))

    print("=" * 80)
    print("ASCA-AD fixed-combined evaluation")
    print("=" * 80)
    print(f"Project root : {ROOT}")
    print(f"Datasets     : {', '.join(datasets)}")
    print(f"Output dir   : {out_dir}")
    print(f"Seed         : {seed}")
    print("=" * 80)

    results = []
    failures = []

    for dataset in datasets:
        cfg = dataset_cfg[dataset]
        ratio = float(cfg["anormly_ratio"])
        epochs = int(args.epochs) if args.epochs is not None else int(cfg.get("epochs", 3))
        try:
            result = engine.evaluate_dataset(
                root=ROOT,
                dataset=dataset,
                ratio=ratio,
                output_root=out_dir,
                seed=seed,
                v4_epochs=epochs,
                train_if_missing=args.train_if_missing,
            )
            result["dataset_config"] = cfg
            results.append(result)
        except Exception as exc:
            print(f"[{dataset}] FAILED: {exc}", file=sys.stderr)
            failures.append({"dataset": dataset, "error": str(exc)})

    if results:
        engine.write_global_outputs(out_dir, results)

    if failures:
        fail_path = out_dir / "failures.json"
        fail_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit(1)

    print()
    print("Done.")
    print(f"Summary: {out_dir / 'summary.md'}")

if __name__ == "__main__":
    main()
