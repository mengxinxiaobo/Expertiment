from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from .config import (
    PROJECT_ROOT,
    load_dataset_config,
    load_experiment_config,
    make_output_dir,
    select_datasets,
)


ENGINE_PATH = PROJECT_ROOT / "scripts" / "run_all_fixed_combined.py"


def load_fixed_combined_engine():
    """Load the existing fixed-combined evaluation engine.

    This keeps backward compatibility with the current validated implementation
    while moving the public evaluation API into the asca_ad package.
    """
    if not ENGINE_PATH.exists():
        raise FileNotFoundError(f"Missing evaluation engine: {ENGINE_PATH}")

    spec = importlib.util.spec_from_file_location(
        "asca_ad_fixed_combined_engine",
        str(ENGINE_PATH),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import evaluation engine: {ENGINE_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["asca_ad_fixed_combined_engine"] = module
    spec.loader.exec_module(module)
    return module


def run_evaluation(
    dataset: str,
    output_dir: str | None = None,
    seed: int | None = None,
    epochs: int | None = None,
    train_if_missing: bool = False,
) -> list[dict[str, Any]]:
    dataset_config = load_dataset_config()
    experiment_config = load_experiment_config()
    engine = load_fixed_combined_engine()

    datasets = select_datasets(dataset, dataset_config)
    out_dir = make_output_dir(dataset, output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    final_seed = (
        int(seed)
        if seed is not None
        else int(experiment_config.get("default_seed", 42))
    )

    print("=" * 80)
    print("ASCA-AD fixed-combined evaluation")
    print("=" * 80)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Datasets     : {', '.join(datasets)}")
    print(f"Output dir   : {out_dir}")
    print(f"Seed         : {final_seed}")
    print("=" * 80)

    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for item in datasets:
        cfg = dataset_config[item]
        ratio = float(cfg["anormly_ratio"])
        final_epochs = int(epochs) if epochs is not None else int(cfg.get("epochs", 3))

        try:
            result = engine.evaluate_dataset(
                root=PROJECT_ROOT,
                dataset=item,
                ratio=ratio,
                output_root=out_dir,
                seed=final_seed,
                v4_epochs=final_epochs,
                train_if_missing=train_if_missing,
            )
            result["dataset_config"] = cfg
            results.append(result)
        except Exception as exc:
            print(f"[{item}] FAILED: {exc}", file=sys.stderr)
            failures.append({"dataset": item, "error": str(exc)})

    if results:
        engine.write_global_outputs(out_dir, results)

    if failures:
        failure_path = out_dir / "failures.json"
        failure_path.write_text(
            json.dumps(failures, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Some datasets failed. See: {failure_path}", file=sys.stderr)
        raise SystemExit(1)

    print()
    print("Done.")
    print(f"Summary: {out_dir / 'summary.md'}")
    return results
