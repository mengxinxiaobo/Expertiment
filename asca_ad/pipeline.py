from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT
from .evaluator import run_evaluation
from .trainer import train_datasets


VALID_MODES = {"eval", "train", "train-eval"}


def run_pipeline(
    dataset: str,
    mode: str = "train-eval",
    output_dir: str | None = None,
    seed: int | None = None,
    epochs: int | None = None,
    train_if_missing: bool = True,
    force_train: bool = False,
) -> dict[str, Any]:
    """Run ASCA-AD from a single entrypoint.

    Modes:
    - eval: evaluate existing checkpoint only.
    - train: train missing checkpoints; use --force-train to retrain existing ones.
    - train-eval: train missing checkpoints, then evaluate.
    """
    mode = mode.strip().lower()
    if mode not in VALID_MODES:
        valid = ", ".join(sorted(VALID_MODES))
        raise ValueError(f"Unknown mode: {mode}. Valid modes: {valid}")

    train_results: list[dict[str, Any]] = []
    eval_results: list[dict[str, Any]] = []

    if mode in {"train", "train-eval"}:
        train_results = train_datasets(
            dataset=dataset,
            seed=seed,
            epochs=epochs,
            force=force_train,
        )

    if mode in {"eval", "train-eval"}:
        eval_results = run_evaluation(
            dataset=dataset,
            output_dir=output_dir,
            seed=seed,
            epochs=epochs,
            train_if_missing=train_if_missing,
        )

    summary = {
        "dataset": dataset,
        "mode": mode,
        "force_train": bool(force_train),
        "train_results": train_results,
        "eval_results": eval_results,
    }

    pipeline_dir = PROJECT_ROOT / "results" / "PIPELINE_RUNS"
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    safe_name = dataset.strip().upper().replace(",", "_")
    summary_path = pipeline_dir / f"{safe_name}_{mode}.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("Pipeline summary:")
    print(summary_path)

    return summary
