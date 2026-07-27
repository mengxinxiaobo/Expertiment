#!/usr/bin/env python3
"""Checkpoint/state/source validation for ASCA-AD V4-IO (no training)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(ROOT_BOOTSTRAP))

from scripts.benchmarks.asca_optimization_common import (
    DATASETS, OUT, checkpoint_compatibility, protected_snapshot, require_cuda, save_json, set_seed,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, action="append")
    args = parser.parse_args()
    datasets = tuple(args.dataset or ("SKAB",))
    set_seed()
    device = require_cuda()
    before = protected_snapshot()
    results = []
    for dataset in datasets:
        result = checkpoint_compatibility(dataset, device)
        results.append(result)
        save_json(OUT / dataset / "checkpoint_compatibility.json", result)
        print(
            f"[{dataset}] checkpoint={result['status']} params="
            f"{result['old_parameters']}/{result['optimized_parameters']} "
            f"keys_equal={result['state_dict_keys_equal']}", flush=True,
        )
    after = protected_snapshot()
    source_equal = before == after
    payload = {
        "status": "PASS" if source_equal and all(r["status"] == "PASS" for r in results) else "FAIL",
        "datasets": results,
        "source_integrity_before": before,
        "source_integrity_after": after,
        "existing_protected_artifacts_unchanged": source_equal,
        "training": False,
        "checkpoint_written": False,
        "old_formal_call_chain": [
            "formal data loader", "formal window batch", "ASCAV4ScoreAdapter",
            "solver._forward_batch", "AdaptiveSparseAnchorCompetitiveModelV4.forward",
            "solver._score_dict(gap,total,combined)", "total score",
        ],
        "optimized_call_chain": [
            "same formal data loader", "same window batch", "ASCAOptimizedScoreAdapter",
            "ASCASolverInferenceOptimized.forward_total_score",
            "ASCAInferenceOptimized.forward_total_score_only", "total-only normalization",
            "total score",
        ],
    }
    save_json(OUT / "validation" / "validation.json", payload)
    save_json(OUT / "checkpoint_compatibility" / "checkpoint_compatibility.json", payload)
    if payload["status"] != "PASS":
        raise SystemExit("ASCA V4-IO validation FAILED")
    print(f"validation=PASS artifact={OUT / 'validation' / 'validation.json'}", flush=True)


if __name__ == "__main__":
    main()
