#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from asca_ad.evaluator import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ASCA-AD fixed-combined evaluation by dataset name."
    )
    parser.add_argument(
        "dataset",
        help="Dataset name: SKAB, PUMP, PSM, MSL, SMAP, HAI, SMD, WADI, or all.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train-if-missing", action="store_true")
    args = parser.parse_args()

    run_evaluation(
        dataset=args.dataset,
        output_dir=args.output_dir,
        seed=args.seed,
        epochs=args.epochs,
        train_if_missing=args.train_if_missing,
    )


if __name__ == "__main__":
    main()
