#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from asca_ad.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ASCA-AD training and evaluation from one script."
    )
    parser.add_argument(
        "dataset",
        help="Dataset name: SKAB, PUMP, PSM, MSL, SMAP, HAI, SMD, WADI, or all.",
    )
    parser.add_argument(
        "--mode",
        choices=["eval", "train", "train-eval"],
        default="train-eval",
        help="Default: train-eval. Existing checkpoints are reused unless --force-train is set.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Retrain even when a checkpoint already exists.",
    )
    parser.add_argument(
        "--no-train-if-missing",
        action="store_true",
        help="In eval/train-eval mode, fail instead of training when checkpoint is missing.",
    )
    args = parser.parse_args()

    run_pipeline(
        dataset=args.dataset,
        mode=args.mode,
        output_dir=args.output_dir,
        seed=args.seed,
        epochs=args.epochs,
        train_if_missing=not args.no_train_if_missing,
        force_train=args.force_train,
    )


if __name__ == "__main__":
    main()
