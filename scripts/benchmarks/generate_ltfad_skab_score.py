"""
LTFAD SKAB score generator.

This script should train/load the corresponding model and save:

results/SKAB_scores/LTFAD_train_energy.npy
results/SKAB_scores/LTFAD_test_energy.npy

The evaluation stage is fixed by evaluate_pplad_protocol.py.
"""
from pathlib import Path
import numpy as np

OUT = Path("results/SKAB_scores")
OUT.mkdir(parents=True, exist_ok=True)

raise RuntimeError(
    "LTFAD score generator has to be connected to the existing model "
    "implementation before running. The evaluation protocol is ready."
)
