"""Score-only adapters for the frozen SKAB three-model benchmark.

The adapters in this package do not import test labels and do not calculate
thresholds or detection metrics.
"""

from .common import (
    LabelFreeWindowDataset,
    ScoreCollection,
    aggregate_point_scores,
    collect_point_energy,
)
from .asca_adapter import ASCAV4ScoreAdapter
from .ltfad_adapter import LTFADScoreAdapter
from .pplad_adapter import PPLADScoreAdapter

__all__ = [
    "LabelFreeWindowDataset",
    "ASCAV4ScoreAdapter",
    "LTFADScoreAdapter",
    "PPLADScoreAdapter",
    "ScoreCollection",
    "aggregate_point_scores",
    "collect_point_energy",
]
