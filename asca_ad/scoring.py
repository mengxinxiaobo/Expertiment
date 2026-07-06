from __future__ import annotations

import numpy as np

def total_score(loss_local, loss_global):
    return np.asarray(loss_local) + np.asarray(loss_global)

def gap_score(loss_local, loss_global):
    return np.abs(np.asarray(loss_local) - np.asarray(loss_global))

def combined_score(loss_local, loss_global, gap_weight: float = 1.0):
    local = np.asarray(loss_local)
    global_ = np.asarray(loss_global)
    return local + global_ + float(gap_weight) * np.abs(local - global_)

def percentile_threshold(train_scores, test_scores, anormly_ratio: float) -> float:
    values = np.concatenate([
        np.asarray(train_scores).reshape(-1),
        np.asarray(test_scores).reshape(-1),
    ], axis=0)
    return float(np.percentile(values, 100.0 - float(anormly_ratio)))
