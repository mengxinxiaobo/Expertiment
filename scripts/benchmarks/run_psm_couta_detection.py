#!/usr/bin/env python3
"""Formal, fixed-protocol COUTA experiment on PSM.

The official DeepOD COUTA fit/decision_function methods are used unchanged.
This external runner enforces split access, train-only scaling and fixed evaluation.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import platform
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from asca_ad.model import AdaptiveSparseAnchorSolverV4
from scripts.benchmarks.adapters.couta_dataset_adapter import (
    PSMCOUTADataAdapter, RAY_AUDIT, construct_official_couta,
    load_couta_config, make_bundle, restore_from_bundle, sha256,
)

OUT = ROOT / "results" / "PSM_COUTA_RESULTS"
DET = OUT / "Detection"
SCORES = OUT / "scores"
CKPTS = OUT / "checkpoints"
BUNDLE = CKPTS / "COUTA_PSM_bundle.pt"
STATE = CKPTS / "COUTA_PSM_state_dict.pt"
TRAIN_SCORE = SCORES / "train_score.npy"
TEST_SCORE = SCORES / "test_score.npy"
TRAIN_AUDIT = OUT / "training_audit.json"
SCORE_AUDIT = OUT / "score_audit.json"
DATASET_NAME = "PSM"


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def stage(title: str):
    print("=" * 72, flush=True)
    print(title, flush=True)
    print(f"Start Time: {now()}", flush=True)
    return time.perf_counter()


def finish(title: str, started: float):
    elapsed = time.perf_counter() - started
    print(f"{title} Finished", flush=True)
    print(f"End Time: {now()}", flush=True)
    print(f"Elapsed Time: {elapsed:.3f}s", flush=True)
    return elapsed


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EpochTimingWriter:
    """Annotate official epoch log lines without changing the training loop."""
    def __init__(self, stream, epochs: int):
        self.stream=stream; self.epochs=epochs; self.buffer=""; self.started=time.perf_counter(); self.previous=self.started
    def write(self, value):
        self.buffer+=value
        while "\n" in self.buffer:
            line,self.buffer=self.buffer.split("\n",1); self.stream.write(line+"\n")
            match=re.search(r"epoch:\s*(\d+)",line)
            if match:
                current=time.perf_counter(); epoch=int(match.group(1)); epoch_time=current-self.previous; elapsed=current-self.started
                eta=(elapsed/max(epoch,1))*(self.epochs-epoch); self.previous=current
                self.stream.write(f"[epoch-time] Epoch={epoch}/{self.epochs} EpochTime={epoch_time:.3f}s "
                                  f"TotalElapsed={elapsed:.3f}s EstimatedRemaining={eta:.3f}s\n")
        return len(value)
    def flush(self):
        if self.buffer: self.stream.write(self.buffer); self.buffer=""
        self.stream.flush()


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def train(config: dict, device: str, overwrite: bool) -> None:
    if BUNDLE.exists() and not overwrite:
        raise FileExistsError(f"Formal checkpoint exists; refusing overwrite: {BUNDLE}")
    started = stage(f"Stage A: Official COUTA fit on {DATASET_NAME} train only")
    adapter = PSMCOUTADataAdapter("train")
    raw_train = adapter.load_train()
    scaler, scaled_train = adapter.fit_train_scaler(raw_train)
    adapter.assert_training_isolated()
    model = construct_official_couta(config, device)
    bypass_internal = bool(config["model_config"].get("bypass_unused_internal_fit_scoring", False))
    original_decision_function = model.decision_function
    if bypass_internal:
        # COUTA.fit uses this only to create its contamination-based threshold,
        # which the fixed paper protocol forbids. Avoid materializing the full
        # stride-1 HAI train window tensor; restore official scoring immediately.
        model.decision_function = lambda x, *args, **kwargs: np.zeros(len(x), dtype=np.float64)
    try:
        with contextlib.redirect_stdout(EpochTimingWriter(sys.stdout, int(config["model_config"]["epochs"]))):
            model.fit(scaled_train)
    finally:
        model.decision_function = original_decision_function
    adapter.assert_training_isolated()
    params = sum(p.numel() for p in model.net.parameters())
    if params != config["model_config"]["expected_parameters"]:
        raise RuntimeError(f"Unexpected COUTA parameter count: {params}")
    CKPTS.mkdir(parents=True, exist_ok=True)
    torch.save(make_bundle(model, scaler, config), BUNDLE)
    torch.save(model.net.state_dict(), STATE)
    elapsed = finish("Stage A", started)
    save_json(TRAIN_AUDIT, {
        **adapter.audit(), "training_seconds": elapsed, "parameters": params,
        "checkpoint": str(BUNDLE.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": sha256(BUNDLE), "state_dict_sha256": sha256(STATE),
        "training_test_access": False, "training_test_label_access": False,
        "unused_internal_fit_scoring_bypassed": bypass_internal,
        "internal_threshold_used_for_paper": False,
        "ray_tune_used": False, "training_ray_used": False,
    })


def official_scores_chunked(model, data: np.ndarray, chunk_points: int, split: str) -> np.ndarray:
    """Call official decision_function on overlapping chunks with exact endpoint alignment."""
    seq_len=int(model.seq_len); total=len(data); pieces=[]; endpoint_start=0; chunk_index=0
    total_chunks=(total + chunk_points - 1)//chunk_points
    started=time.perf_counter()
    while endpoint_start < total:
        endpoint_stop=min(total, endpoint_start+chunk_points)
        input_start=0 if endpoint_start==0 else endpoint_start-(seq_len-1)
        chunk=np.ascontiguousarray(data[input_start:endpoint_stop])
        chunk_score=np.asarray(model.decision_function(chunk),dtype=np.float64)
        selected=chunk_score if endpoint_start==0 else chunk_score[seq_len-1:]
        pieces.append(selected); chunk_index+=1
        elapsed=time.perf_counter()-started
        eta=(elapsed/chunk_index)*(total_chunks-chunk_index)
        print(f"[score-progress] split={split} chunk={chunk_index}/{total_chunks} "
              f"timeline={endpoint_stop}/{total} windows={max(0,endpoint_stop-seq_len+1)}/"
              f"{max(0,total-seq_len+1)} elapsed={elapsed:.3f}s eta={eta:.3f}s",flush=True)
        endpoint_start=endpoint_stop
    result=np.concatenate(pieces)
    if result.shape!=(total,): raise RuntimeError(f"Chunked score alignment failed: {result.shape}/{total}")
    return result


def score(config: dict, device: str, overwrite: bool) -> None:
    if not BUNDLE.exists():
        raise FileNotFoundError(f"Run training first: {BUNDLE}")
    if (TRAIN_SCORE.exists() or TEST_SCORE.exists()) and not overwrite:
        raise FileExistsError("Score artifacts exist; use --overwrite only for an intentional rerun")
    started = stage("Stage B: Official COUTA label-free decision_function")
    adapter = PSMCOUTADataAdapter("score")
    train_x, test_x = adapter.load_train(), adapter.load_test()
    bundle = torch.load(BUNDLE, map_location=device, weights_only=False)
    model, scaler = restore_from_bundle(bundle, device)
    train_x = np.asarray(scaler.transform(train_x), dtype=np.float32)
    test_x = np.asarray(scaler.transform(test_x), dtype=np.float32)
    expected_train = int(config["expected_shapes"]["train"][0])
    expected_test = int(config["expected_shapes"]["test"][0])
    train_started=time.perf_counter()
    chunk_points=int(config["model_config"].get("score_chunk_points",0))
    train_score = (official_scores_chunked(model,train_x,chunk_points,"train") if chunk_points
                   else np.asarray(model.decision_function(train_x), dtype=np.float64))
    train_score_seconds=time.perf_counter()-train_started
    print(f"[score] split=train windows={expected_train-config['model_config']['seq_len']+1} "
          f"elapsed={train_score_seconds:.3f}s",flush=True)
    test_started=time.perf_counter()
    test_score = (official_scores_chunked(model,test_x,chunk_points,"test") if chunk_points
                  else np.asarray(model.decision_function(test_x), dtype=np.float64))
    test_score_seconds=time.perf_counter()-test_started
    print(f"[score] split=test windows={expected_test-config['model_config']['seq_len']+1} "
          f"elapsed={test_score_seconds:.3f}s",flush=True)
    if train_score.shape != (expected_train,) or test_score.shape != (expected_test,):
        raise RuntimeError(f"Unexpected scores: {train_score.shape}, {test_score.shape}")
    if not np.isfinite(train_score).all() or not np.isfinite(test_score).all():
        raise RuntimeError("COUTA produced non-finite scores")
    prefix = config["evaluation"]["prefix_padding_points"]
    if not (np.all(train_score[:prefix] == 0) and np.all(test_score[:prefix] == 0)):
        raise RuntimeError("Official COUTA prefix-padding semantics changed")
    SCORES.mkdir(parents=True, exist_ok=True)
    np.save(TRAIN_SCORE, train_score); np.save(TEST_SCORE, test_score)
    elapsed = finish("Stage B", started)
    save_json(SCORE_AUDIT, {
        **adapter.audit(), "score_seconds": elapsed,
        "train_score_seconds": train_score_seconds,
        "test_score_seconds": test_score_seconds,
        "train_score_shape": list(train_score.shape),
        "test_score_shape": list(test_score.shape),
        "score_label_access": False, "official_decision_function": True,
        "score_chunk_points": chunk_points or None,
        "chunk_overlap_points": config["model_config"]["seq_len"]-1 if chunk_points else 0,
        "prefix_padding_points": prefix,
    })


def metrics(y, p):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, p, average="binary", zero_division=0)
    return {"Accuracy": float(accuracy_score(y, p)), "Precision": float(precision),
            "Recall": float(recall), "F1": float(f1)}


def evaluate(config: dict) -> None:
    if not (BUNDLE.exists() and TRAIN_SCORE.exists() and TEST_SCORE.exists()):
        raise FileNotFoundError("Checkpoint and both score files are required")
    started = stage("Stage C: Fixed percentile evaluation")
    train_score = np.load(TRAIN_SCORE, allow_pickle=False)
    test_score = np.load(TEST_SCORE, allow_pickle=False)
    adapter = PSMCOUTADataAdapter("evaluate")
    label = adapter.load_label()
    if test_score.shape != label.shape:
        raise RuntimeError(f"Score/label mismatch: {test_score.shape}/{label.shape}")
    percentile = float(config["evaluation"]["percentile"])
    threshold = float(np.percentile(np.concatenate([train_score, test_score]), percentile))
    raw_pred = (test_score > threshold).astype(np.int64)
    pa_pred = AdaptiveSparseAnchorSolverV4._point_adjust(raw_pred, label)
    raw, pa = metrics(label, raw_pred), metrics(label, pa_pred)
    row = {"Model": "COUTA", **raw,
           "PA-Accuracy": pa["Accuracy"], "PA-Precision": pa["Precision"],
           "PA-Recall": pa["Recall"], "PA-F1": pa["F1"], "Threshold": threshold}
    DET.mkdir(parents=True, exist_ok=True)
    with (DET / "comparison_detection.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row)); writer.writeheader(); writer.writerow(row)
    save_json(DET / "comparison_detection.json", row)
    training_audit = json.loads(TRAIN_AUDIT.read_text(encoding="utf-8"))
    score_audit = json.loads(SCORE_AUDIT.read_text(encoding="utf-8"))
    protocol = {
        "dataset": DATASET_NAME, "model": "COUTA", "seed": 42,
        "config": config, "checkpoint_path": str(BUNDLE.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": sha256(BUNDLE), "threshold": threshold,
        "evaluation_points": int(label.size), "training_audit": training_audit,
        "score_audit": score_audit, "evaluation_file_access": adapter.audit(),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "cuda": torch.version.cuda},
        "audit": {"training_test_access": False, "training_test_label_access": False,
                  "score_label_access": False, "ray_tune_used": False,
                  "training_ray_used": False, "score_search": False,
                  "ratio_search": False, "threshold_search": False,
                  "parameter_search": False, "oracle_search": False,
                  "best_f1_search": False, "model_source_modified": False},
    }
    save_json(OUT / "protocol.json", protocol)
    finish("Stage C", started)
    print(f"RAW Precision={raw['Precision']:.6f} Recall={raw['Recall']:.6f} F1={raw['F1']:.6f}")
    print(f"PA Accuracy={pa['Accuracy']:.6f} Precision={pa['Precision']:.6f} Recall={pa['Recall']:.6f} F1={pa['F1']:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("all", "train", "score", "evaluate"), default="all")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    pipeline_started_at = now()
    pipeline_started = time.perf_counter()
    config = load_couta_config(); seed_all(config["seed"])
    if not torch.cuda.is_available(): raise RuntimeError("Formal COUTA protocol requires CUDA")
    device = "cuda:0"
    if args.stage in {"all", "train"}: train(config, device, args.overwrite)
    if args.stage in {"all", "score"}: score(config, device, args.overwrite)
    if args.stage in {"all", "evaluate"}: evaluate(config)
    if args.stage == "all":
        protocol_path = OUT / "protocol.json"
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        protocol["formal_pipeline_start_time"] = pipeline_started_at
        protocol["formal_pipeline_end_time"] = now()
        protocol["formal_pipeline_seconds"] = time.perf_counter() - pipeline_started
        save_json(protocol_path, protocol)


if __name__ == "__main__":
    main()
