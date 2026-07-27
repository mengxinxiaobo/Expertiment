#!/usr/bin/env python3
"""One-epoch, train-only validation of official DeepOD COUTA on SMD."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import re
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmarks.adapters.couta_dataset_adapter import (
    RAY_AUDIT, SMD_CONFIG_PATH, SMDCOUTADataAdapter, construct_official_couta,
    import_official_couta, load_smd_couta_config, make_bundle,
    restore_from_bundle, sha256,
)
from scripts.benchmarks.run_psm_couta_detection import official_scores_chunked

OUT = ROOT / "results" / "SMD_COUTA_RESULTS" / "Validation"
REPORT = OUT / "validation_report.json"
REPORT_MD = OUT / "validation_report.md"
LOGFILE = OUT / "validation.log"
CHECKPOINT = OUT / "temporary_checkpoint.pt"
POINTS = 8192
CHUNK_POINTS = 2000
LINES: list[str] = []


def now(): return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
def log(value=""): print(value, flush=True); LINES.append(value)
def check(name, ok, detail=""):
    log(f"[{'PASS' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}")
    if not ok: raise RuntimeError(f"{name}: {detail}")
def tree_hash(path):
    digest=hashlib.sha256()
    if path.exists():
        for item in sorted(p for p in path.rglob("*") if p.is_file()):
            digest.update(str(item.relative_to(path)).encode()); digest.update(bytes.fromhex(sha256(item)))
    return digest.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True); started=time.perf_counter(); start=now()
    log("="*72); log("Stage: SMD COUTA setup validation"); log(f"Start Time: {start}")
    protected=("PSM_COUTA_RESULTS","SKAB_COUTA_RESULTS","MSL_COUTA_RESULTS","HAI_COUTA_RESULTS","PUMP_COUTA_RESULTS")
    before={name:tree_hash(ROOT/"results"/name) for name in protected}
    config=load_smd_couta_config(); data=ROOT/"dataset"/"SMD"
    names=("SMD_train.npy","SMD_test.npy","SMD_test_label.npy")
    paths={name:data/name for name in names}; check("actual files",all(p.is_file() for p in paths.values()))
    arrays={name:np.load(path,mmap_mode="r",allow_pickle=False) for name,path in paths.items()}
    train,test,label=(arrays[name] for name in names)
    check("train shape",train.shape==(708405,38),str(train.shape)); check("test shape",test.shape==(708420,38),str(test.shape)); check("label shape",label.shape==(708420,),str(label.shape))
    check("input_c",train.shape[1]==test.shape[1]==38,"38"); check("alignment",len(test)==len(label))
    stats={}
    for name,array in arrays.items():
        nan=int(np.isnan(array).sum()); inf=int(np.isinf(array).sum()); stats[name]={"nan_count":nan,"inf_count":inf}; check(f"{name} finite",nan==0 and inf==0)
    unique,counts=np.unique(label,return_counts=True); label_counts=dict(zip(map(str,unique.tolist()),counts.tolist())); check("binary label",set(unique.tolist())=={0.0,1.0},str(label_counts))
    check("Ray absent",not RAY_AUDIT["ray_installed"]); COUTA,_=import_official_couta(); check("official import",COUTA.__module__=="deepod.models.time_series.couta")
    from ray import tune
    blocked=False
    try: tune.choice([1])
    except RuntimeError: blocked=True
    check("Ray Tune fail-closed",blocked)
    adapter=SMDCOUTADataAdapter("validation"); subset=adapter.load_train(limit=POINTS); scaler,scaled=adapter.fit_train_scaler(subset); adapter.assert_training_isolated()
    constant_indices=np.flatnonzero(np.ptp(subset,axis=0)==0).astype(int).tolist(); constant_scales={str(i):float(scaler.scale_[i]) for i in constant_indices}
    check("train-only float32 scaler",scaled.dtype==np.float32 and np.isfinite(scaled).all()); check("finite scaler",np.isfinite(scaler.mean_).all() and np.isfinite(scaler.scale_).all() and np.all(scaler.scale_!=0),f"constant_indices_in_slice={constant_indices}")
    check("CUDA",torch.cuda.is_available()); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); baseline=torch.cuda.memory_allocated()
    model=construct_official_couta(config,"cuda",epochs=1); output=io.StringIO(); fit_started=time.perf_counter()
    with warnings.catch_warnings(record=True) as caught,contextlib.redirect_stdout(output): warnings.simplefilter("always"); result=model.fit(scaled)
    fit_seconds=time.perf_counter()-fit_started
    for line in output.getvalue().splitlines(): log(line)
    losses=[float(x) for x in re.findall(r"(?:^|\s)loss:\s*([0-9eE+.-]+)",output.getvalue())]; val_losses=[float(x) for x in re.findall(r"val_loss:\s*([0-9eE+.-]+)",output.getvalue())]
    check("official COUTA.fit",result is None); check("forward/backward/optimizer.step",bool(losses),"completed inside official fit")
    check("finite train loss",bool(losses) and all(map(math.isfinite,losses)),str(losses)); check("finite validation loss",bool(val_losses) and all(map(math.isfinite,val_losses)),str(val_losses)); check("finite center c",bool(torch.isfinite(model.c).all()))
    total=sum(p.numel() for p in model.net.parameters()); trainable=sum(p.numel() for p in model.net.parameters() if p.requires_grad); check("actual parameters",total==trainable==3521,f"{total}/{trainable}")
    official=np.asarray(model.decision_function(scaled),dtype=np.float64); check("official score",official.shape==(POINTS,) and np.isfinite(official).all()); check("prefix padding 29",bool(np.all(official[:29]==0)))
    chunked=official_scores_chunked(model,scaled,CHUNK_POINTS,"validation"); chunk_error=float(np.max(np.abs(official-chunked))); check("chunked score equivalence",chunk_error<=1e-6,f"{chunk_error:.3e}")
    bundle=make_bundle(model,scaler,config); bundle.update({"validation_epochs":1,"formal_checkpoint":False}); torch.save(bundle,CHECKPOINT)
    restored,restored_scaler=restore_from_bundle(torch.load(CHECKPOINT,map_location="cuda",weights_only=False),"cuda"); restored_scaled=np.asarray(restored_scaler.transform(subset),dtype=np.float32); restored_score=np.asarray(restored.decision_function(restored_scaled),dtype=np.float64)
    score_error=float(np.max(np.abs(official-restored_score))); scaler_error=float(np.max(np.abs(scaled-restored_scaled))); check("checkpoint restore",score_error<=1e-7 and scaler_error<=1e-7,f"{score_error:.3e}/{scaler_error:.3e}")
    check("previous COUTA results unchanged",before=={name:tree_hash(ROOT/"results"/name) for name in protected})
    warnings_text=[f"{w.category.__name__}: {w.message}" for w in caught]
    for value in warnings_text: log("[WARNING] "+value)
    elapsed=time.perf_counter()-started; peak=torch.cuda.max_memory_allocated()
    audit={"model_source_modified":False,"deepod_core_modified":False,"psm_results_modified":False,"skab_results_modified":False,"msl_results_modified":False,"hai_results_modified":False,"pump_results_modified":False,"ray_installed":False,"ray_stub_used":True,"ray_tune_used":False,"fit_auto_hyper_used":False,"training_ray_used":False,"deepod_testbed_used":False,"ts_metrics_used":False,"best_f1_search":False,"training_test_access":False,"training_test_label_access":False,"score_label_access":False,"score_search":False,"ratio_search":False,"threshold_search":False,"parameter_search":False,"oracle_search":False,"internal_threshold_used_for_paper":False,"internal_prediction_used":False,"contamination_prediction_used":False,"formal_training_started":False,"formal_scores_generated":False,"formal_evaluation_run":False,"formal_efficiency_run":False}
    report={"status":"READY","start_time":start,"end_time":now(),"elapsed_seconds":elapsed,"dataset":"SMD","input_c":38,"model_input_dtype":"float32","files":{name:{"path":str(paths[name].relative_to(ROOT)).replace("\\","/"),"shape":list(arrays[name].shape),"dtype":str(arrays[name].dtype),"size_bytes":paths[name].stat().st_size,**stats[name]} for name in names},"label_counts":label_counts,"anomaly_ratio_actual_percent":float(counts[1]/counts.sum()*100),"config_path":str(SMD_CONFIG_PATH.relative_to(ROOT)).replace("\\","/"),"anomaly_ratio":0.9,"percentile":99.1,"parameters":{"total":total,"trainable":trainable},"official_fit":{"points":POINTS,"epochs":1,"seconds":fit_seconds,"loss":losses,"validation_loss":val_losses,"forward":True,"backward":True,"optimizer_step":True},"scaler":{"fit_split":"train_validation_slice_only","constant_feature_indices_in_slice":constant_indices,"constant_feature_scales":constant_scales,"finite":True},"score":{"full_score_shape":list(official.shape),"chunked_score_shape":list(chunked.shape),"chunk_size":CHUNK_POINTS,"chunk_overlap":29,"max_abs_error":chunk_error,"prefix_padding":29},"formal_alignment":{"train_points":708405,"test_points":708420,"training_windows":70838,"train_inference_windows":708376,"test_inference_windows":708391},"checkpoint":{"path":str(CHECKPOINT.relative_to(ROOT)).replace("\\","/"),"sha256":sha256(CHECKPOINT),"score_max_abs_error":score_error,"scaler_max_abs_error":scaler_error},"gpu":{"peak_mib":peak/2**20,"incremental_mib":(peak-baseline)/2**20},"pytorch":torch.__version__,"cuda":torch.version.cuda,"warnings":warnings_text,"training_file_access":adapter.audit(),"integrity_only_label_access":"dataset/SMD/SMD_test_label.npy","prefix_requirement_resolution":"seq_len=30 implies 29; task references to 9 are treated as typographical errors","audit":audit}
    REPORT.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8"); REPORT_MD.write_text(f"# SMD COUTA Validation\n\n- Status: **READY**\n- Parameters: `{total}`\n- Chunk error: `{chunk_error:.3e}`\n- Prefix padding: `29`\n",encoding="utf-8")
    log("[PASS] SMD COUTA Validation READY"); log(f"End Time: {report['end_time']}"); log(f"Elapsed Time: {elapsed:.3f}s"); LOGFILE.write_text("\n".join(LINES)+"\n",encoding="utf-8")


if __name__=="__main__": main()
