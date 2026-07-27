#!/usr/bin/env python3
"""SMD binding, aggregation and post-evaluation diagnostics for official COUTA."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.benchmarks.run_psm_couta_detection as runner
from scripts.benchmarks.adapters.couta_dataset_adapter import (
    SMDCOUTADataAdapter, load_smd_couta_config,
)

runner.DATASET_NAME="SMD"; runner.PSMCOUTADataAdapter=SMDCOUTADataAdapter; runner.load_couta_config=load_smd_couta_config
runner.OUT=ROOT/"results"/"SMD_COUTA_RESULTS"; runner.DET=runner.OUT/"Detection"; runner.SCORES=runner.OUT/"scores"; runner.CKPTS=runner.OUT/"checkpoints"
runner.BUNDLE=runner.CKPTS/"COUTA_SMD_bundle.pt"; runner.STATE=runner.CKPTS/"COUTA_SMD_state_dict.pt"; runner.TRAIN_SCORE=runner.SCORES/"train_score.npy"; runner.TEST_SCORE=runner.SCORES/"test_score.npy"
runner.TRAIN_AUDIT=runner.OUT/"training_audit.json"; runner.SCORE_AUDIT=runner.OUT/"score_audit.json"

FIELDS=["Model","RAW Accuracy","RAW Precision","RAW Recall","RAW F1","PA Accuracy","PA Precision","PA Recall","PA F1","Threshold","Anomaly Ratio","Percentile"]


def verify_protocol(path: Path):
    protocol=json.loads(path.read_text(encoding="utf-8")); evaluation=protocol.get("evaluation",protocol.get("config",{}).get("evaluation",{}))
    if protocol.get("dataset")!="SMD" or not np.isclose(float(evaluation.get("anomaly_ratio",-1)),.9) or not np.isclose(float(evaluation.get("percentile",-1)),99.1):
        raise RuntimeError(f"SMD protocol mismatch: {path}")


def aggregate_detection():
    sources=[(ROOT/"results"/"SMD_PAPER_RESULTS"/"protocol.json",ROOT/"results"/"SMD_PAPER_RESULTS"/"Detection"/"comparison_detection.csv"),(ROOT/"results"/"SMD_TRANAD_RESULTS"/"protocol.json",ROOT/"results"/"SMD_TRANAD_RESULTS"/"Detection"/"comparison_detection.csv"),(runner.OUT/"protocol.json",runner.DET/"comparison_detection.csv")]
    rows=[]
    for protocol_path,metrics_path in sources:
        verify_protocol(protocol_path)
        with metrics_path.open(newline="",encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.append({"Model":row["Model"],"RAW Accuracy":row["Accuracy"],"RAW Precision":row["Precision"],"RAW Recall":row["Recall"],"RAW F1":row["F1"],"PA Accuracy":row["PA-Accuracy"],"PA Precision":row["PA-Precision"],"PA Recall":row["PA-Recall"],"PA F1":row["PA-F1"],"Threshold":row["Threshold"],"Anomaly Ratio":.9,"Percentile":99.1})
    expected={"ASCA-AD V4","PPLAD","LTFAD","TranAD","COUTA"}
    if {row["Model"] for row in rows}!=expected: raise RuntimeError(f"Unexpected model set: {[row['Model'] for row in rows]}")
    output=runner.OUT/"five_model_detection_comparison.csv"
    with output.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    print(f"five_model_detection={output}",flush=True)


def diagnostics():
    train_score=np.load(runner.TRAIN_SCORE,allow_pickle=False); test_score=np.load(runner.TEST_SCORE,allow_pickle=False)
    adapter=SMDCOUTADataAdapter("evaluate"); label=adapter.load_label(); threshold=float(np.percentile(np.concatenate([train_score,test_score]),99.1)); prediction=test_score>threshold
    distributions={}
    for name,values in (("normal",test_score[label==0]),("anomaly",test_score[label==1])):
        distributions[name]={"count":int(values.size),"mean":float(values.mean()),"median":float(np.median(values)),"p90":float(np.percentile(values,90)),"p95":float(np.percentile(values,95)),"p99":float(np.percentile(values,99)),"max":float(values.max())}
    tp=int(np.sum(prediction&(label==1))); fp=int(np.sum(prediction&(label==0))); fn=int(np.sum((~prediction)&(label==1)))
    starts=np.flatnonzero((label==1)&(np.r_[0,label[:-1]]==0)); ends=np.flatnonzero((label==1)&(np.r_[label[1:],0]==0)); hit=sum(bool(prediction[start:end+1].any()) for start,end in zip(starts,ends))
    report={"dataset":"SMD","formal_results_unchanged":True,"threshold":threshold,"threshold_above_anomaly_p99":threshold>distributions["anomaly"]["p99"],"threshold_in_normal_extreme_tail":threshold>distributions["normal"]["p95"],"score_distributions":distributions,"prediction":{"predicted_anomaly_points":int(prediction.sum()),"tp":tp,"fp":fp,"fn":fn,"anomaly_segment_count":int(len(starts)),"hit_anomaly_segment_count":int(hit)},"ranking":{"roc_auc":float(roc_auc_score(label,test_score)),"pr_auc":float(average_precision_score(label,test_score)),"reversed_score_roc_auc":float(roc_auc_score(label,-test_score))},"label_access_stage":"post-formal-evaluation diagnostics only","threshold_or_parameter_changed":False}
    out=runner.OUT/"Diagnostics"; out.mkdir(parents=True,exist_ok=True); (out/"score_distribution_diagnostics.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    (out/"score_distribution_diagnostics.md").write_text(f"# SMD COUTA Score Diagnostics\n\n- Threshold: `{threshold}`\n- ROC-AUC: `{report['ranking']['roc_auc']:.6f}`\n- PR-AUC: `{report['ranking']['pr_auc']:.6f}`\n- Predicted/TP/FP/FN: `{prediction.sum()}/{tp}/{fp}/{fn}`\n- Segments hit: `{hit}/{len(starts)}`\n",encoding="utf-8")
    print(f"diagnostics={out/'score_distribution_diagnostics.json'}",flush=True)


def finalize_protocol():
    path=runner.OUT/"protocol.json"; protocol=json.loads(path.read_text(encoding="utf-8")); protocol["audit"].update({"deepod_core_modified":False,"psm_results_modified":False,"skab_results_modified":False,"msl_results_modified":False,"hai_results_modified":False,"pump_results_modified":False,"ray_installed":False,"ray_stub_used":True,"fit_auto_hyper_used":False,"deepod_testbed_used":False,"ts_metrics_used":False,"internal_threshold_used_for_paper":False,"internal_prediction_used":False,"contamination_prediction_used":False}); path.write_text(json.dumps(protocol,indent=2,ensure_ascii=False),encoding="utf-8")


if __name__=="__main__":
    runner.main(); finalize_protocol(); diagnostics(); aggregate_detection()
