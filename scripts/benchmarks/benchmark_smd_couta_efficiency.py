#!/usr/bin/env python3
"""SMD binding for the streaming, inference-only COUTA efficiency runner."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

import scripts.benchmarks.benchmark_pump_couta_efficiency as benchmark
from scripts.benchmarks.adapters.couta_dataset_adapter import SMDCOUTADataAdapter,load_smd_couta_config

benchmark.PUMPCOUTADataAdapter=SMDCOUTADataAdapter; benchmark.load_pump_couta_config=load_smd_couta_config
benchmark.DATASET_NAME="SMD"
benchmark.OUT=ROOT/"results"/"SMD_COUTA_RESULTS"; benchmark.EFF=benchmark.OUT/"Efficiency"; benchmark.BUNDLE=benchmark.OUT/"checkpoints"/"COUTA_SMD_bundle.pt"


def pick(row,*names,default=""):
    for name in names:
        if name in row and row[name]!="": return row[name]
    return default


def mean_std(mean,std): return f"{float(mean):.6f} ± {float(std):.6f}" if std else f"{float(mean):.6f} ± N/A"


def aggregate_efficiency():
    sources=[ROOT/"results"/"SMD_PAPER_RESULTS"/"Efficiency"/"comparison_efficiency.csv",ROOT/"results"/"SMD_TRANAD_RESULTS"/"Efficiency"/"comparison_efficiency.csv",benchmark.EFF/"comparison_efficiency.csv"]
    fields=["Model","Params","State Dict (KiB)","B=1 mean ± std (ms)","B=128 mean ± std (ms)","Full Test mean ± std (s)","Throughput (points/s)","GPU Peak (MiB)"]
    rows=[]; coverage={}; known={"ASCA-AD V4":708400,"PPLAD":708330,"LTFAD":708390,"TranAD":708420,"COUTA":708420}
    for path in sources:
        with path.open(newline="",encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                model=row["Model"]; b1=pick(row,"Latency B=1 (ms)","Latency B=1 Mean (ms)"); b128=pick(row,"Latency B=128 (ms)","Latency B=128 Mean (ms)"); full=pick(row,"Full Test Time (s)","Full Test Time Mean (s)"); coverage[model]=int(float(pick(row,"Actual Processed Points",default=str(known[model]))))
                rows.append({"Model":model,"Params":pick(row,"Parameters","Total Parameters"),"State Dict (KiB)":pick(row,"State Dict (KiB)"),"B=1 mean ± std (ms)":mean_std(b1,pick(row,"Latency B=1 Std (ms)")),"B=128 mean ± std (ms)":mean_std(b128,pick(row,"Latency B=128 Std (ms)")),"Full Test mean ± std (s)":mean_std(full,pick(row,"Full Test Time Std (s)")),"Throughput (points/s)":pick(row,"Throughput (points/s)"),"GPU Peak (MiB)":pick(row,"GPU Peak (MiB)")})
    expected={"ASCA-AD V4","PPLAD","LTFAD","TranAD","COUTA"}
    if {row["Model"] for row in rows}!=expected: raise RuntimeError(f"Unexpected model set: {[row['Model'] for row in rows]}")
    output=benchmark.OUT/"five_model_efficiency_comparison.csv"
    with output.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    equal=len(set(coverage.values()))==1
    if not equal: print(f"[WARNING] Existing SMD efficiency rows cover different points: {coverage}; strict full-time comparability is not claimed.",flush=True)
    return {"path":str(output.relative_to(ROOT)).replace("\\","/"),"processed_points_by_model":coverage,"processed_points_equal":equal}


benchmark.aggregate_efficiency=aggregate_efficiency

if __name__=="__main__": benchmark.main()
