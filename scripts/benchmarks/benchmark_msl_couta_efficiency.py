#!/usr/bin/env python3
from pathlib import Path
import csv,sys
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import scripts.benchmarks.benchmark_psm_couta_efficiency as benchmark
from scripts.benchmarks.adapters.couta_dataset_adapter import MSLCOUTADataAdapter,load_msl_couta_config
benchmark.DATASET_NAME='MSL'; benchmark.CSV_INCLUDE_INCREMENTAL=False
benchmark.PSMCOUTADataAdapter=MSLCOUTADataAdapter; benchmark.load_couta_config=load_msl_couta_config
benchmark.OUT=ROOT/'results'/'MSL_COUTA_RESULTS'; benchmark.EFF=benchmark.OUT/'Efficiency'; benchmark.BUNDLE=benchmark.OUT/'checkpoints'/'COUTA_MSL_bundle.pt'
FIELDS=['Model','Total Parameters','State Dict (KiB)','Latency B=1 (ms)','Latency B=128 (ms)','Full Test Time (s)','Throughput (points/s)','GPU Peak (MiB)']
def pick(row,*names):
    for n in names:
        if n in row: return row[n]
    raise KeyError(names)
def aggregate():
    paths=[ROOT/'results'/'MSL_PAPER_RESULTS'/'Efficiency_ratio083'/'comparison_efficiency.csv',ROOT/'results'/'MSL_TRANAD_RESULTS'/'Efficiency'/'comparison_efficiency.csv',benchmark.EFF/'comparison_efficiency.csv']
    output=[]
    for path in paths:
        with path.open(newline='',encoding='utf-8') as f:
            for r in csv.DictReader(f):
                output.append({'Model':r['Model'],'Total Parameters':pick(r,'Parameters','Total Parameters'),
                    'State Dict (KiB)':pick(r,'State Dict(KiB)','State Dict (KiB)'),
                    'Latency B=1 (ms)':pick(r,'Latency_B1(ms)','Latency B=1 Mean (ms)'),
                    'Latency B=128 (ms)':pick(r,'Latency_B128(ms)','Latency B=128 Mean (ms)'),
                    'Full Test Time (s)':pick(r,'Full_Test_Time(s)','Full Test Time Mean (s)'),
                    'Throughput (points/s)':pick(r,'Throughput(points/s)','Throughput (points/s)'),
                    'GPU Peak (MiB)':pick(r,'GPU_Peak(MiB)','GPU Peak (MiB)')})
    with (benchmark.OUT/'five_model_efficiency_comparison.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader(); w.writerows(output)
if __name__=='__main__': benchmark.main(); aggregate()
