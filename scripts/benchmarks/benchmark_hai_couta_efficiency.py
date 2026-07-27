#!/usr/bin/env python3
from pathlib import Path
import csv,sys
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import scripts.benchmarks.benchmark_psm_couta_efficiency as benchmark
from scripts.benchmarks.adapters.couta_dataset_adapter import HAICOUTADataAdapter,load_hai_couta_config
benchmark.DATASET_NAME='HAI'; benchmark.CSV_INCLUDE_INCREMENTAL=False; benchmark.PSMCOUTADataAdapter=HAICOUTADataAdapter; benchmark.load_couta_config=load_hai_couta_config
benchmark.OUT=ROOT/'results'/'HAI_COUTA_RESULTS'; benchmark.EFF=benchmark.OUT/'Efficiency'; benchmark.BUNDLE=benchmark.OUT/'checkpoints'/'COUTA_HAI_bundle.pt'
FIELDS=['Model','Total Parameters','State Dict (KiB)','Latency B=1 Mean (ms)','Latency B=1 Std (ms)','Latency B=128 Mean (ms)','Latency B=128 Std (ms)','Full Test Time Mean (s)','Full Test Time Std (s)','Processed Points','Throughput (points/s)','GPU Peak (MiB)']
def pick(r,*names,default=''):
    for n in names:
        if n in r: return r[n]
    return default
def aggregate():
    paths=[ROOT/'results'/'HAI_PAPER_RESULTS'/'Efficiency'/'comparison_efficiency.csv',ROOT/'results'/'HAI_TRANAD_RESULTS'/'Efficiency'/'comparison_efficiency.csv',benchmark.EFF/'comparison_efficiency.csv']; rows=[]
    for path in paths:
        with path.open(newline='',encoding='utf-8') as f:
            for r in csv.DictReader(f):
                rows.append({'Model':r['Model'],'Total Parameters':pick(r,'Parameters','Total Parameters'),'State Dict (KiB)':pick(r,'State Dict(KiB)','State Dict (KiB)'),
                  'Latency B=1 Mean (ms)':pick(r,'Latency_B1(ms)','Latency B=1 Mean (ms)'),'Latency B=1 Std (ms)':pick(r,'Latency B=1 Std (ms)'),
                  'Latency B=128 Mean (ms)':pick(r,'Latency_B128(ms)','Latency B=128 Mean (ms)'),'Latency B=128 Std (ms)':pick(r,'Latency B=128 Std (ms)'),
                  'Full Test Time Mean (s)':pick(r,'Full_Test_Time(s)','Full Test Time Mean (s)'),'Full Test Time Std (s)':pick(r,'Full Test Time Std (s)'),
                  'Processed Points':pick(r,'Actual Processed Points',default='284400'),'Throughput (points/s)':pick(r,'Throughput(points/s)','Throughput (points/s)'),
                  'GPU Peak (MiB)':pick(r,'GPU_Peak(MiB)','GPU Peak (MiB)')})
    with (benchmark.OUT/'five_model_efficiency_comparison.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
if __name__=='__main__': benchmark.main(); aggregate()
