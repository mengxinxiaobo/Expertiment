#!/usr/bin/env python3
from pathlib import Path
import csv,json,sys
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import scripts.benchmarks.run_psm_couta_detection as runner
from scripts.benchmarks.adapters.couta_dataset_adapter import MSLCOUTADataAdapter,load_msl_couta_config
runner.DATASET_NAME='MSL'; runner.PSMCOUTADataAdapter=MSLCOUTADataAdapter; runner.load_couta_config=load_msl_couta_config
runner.OUT=ROOT/'results'/'MSL_COUTA_RESULTS'; runner.DET=runner.OUT/'Detection'; runner.SCORES=runner.OUT/'scores'; runner.CKPTS=runner.OUT/'checkpoints'
runner.BUNDLE=runner.CKPTS/'COUTA_MSL_bundle.pt'; runner.STATE=runner.CKPTS/'COUTA_MSL_state_dict.pt'
runner.TRAIN_SCORE=runner.SCORES/'train_score.npy'; runner.TEST_SCORE=runner.SCORES/'test_score.npy'; runner.TRAIN_AUDIT=runner.OUT/'training_audit.json'; runner.SCORE_AUDIT=runner.OUT/'score_audit.json'
def aggregate():
    sources=[ROOT/'results'/'MSL_PAPER_RESULTS'/'Detection_ratio083'/'comparison_detection.csv',ROOT/'results'/'MSL_TRANAD_RESULTS'/'Detection'/'comparison_detection.csv',runner.DET/'comparison_detection.csv']
    rows=[]
    for path in sources:
        with path.open(newline='',encoding='utf-8') as f: rows.extend(csv.DictReader(f))
    out=runner.OUT/'five_model_detection_comparison.csv'
    with out.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def finalize_protocol():
    path=runner.OUT/'protocol.json'; protocol=json.loads(path.read_text(encoding='utf-8'))
    protocol['audit'].update({'deepod_core_modified':False,'psm_results_modified':False,
        'skab_results_modified':False,'ray_installed':False,'ray_stub_used':True,
        'fit_auto_hyper_used':False,'deepod_testbed_used':False,'ts_metrics_used':False,
        'internal_threshold_used_for_paper':False,'internal_prediction_used':False,
        'contamination_prediction_used':False})
    path.write_text(json.dumps(protocol,indent=2,ensure_ascii=False),encoding='utf-8')
if __name__=='__main__': runner.main(); finalize_protocol(); aggregate()
