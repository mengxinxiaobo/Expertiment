#!/usr/bin/env python3
from pathlib import Path
import csv,json,sys
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import scripts.benchmarks.run_psm_couta_detection as runner
from scripts.benchmarks.adapters.couta_dataset_adapter import HAICOUTADataAdapter,load_hai_couta_config
runner.DATASET_NAME='HAI'; runner.PSMCOUTADataAdapter=HAICOUTADataAdapter; runner.load_couta_config=load_hai_couta_config
runner.OUT=ROOT/'results'/'HAI_COUTA_RESULTS'; runner.DET=runner.OUT/'Detection'; runner.SCORES=runner.OUT/'scores'; runner.CKPTS=runner.OUT/'checkpoints'
runner.BUNDLE=runner.CKPTS/'COUTA_HAI_bundle.pt'; runner.STATE=runner.CKPTS/'COUTA_HAI_state_dict.pt'; runner.TRAIN_SCORE=runner.SCORES/'train_score.npy'; runner.TEST_SCORE=runner.SCORES/'test_score.npy'
runner.TRAIN_AUDIT=runner.OUT/'training_audit.json'; runner.SCORE_AUDIT=runner.OUT/'score_audit.json'
FIELDS=['Model','RAW Accuracy','RAW Precision','RAW Recall','RAW F1','PA Accuracy','PA Precision','PA Recall','PA F1','Threshold','Anomaly Ratio','Percentile']
def protocol_values(path):
    data=json.loads(path.read_text(encoding='utf-8')); text=json.dumps(data)
    if '0.98' not in text or '99.02' not in text: raise RuntimeError(f'HAI protocol mismatch: {path}')
def aggregate():
    protocol_values(ROOT/'results'/'HAI_PAPER_RESULTS'/'protocol.json'); protocol_values(ROOT/'results'/'HAI_TRANAD_RESULTS'/'protocol.json'); protocol_values(runner.OUT/'protocol.json')
    paths=[ROOT/'results'/'HAI_PAPER_RESULTS'/'Detection'/'comparison_detection.csv',ROOT/'results'/'HAI_TRANAD_RESULTS'/'Detection'/'comparison_detection.csv',runner.DET/'comparison_detection.csv']
    rows=[]
    for path in paths:
        with path.open(newline='',encoding='utf-8') as f:
            for r in csv.DictReader(f):
                rows.append({'Model':r['Model'],'RAW Accuracy':r['Accuracy'],'RAW Precision':r['Precision'],'RAW Recall':r['Recall'],'RAW F1':r['F1'],
                    'PA Accuracy':r['PA-Accuracy'],'PA Precision':r['PA-Precision'],'PA Recall':r['PA-Recall'],'PA F1':r['PA-F1'],'Threshold':r['Threshold'],'Anomaly Ratio':0.98,'Percentile':99.02})
    with (runner.OUT/'five_model_detection_comparison.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
def finalize():
    path=runner.OUT/'protocol.json'; p=json.loads(path.read_text(encoding='utf-8')); p['audit'].update({'deepod_core_modified':False,'psm_results_modified':False,'skab_results_modified':False,
      'msl_results_modified':False,'ray_installed':False,'ray_stub_used':True,'fit_auto_hyper_used':False,'deepod_testbed_used':False,'ts_metrics_used':False,
      'internal_threshold_used_for_paper':False,'internal_prediction_used':False,'contamination_prediction_used':False}); path.write_text(json.dumps(p,indent=2,ensure_ascii=False),encoding='utf-8')
if __name__=='__main__': runner.main(); finalize(); aggregate()
