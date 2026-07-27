#!/usr/bin/env python3
"""Train-only one-epoch validation for official DeepOD COUTA on HAI."""
from __future__ import annotations
import contextlib,hashlib,io,json,math,re,sys,time,warnings
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.benchmarks.adapters.couta_dataset_adapter import (
    HAICOUTADataAdapter,HAI_CONFIG_PATH,RAY_AUDIT,construct_official_couta,
    import_official_couta,load_hai_couta_config,make_bundle,restore_from_bundle,sha256)
from scripts.benchmarks.run_psm_couta_detection import official_scores_chunked
OUT=ROOT/'results'/'HAI_COUTA_RESULTS'/'Validation'; REPORT=OUT/'validation_report.json'; MD=OUT/'validation_report.md'; LOGFILE=OUT/'validation.log'; CKPT=OUT/'temporary_checkpoint.pt'
POINTS=4096; LINES=[]
def now(): return datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')
def log(x=''): print(x,flush=True); LINES.append(x)
def check(name,ok,detail=''):
    log(f"[{'PASS' if ok else 'FAIL'}] {name}{': '+detail if detail else ''}")
    if not ok: raise RuntimeError(f'{name}: {detail}')
def tree_hash(path):
    h=hashlib.sha256()
    if path.exists():
        for p in sorted(x for x in path.rglob('*') if x.is_file()): h.update(str(p.relative_to(path)).encode()); h.update(bytes.fromhex(sha256(p)))
    return h.hexdigest()
def main():
    OUT.mkdir(parents=True,exist_ok=True); started=time.perf_counter(); start=now(); log('='*72); log('Stage: HAI COUTA setup validation'); log(f'Start Time: {start}')
    previous={d:tree_hash(ROOT/'results'/d) for d in ('PSM_COUTA_RESULTS','SKAB_COUTA_RESULTS','MSL_COUTA_RESULTS')}
    config=load_hai_couta_config(); data=ROOT/'dataset'/'HAI'; paths={n:data/n for n in ('HAI_train.npy','HAI_test.npy','HAI_test_label.npy')}
    arrays={n:np.load(p,mmap_mode='r',allow_pickle=False) for n,p in paths.items()}; train,test,label=arrays.values()
    check('files',all(p.is_file() for p in paths.values())); check('train shape',train.shape==(896400,86),str(train.shape)); check('test shape',test.shape==(284400,86),str(test.shape)); check('label shape',label.shape==(284400,),str(label.shape))
    stats={};
    for name,array in arrays.items():
        nan=int(np.isnan(array).sum()); inf=int(np.isinf(array).sum()); stats[name]={'nan_count':nan,'inf_count':inf}; check(f'{name} finite',nan==0 and inf==0)
    unique,counts=np.unique(label,return_counts=True); check('binary label',set(unique.tolist())=={0,1},str(dict(zip(unique.tolist(),counts.tolist())))); check('alignment',len(test)==len(label))
    check('Ray absent',not RAY_AUDIT['ray_installed']); COUTA,_=import_official_couta(); check('official import',COUTA.__module__=='deepod.models.time_series.couta')
    from ray import tune
    blocked=False
    try: tune.choice([1])
    except RuntimeError: blocked=True
    check('Ray fail-closed',blocked)
    adapter=HAICOUTADataAdapter('validation'); subset=adapter.load_train(limit=POINTS); scaler,scaled=adapter.fit_train_scaler(subset); adapter.assert_training_isolated()
    check('float32 scaled train',scaled.dtype==np.float32 and np.isfinite(scaled).all()); constant_count=int(np.sum(np.var(subset,axis=0)==0)); check('finite scaler',np.isfinite(scaler.mean_).all() and np.isfinite(scaler.scale_).all() and np.all(scaler.scale_!=0),f'constant_features_in_slice={constant_count}')
    check('CUDA',torch.cuda.is_available()); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); baseline=torch.cuda.memory_allocated()
    model=construct_official_couta(config,'cuda',epochs=1); output=io.StringIO(); fit_start=time.perf_counter()
    with warnings.catch_warnings(record=True) as caught,contextlib.redirect_stdout(output): warnings.simplefilter('always'); result=model.fit(scaled)
    fit_seconds=time.perf_counter()-fit_start
    for line in output.getvalue().splitlines(): log(line)
    loss=[float(x) for x in re.findall(r'(?:^|\s)loss:\s*([0-9eE+.-]+)',output.getvalue())]; val=[float(x) for x in re.findall(r'val_loss:\s*([0-9eE+.-]+)',output.getvalue())]
    check('official fit',result is None); check('finite loss',bool(loss) and all(map(math.isfinite,loss)),str(loss)); check('finite val loss',bool(val) and all(map(math.isfinite,val)),str(val)); check('finite center',bool(torch.isfinite(model.c).all()))
    total=sum(p.numel() for p in model.net.parameters()); trainable=sum(p.numel() for p in model.net.parameters() if p.requires_grad); check('actual parameters',total==trainable==5825,f'{total}/{trainable}')
    score=np.asarray(model.decision_function(scaled)); check('score',score.shape==(POINTS,) and np.isfinite(score).all()); check('prefix 29',np.all(score[:29]==0))
    chunked_score=official_scores_chunked(model,scaled,1000,'validation'); chunk_error=float(np.max(np.abs(score-chunked_score)))
    check('chunked official score equivalence',chunk_error<=1e-7,f'{chunk_error:.3e}')
    bundle=make_bundle(model,scaler,config); bundle.update({'validation_epochs':1,'formal_checkpoint':False}); torch.save(bundle,CKPT)
    with warnings.catch_warnings(record=True) as more: warnings.simplefilter('always'); restored,restored_scaler=restore_from_bundle(torch.load(CKPT,map_location='cuda',weights_only=False),'cuda')
    caught.extend(more); restored_scaled=np.asarray(restored_scaler.transform(subset),dtype=np.float32); restored_score=np.asarray(restored.decision_function(restored_scaled))
    score_error=float(np.max(np.abs(score-restored_score))); scaler_error=float(np.max(np.abs(scaled-restored_scaled))); check('checkpoint restore',score_error<=1e-7 and scaler_error<=1e-7,f'{score_error:.3e}/{scaler_error:.3e}')
    check('previous results unchanged',previous=={d:tree_hash(ROOT/'results'/d) for d in previous}); peak=torch.cuda.max_memory_allocated(); warnings_text=[f'{w.category.__name__}: {w.message}' for w in caught]
    for x in warnings_text: log('[WARNING] '+x)
    audit={'model_source_modified':False,'deepod_core_modified':False,'psm_results_modified':False,'skab_results_modified':False,'msl_results_modified':False,
      'ray_installed':False,'ray_stub_used':True,'ray_tune_used':False,'fit_auto_hyper_used':False,'training_ray_used':False,'deepod_testbed_used':False,'ts_metrics_used':False,
      'best_f1_search':False,'training_test_access':False,'training_test_label_access':False,'score_label_access':False,'score_search':False,'ratio_search':False,
      'threshold_search':False,'parameter_search':False,'oracle_search':False,'internal_threshold_used_for_paper':False,'internal_prediction_used':False,
      'contamination_prediction_used':False,'formal_training_started':False,'formal_scores_generated':False,'formal_evaluation_run':False,'formal_efficiency_run':False}
    elapsed=time.perf_counter()-started
    report={'status':'READY','start_time':start,'end_time':now(),'elapsed_seconds':elapsed,'dataset':'HAI','input_c':86,'original_dtype':'float32','model_dtype':'float32',
      'files':{n:{'path':str(p.relative_to(ROOT)).replace('\\','/'),'shape':list(arrays[n].shape),'dtype':str(arrays[n].dtype),'size_bytes':p.stat().st_size,**stats[n]} for n,p in paths.items()},
      'label_counts':dict(zip(map(str,unique.tolist()),counts.tolist())),'anomaly_ratio_actual_percent':float(counts[1]/counts.sum()*100),'config_path':str(HAI_CONFIG_PATH.relative_to(ROOT)).replace('\\','/'),
      'anomaly_ratio':.98,'percentile':99.02,'parameters':{'total':total,'trainable':trainable},'official_fit':{'points':POINTS,'epochs':1,'seconds':fit_seconds,'loss':loss,'val_loss':val},
      'scaler':{'constant_features_in_validation_slice':constant_count,'mean_finite':True,'scale_finite_nonzero':True},'score':{'shape':list(score.shape),'prefix_padding':29,'finite':True,'chunked_equivalence_max_abs_error':chunk_error},
      'formal_alignment':{'train_points':896400,'training_windows':89638,'train_inference_windows':896371,'test_points':284400,'test_inference_windows':284371},
      'checkpoint':{'path':str(CKPT.relative_to(ROOT)).replace('\\','/'),'sha256':sha256(CKPT),'score_max_abs_error':score_error,'scaler_max_abs_error':scaler_error},
      'gpu':{'peak_mib':peak/2**20,'incremental_mib':(peak-baseline)/2**20},'pytorch':torch.__version__,'cuda':torch.version.cuda,'warnings':warnings_text,
      'training_file_access':adapter.audit(),'integrity_only_label_access':'dataset/HAI/HAI_test_label.npy','audit':audit}
    REPORT.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8'); MD.write_text(f"# HAI COUTA Validation\n\n- Status: **READY**\n- Shapes: train `(896400,86)`, test `(284400,86)`, label `(284400,)`\n- Parameters: `{total}`\n- Checkpoint score error: `{score_error:.3e}`\n",encoding='utf-8')
    log('[PASS] HAI COUTA Validation READY'); log(f"End Time: {report['end_time']}"); log(f'Elapsed Time: {elapsed:.3f}s'); LOGFILE.write_text('\n'.join(LINES)+'\n',encoding='utf-8')
if __name__=='__main__': main()
