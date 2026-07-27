#!/usr/bin/env python3
"""One-epoch, train-only official COUTA validation for MSL."""
from __future__ import annotations
import contextlib, hashlib, io, json, math, re, sys, time, warnings
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.benchmarks.adapters.couta_dataset_adapter import (
    MSLCOUTADataAdapter, MSL_CONFIG_PATH, RAY_AUDIT, construct_official_couta,
    import_official_couta, load_msl_couta_config, make_bundle, restore_from_bundle, sha256)
OUT=ROOT/"results"/"MSL_COUTA_RESULTS"/"Validation"; JSON_PATH=OUT/"validation_report.json"
MD_PATH=OUT/"validation_report.md"; LOG_PATH=OUT/"validation.log"; CKPT=OUT/"temporary_checkpoint.pt"
POINTS=4096; LOG=[]
def now(): return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
def emit(x=""): print(x,flush=True); LOG.append(x)
def check(name,ok,detail=""):
    emit(f"[{'PASS' if ok else 'FAIL'}] {name}{': '+detail if detail else ''}")
    if not ok: raise RuntimeError(f"{name}: {detail}")
def tree_hash(path):
    h=hashlib.sha256()
    if path.exists():
        for p in sorted(x for x in path.rglob('*') if x.is_file()):
            h.update(str(p.relative_to(path)).encode()); h.update(bytes.fromhex(sha256(p)))
    return h.hexdigest()
def main():
    OUT.mkdir(parents=True,exist_ok=True); started=time.perf_counter(); start_time=now()
    emit("="*72); emit("Stage: MSL COUTA setup validation"); emit(f"Start Time: {start_time}")
    old={d:tree_hash(ROOT/'results'/d) for d in ('PSM_COUTA_RESULTS','SKAB_COUTA_RESULTS')}
    config=load_msl_couta_config(); data_dir=ROOT/'dataset'/'MSL'
    paths={n:data_dir/n for n in ('MSL_train.npy','MSL_test.npy','MSL_test_label.npy')}
    arrays={n:np.load(p,allow_pickle=False) for n,p in paths.items()}
    train,test,label=arrays.values(); stats={n:{'nan_count':int(np.isnan(a).sum()),'inf_count':int(np.isinf(a).sum())} for n,a in arrays.items()}
    check('files exist',all(p.is_file() for p in paths.values()))
    check('train shape',train.shape==(58317,55),str(train.shape)); check('test shape',test.shape==(73729,55),str(test.shape))
    check('label shape',label.shape==(73729,),str(label.shape)); check('finite arrays',all(np.isfinite(a).all() for a in arrays.values()))
    unique,counts=np.unique(label,return_counts=True); check('binary labels',set(unique.tolist())=={False,True},str(dict(zip(unique.tolist(),counts.tolist()))))
    check('timeline alignment',len(test)==len(label)); del test,label,arrays
    check('Ray absent',not RAY_AUDIT['ray_installed']); COUTA,_=import_official_couta()
    check('official COUTA import',COUTA.__module__=='deepod.models.time_series.couta')
    from ray import tune
    blocked=False
    try: tune.choice([1])
    except RuntimeError: blocked=True
    check('Ray Tune fail-closed',blocked)
    adapter=MSLCOUTADataAdapter('validation'); subset=adapter.load_train(limit=POINTS)
    scaler,scaled=adapter.fit_train_scaler(subset); adapter.assert_training_isolated()
    check('train-only float32 scaler',scaled.dtype==np.float32 and np.isfinite(scaled).all())
    check('CUDA available',torch.cuda.is_available()); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); baseline=torch.cuda.memory_allocated()
    model=construct_official_couta(config,'cuda',epochs=1); buffer=io.StringIO(); fit_start=time.perf_counter()
    with warnings.catch_warnings(record=True) as caught,contextlib.redirect_stdout(buffer):
        warnings.simplefilter('always'); result=model.fit(scaled)
    fit_seconds=time.perf_counter()-fit_start
    for line in buffer.getvalue().splitlines(): emit(line)
    loss=[float(x) for x in re.findall(r'(?:^|\s)loss:\s*([0-9eE+.-]+)',buffer.getvalue())]
    val_loss=[float(x) for x in re.findall(r'val_loss:\s*([0-9eE+.-]+)',buffer.getvalue())]
    check('official fit',result is None); check('finite loss',bool(loss) and all(map(math.isfinite,loss)),str(loss))
    check('finite validation loss',bool(val_loss) and all(map(math.isfinite,val_loss)),str(val_loss)); check('finite center',bool(torch.isfinite(model.c).all()))
    total=sum(p.numel() for p in model.net.parameters()); trainable=sum(p.numel() for p in model.net.parameters() if p.requires_grad)
    check('actual parameters',total==trainable==4337,f'{total}/{trainable}')
    score=np.asarray(model.decision_function(scaled)); check('decision score',score.shape==(POINTS,) and np.isfinite(score).all())
    check('prefix padding',np.all(score[:29]==0),'29 zero points')
    bundle=make_bundle(model,scaler,config); bundle.update({'validation_epochs':1,'formal_checkpoint':False}); torch.save(bundle,CKPT)
    with warnings.catch_warnings(record=True) as restore_warnings:
        warnings.simplefilter('always'); restored,restored_scaler=restore_from_bundle(torch.load(CKPT,map_location='cuda',weights_only=False),'cuda')
    caught.extend(restore_warnings); restored_scaled=np.asarray(restored_scaler.transform(subset),dtype=np.float32)
    restored_score=np.asarray(restored.decision_function(restored_scaled)); score_error=float(np.max(np.abs(score-restored_score)))
    scaler_error=float(np.max(np.abs(scaled-restored_scaled))); check('checkpoint restoration',score_error<=1e-7 and scaler_error<=1e-7,f'{score_error:.3e}/{scaler_error:.3e}')
    new={d:tree_hash(ROOT/'results'/d) for d in old}; check('prior COUTA results unchanged',old==new)
    peak=torch.cuda.max_memory_allocated(); warning_text=[f'{w.category.__name__}: {w.message}' for w in caught]
    for x in warning_text: emit('[WARNING] '+x)
    audit={'model_source_modified':False,'deepod_core_modified':False,'psm_results_modified':False,'skab_results_modified':False,
           'ray_installed':False,'ray_stub_used':True,'ray_tune_used':False,'fit_auto_hyper_used':False,'training_ray_used':False,
           'deepod_testbed_used':False,'ts_metrics_used':False,'best_f1_search':False,'training_test_access':False,
           'training_test_label_access':False,'score_label_access':False,'score_search':False,'ratio_search':False,
           'threshold_search':False,'parameter_search':False,'oracle_search':False,'internal_threshold_used_for_paper':False,
           'internal_prediction_used':False,'contamination_prediction_used':False,'formal_training_started':False,
           'formal_scores_generated':False,'formal_evaluation_run':False,'formal_efficiency_run':False}
    elapsed=time.perf_counter()-started
    report={'status':'READY','start_time':start_time,'end_time':now(),'elapsed_seconds':elapsed,'dataset':'MSL','input_c':55,
            'files':{n:{'path':str(p.relative_to(ROOT)).replace('\\','/'),'shape':list(np.load(p,mmap_mode='r').shape),
                       'dtype':str(np.load(p,mmap_mode='r').dtype),'size_bytes':p.stat().st_size,**stats[n]} for n,p in paths.items()},
            'label_unique_counts':dict(zip(map(str,unique.tolist()),counts.tolist())),'config_path':str(MSL_CONFIG_PATH.relative_to(ROOT)).replace('\\','/'),
            'anomaly_ratio':.83,'percentile':99.17,'parameters':{'total':total,'trainable':trainable},
            'official_fit':{'points':POINTS,'epochs':1,'seconds':fit_seconds,'loss':loss,'val_loss':val_loss},
            'score':{'shape':list(score.shape),'finite':True,'prefix_padding':29,'definition':'sum((rep-c)^2)+sum((rep_dup-c)^2)'},
            'formal_alignment':{'train_points':58317,'train_windows':58288,'test_points':73729,'test_windows':73700},
            'checkpoint':{'path':str(CKPT.relative_to(ROOT)).replace('\\','/'),'sha256':sha256(CKPT),'score_max_abs_error':score_error,'scaler_max_abs_error':scaler_error},
            'gpu':{'peak_mib':peak/2**20,'incremental_mib':(peak-baseline)/2**20},'pytorch':torch.__version__,'cuda':torch.version.cuda,
            'warnings':warning_text,'training_file_access':adapter.audit(),'integrity_only_label_access':'dataset/MSL/MSL_test_label.npy','audit':audit}
    JSON_PATH.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    MD_PATH.write_text(f"# MSL COUTA Validation\n\n- Status: **READY**\n- Shape: train `(58317,55)`, test `(73729,55)`, label `(73729,)`\n- Parameters: `{total}`\n- 1-epoch official fit: `{fit_seconds:.3f}s`\n- Checkpoint score error: `{score_error:.3e}`\n- GPU peak: `{peak/2**20:.3f} MiB`\n",encoding='utf-8')
    emit('[PASS] MSL COUTA Validation READY'); emit(f"End Time: {report['end_time']}"); emit(f'Elapsed Time: {elapsed:.3f}s')
    LOG_PATH.write_text('\n'.join(LOG)+'\n',encoding='utf-8')
if __name__=='__main__': main()
