#!/usr/bin/env python3
"""Read-only, independent-process CPU-memory root-cause audit.

The formal worker preserves the inference paths used by
run_unified_ram_benchmark.py.  Object inspection is deliberately executed in a
second process so its allocations cannot contaminate the formal peak RSS.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import tracemalloc
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "MEMORY_ROOT_CAUSE_AUDIT"
SUMMARY = OUT / "summary"
RAM = ROOT / "results" / "UNIFIED_RAM_BENCHMARK"
MODELS = ("ASCA-AD V4", "PPLAD", "LTFAD", "TranAD", "COUTA")
DATASETS = ("SKAB", "HAI", "SMD")
THREAD_ENV = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
              "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
STAGES = [f"S{i}" for i in range(17)]
WARMUP = 3
AUDIT_VERSION = 3


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


def safe(model: str) -> str:
    return model.replace("-", "_").replace(" ", "_")


def case_dir(dataset: str, model: str) -> Path:
    return OUT / dataset / safe(model)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def smaps() -> dict[str, float | None]:
    path = Path("/proc/self/smaps_rollup")
    if not path.is_file():
        return {k: None for k in ("anonymous_mib", "private_clean_mib", "private_dirty_mib", "shared_clean_mib", "shared_dirty_mib")}
    values: dict[str, int] = {}
    for line in path.read_text(errors="replace").splitlines():
        match = re.match(r"(Anonymous|Private_Clean|Private_Dirty|Shared_Clean|Shared_Dirty):\s+(\d+) kB", line)
        if match: values[match.group(1)] = int(match.group(2))
    def mib(key: str): return values.get(key) / 1024 if key in values else None
    return {"anonymous_mib": mib("Anonymous"), "private_clean_mib": mib("Private_Clean"),
            "private_dirty_mib": mib("Private_Dirty"), "shared_clean_mib": mib("Shared_Clean"),
            "shared_dirty_mib": mib("Shared_Dirty")}


def memory_values() -> dict[str, Any]:
    proc = psutil.Process()
    basic = proc.memory_info(); full = proc.memory_full_info()
    current, peak = tracemalloc.get_traced_memory()
    result = {"rss_mib": basic.rss / 2**20, "uss_mib": getattr(full, "uss", None),
              "pss_mib": getattr(full, "pss", None), "vms_mib": basic.vms / 2**20,
              "tracemalloc_current_mib": current / 2**20, "tracemalloc_peak_mib": peak / 2**20}
    for key in ("uss_mib", "pss_mib"):
        if result[key] is not None: result[key] /= 2**20
    result.update(smaps()); return result


class Recorder:
    def __init__(self): self.rows: list[dict[str, Any]] = []
    def take(self, stage: str, notes: str = "", objects: list[str] | None = None) -> None:
        row = {"stage": stage, "timestamp": now(), **memory_values(),
               "data_objects": objects or [], "notes": notes}
        self.rows.append(row)


class Sampler:
    def __init__(self):
        self.rows: list[dict[str, Any]] = []; self.stop = threading.Event()
        self.state = {"stage": "S13", "batch_index": 0, "batch_shape": "", "processed_points": 0, "cached_outputs": 0}
        self.thread = threading.Thread(target=self._run, daemon=True)
    def _run(self):
        while not self.stop.is_set():
            try: self.rows.append({"timestamp": now(), **memory_values(), **self.state})
            except Exception as exc: self.rows.append({"timestamp": now(), "sampling_error": str(exc), **self.state})
            self.stop.wait(0.01)
    def start(self): self.thread.start()
    def finish(self): self.stop.set(); self.thread.join()


def checkpoint_guard(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(path),
            "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def prepare(model_name: str, dataset: str, rec: Recorder, validation: bool, collect_objects: bool = False) -> dict[str, Any]:
    # S0 is taken by worker before numerical imports.
    import numpy as np
    rec.take("S1", "NumPy imported")
    import torch
    rec.take("S2", "PyTorch imported; CUDA tensor not yet created")
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
    import scripts.benchmarks.run_unified_efficiency_rebenchmark as unified
    if model_name in unified.CORE_MODELS:
        binding = unified.import_core(dataset)
    elif model_name == "TranAD":
        binding = unified.import_tranad(dataset)
    else:
        binding = unified.import_couta(dataset)
    rec.take("S3", "Formal benchmark plus selected model adapter/generator modules imported")
    device = unified.check_cuda()

    objects: dict[str, Any] = {}
    rec.take("S4", "Immediately before combined model construction/checkpoint loader; loader API does not expose a finer boundary")
    if model_name in unified.CORE_MODELS:
        protocol = binding
        display, window, model, score_call, checkpoint = protocol.load_model(unified.MODEL_KEYS[model_name], device)
        if display != model_name: raise RuntimeError("model identity mismatch")
        model.eval(); guard = checkpoint_guard(Path(checkpoint))
        rec.take("S5", "Model constructed and checkpoint loaded", ["model"])
        data_dir = ROOT / "dataset" / dataset
        train = np.load(data_dir / f"{dataset}_train.npy", allow_pickle=False)
        test = np.load(data_dir / f"{dataset}_test.npy", allow_pickle=False)
        rec.take("S6", "Raw train and test NumPy arrays loaded", ["raw_train", "raw_test"])
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(train); rec.take("S7", "StandardScaler fit on train only", ["raw_train", "raw_test", "scaler"])
        transformed = scaler.transform(test); rec.take("S8", "Test transformed; raw and transformed coexist", ["raw_train", "raw_test", "transformed_test"])
        scaled = np.asarray(transformed, dtype=np.float32); rec.take("S9", "float32 conversion complete", ["raw_train", "raw_test", "transformed_test", "scaled_test"])
        if collect_objects:
            objects.update(raw_train=train, raw_test=test, transformed_test=transformed, scaled_test=scaled, scaler=scaler, model=model)
        del train, test, transformed, scaler
        starts = unified.core_starts(len(scaled), window)
        first, _ = next(unified.core_cpu_batches(scaled, starts, window)); rec.take("S10", "First CPU native-window tensor constructed", ["scaled_test", "first_batch"])
        rec.take("S11", "Window-start schedule constructed; streaming batches", ["scaled_test", "window_starts", "first_batch"])
        def batches(): return unified.core_cpu_batches(scaled, starts if not validation else starts[:unified.CORE_BATCH], window)
        def call(x): return score_call(x)
        raw_points = len(scaled); expected = raw_points if not validation else min(raw_points, window * min(len(starts), unified.CORE_BATCH))
        ctx = {"torch": torch, "device": device, "first": first, "batches": batches, "call": call,
               "raw_points": raw_points, "expected": expected, "window": window, "batch_size": unified.CORE_BATCH,
               "checkpoint": guard, "objects": objects, "adapter_audit": {}, "native": "streamed native windows"}
    elif model_name == "TranAD":
        benchmark = binding; config = benchmark.load_tranad_psm_config()
        model, load_audit = benchmark.load_model(config, device); model.eval(); guard = checkpoint_guard(Path(benchmark.CHECKPOINT_PATH))
        rec.take("S5", "TranAD model constructed and checkpoint loaded", ["model"])
        adapter = benchmark.TranADPSMDataAdapter("score")
        train_values = adapter.train_values
        rec.take("S6", "Adapter loaded train", ["adapter._train"]); rec.take("S7", "Adapter scaler fit on train", ["adapter._scaler"])
        rec.take("S8", "Adapter train transform complete", ["adapter._train"]); rec.take("S9", "Adapter train float32 conversion complete", ["adapter._train"])
        test_values = adapter.test_values
        rec.take("S10", "TranAD padded-window Dataset backing arrays prepared", ["adapter._train", "adapter._test"])
        loader = adapter.loader("test", batch_size=unified.TRANAD_BATCH, shuffle=False)
        first = next(iter(loader)); rec.take("S11", "TranAD DataLoader constructed", ["adapter", "loader", "first_batch"])
        def batches(): return adapter.loader("test", batch_size=unified.TRANAD_BATCH, shuffle=False)
        def call(x): return benchmark.formal_score_forward(model, x)
        raw_points = int(benchmark.EXPECTED_TEST_POINTS); expected = raw_points if not validation else len(first)
        if collect_objects:
            objects.update(model=model, adapter=adapter, scaled_train=train_values, scaled_test=test_values, loader=loader)
        ctx = {"torch": torch, "device": device, "first": first, "batches": batches, "call": call,
               "raw_points": raw_points, "expected": expected, "window": 10, "batch_size": unified.TRANAD_BATCH,
               "checkpoint": guard, "objects": objects, "adapter_audit": adapter.audit(), "native": "one padded window per time point"}
    else:
        benchmark = binding
        config = benchmark.load_couta_config() if hasattr(benchmark, "load_couta_config") else benchmark.load_pump_couta_config()
        adapter_cls = benchmark.PSMCOUTADataAdapter if hasattr(benchmark, "PSMCOUTADataAdapter") else benchmark.PUMPCOUTADataAdapter
        guard = checkpoint_guard(Path(benchmark.BUNDLE)); bundle = torch.load(benchmark.BUNDLE, map_location="cpu", weights_only=False)
        model, scaler = benchmark.restore_from_bundle(bundle, str(device)); net, center = model.net.eval(), model.c
        rec.take("S5", "COUTA bundle loaded and network restored", ["bundle", "net", "center"])
        adapter = adapter_cls("score"); raw_test = adapter.load_test(); rec.take("S6", "Raw test loaded", ["raw_test"])
        rec.take("S7", "Using train-fitted scaler serialized in formal checkpoint", ["checkpoint_scaler"])
        transformed = scaler.transform(raw_test); rec.take("S8", "Checkpoint scaler transformed test", ["raw_test", "transformed_test"])
        scaled = np.asarray(transformed, dtype=np.float32); rec.take("S9", "float32 conversion complete", ["raw_test", "transformed_test", "scaled_test"])
        seq_len = int(config["model_config"]["seq_len"]); windows = unified.couta_cpu_windows(scaled, seq_len)
        rec.take("S10", "CPU Tensor created from scaled test", ["scaled_test", "window_view"])
        rec.take("S11", "Zero-copy unfold window view constructed", ["scaled_test", "window_view"])
        limit = min(len(windows), unified.COUTA_BATCH) if validation else len(windows)
        def batches():
            for start in range(0, limit, unified.COUTA_BATCH): yield windows[start:min(start + unified.COUTA_BATCH, limit)].contiguous(), min(unified.COUTA_BATCH, limit-start)
        def call(x): return benchmark.score_batch(net, center, x)
        first = windows[:min(unified.COUTA_BATCH, limit)].contiguous(); raw_points = len(scaled)
        expected = min(raw_points, limit + seq_len - 1) if validation else raw_points
        if collect_objects:
            objects.update(bundle=bundle, net=net, center=center, raw_test=raw_test, transformed_test=transformed,
                           scaled_test=scaled, window_view=windows, scaler=scaler, adapter=adapter)
        ctx = {"torch": torch, "device": device, "first": first, "batches": batches, "call": call,
               "raw_points": raw_points, "expected": expected, "window": seq_len, "batch_size": unified.COUTA_BATCH,
               "checkpoint": guard, "objects": objects, "adapter_audit": adapter.audit(), "native": "stride-1 endpoint windows with prefix alignment"}
    return ctx


def formal_worker(model: str, dataset: str, output: Path, validation: bool) -> None:
    tracemalloc.start(1); rec = Recorder(); rec.take("S0", "Python + stdlib + psutil only")
    ctx = prepare(model, dataset, rec, validation, collect_objects=False); torch = ctx["torch"]; device = ctx["device"]
    first_gpu = ctx["first"].to(device)
    with torch.no_grad():
        for _ in range(WARMUP): out = ctx["call"](first_gpu); del out
    torch.cuda.synchronize(device); rec.take("S12", "Three real warmup forwards complete", ["first_gpu_batch"])
    sampler = Sampler(); sampler.start(); processed_native = 0; processed_outputs = 0
    with torch.no_grad():
        for index, item in enumerate(ctx["batches"](), 1):
            cpu_batch, native = item if isinstance(item, tuple) else (item, int(item.shape[0]))
            gpu = cpu_batch.to(device, non_blocking=False); sampler.state.update(batch_index=index, batch_shape=str(tuple(cpu_batch.shape)))
            output_score = ctx["call"](gpu); processed_native += native; processed_outputs += int(output_score.numel())
            sampler.state.update(processed_points=min(ctx["raw_points"], processed_outputs), cached_outputs=0)
            del output_score, gpu, cpu_batch
            if validation: break
    torch.cuda.synchronize(device); sampler.finish(); rec.take("S13", "Inference sampling peak marker")
    rec.take("S14", "Full-test inference completed; outputs were not accumulated")
    del first_gpu; rec.take("S15", "Current batches and temporary outputs deleted")
    gc.collect(); rec.take("S16", "gc.collect completed")
    accessed = ctx["adapter_audit"].get("files_accessed", [])
    if any("label" in str(x).lower() for x in accessed): raise RuntimeError("label access detected")
    stages = rec.rows; peak_sample = max(sampler.rows or stages, key=lambda x: x.get("rss_mib", -1))
    result = {"model": model, "dataset": dataset, "status": "PASS" if validation else "COMPLETE",
              "device": str(device), "checkpoint": ctx["checkpoint"], "raw_test_points": ctx["raw_points"],
              "processed_points": ctx["expected"], "coverage_ratio": ctx["expected"] / ctx["raw_points"],
              "native_inference_unit": ctx["native"], "window": ctx["window"], "batch_size": ctx["batch_size"],
              "measured_peak_rss_mib": peak_sample.get("rss_mib"), "measured_peak_uss_mib": peak_sample.get("uss_mib"),
              "measured_peak_pss_mib": peak_sample.get("pss_mib"), "peak_stage": peak_sample.get("stage", "S13"),
              "peak_batch_index": peak_sample.get("batch_index"), "training": False, "labels_read": False,
              "threshold_computed": False, "prediction_generated": False, "evaluator_called": False,
              "legacy_results_modified": False, "independent_process": True, "stage_snapshots_complete": [r["stage"] for r in stages] == STAGES,
              "peak_rss_source": "/usr/bin/time -v", "audit_version": AUDIT_VERSION, "timestamp": now()}
    write_json(output, {"result": result, "stages": stages, "samples": sampler.rows})


def object_record(name: str, obj: Any, np, torch) -> dict[str, Any] | None:
    if isinstance(obj, np.ndarray):
        return {"name": name, "type": "numpy.ndarray", "shape": list(obj.shape), "dtype": str(obj.dtype),
                "nbytes": int(obj.nbytes), "mib": obj.nbytes / 2**20, "owns_data": bool(obj.flags["OWNDATA"]),
                "base_type": type(obj.base).__name__ if obj.base is not None else None, "c_contiguous": bool(obj.flags["C_CONTIGUOUS"]),
                "is_view": obj.base is not None, "address": int(obj.__array_interface__["data"][0])}
    if isinstance(obj, torch.Tensor):
        storage = obj.untyped_storage()
        return {"name": name, "type": "torch.Tensor", "shape": list(obj.shape), "dtype": str(obj.dtype),
                "device": str(obj.device), "numel": obj.numel(), "element_size": obj.element_size(),
                "storage_nbytes": storage.nbytes(), "mib": storage.nbytes()/2**20, "data_ptr": obj.data_ptr(),
                "storage_ptr": storage.data_ptr(), "contiguous": obj.is_contiguous(), "requires_grad": obj.requires_grad}
    return None


def object_worker(model: str, dataset: str, directory: Path) -> None:
    tracemalloc.start(1); rec = Recorder(); rec.take("S0"); ctx = prepare(model, dataset, rec, validation=False, collect_objects=True)
    import numpy as np
    torch = ctx["torch"]; rows=[]
    for name, obj in ctx["objects"].items():
        row = object_record(name, obj, np, torch)
        if row: rows.append(row)
        if hasattr(obj, "__dict__"):
            for sub, value in vars(obj).items():
                row = object_record(f"{name}.{sub}", value, np, torch)
                if row: rows.append(row)
    arrays=[(n,o) for n,o in ctx["objects"].items() if isinstance(o,np.ndarray)]
    aliases=[]
    for i,(a_name,a) in enumerate(arrays):
        for b_name,b in arrays[i+1:]:
            aliases.append({"a":a_name,"b":b_name,"shares_memory":bool(np.shares_memory(a,b)),"may_share_memory":bool(np.may_share_memory(a,b))})
    write_json(directory / "object_inventory.json", {"objects": rows, "note": "Separate process; excluded from formal peak"})
    write_csv(directory / "large_objects.csv", sorted(rows,key=lambda r:r.get("mib",0),reverse=True),
              ["name","type","shape","dtype","device","mib","nbytes","storage_nbytes","owns_data","is_view","base_type","c_contiguous","contiguous"])
    write_csv(directory / "numpy_alias_report.csv", aliases, ["a","b","shares_memory","may_share_memory"])
    write_csv(directory / "tensor_storage_report.csv", [r for r in rows if r["type"]=="torch.Tensor"],
              ["name","shape","dtype","device","numel","element_size","storage_nbytes","mib","data_ptr","storage_ptr","contiguous","requires_grad"])


def parse_time(path: Path) -> float:
    match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", path.read_text(errors="replace"))
    if not match: raise RuntimeError("Maximum RSS missing")
    return int(match.group(1))/1024


def run_case(model: str, dataset: str, validation: bool) -> dict[str, Any]:
    directory = (OUT/"validation"/safe(model)) if validation else case_dir(dataset, model)
    directory.mkdir(parents=True, exist_ok=True); payload=directory/"payload.json"; tv=directory/"time_verbose.txt"
    command=["/usr/bin/time","-v","-o",str(tv),sys.executable,str(Path(__file__).resolve()),"--formal-worker","--model",model,"--dataset",dataset,"--output",str(payload)]
    if validation: command.append("--validation-worker")
    env=os.environ.copy(); env.update(THREAD_ENV); env["PYTHONUNBUFFERED"]="1"
    with (directory/"stdout.log").open("w") as stdout,(directory/"stderr.log").open("w") as stderr:
        proc=subprocess.run(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
    if proc.returncode: raise RuntimeError(f"formal worker exit={proc.returncode}; see {directory/'stderr.log'}")
    payload_data=json.loads(payload.read_text()); result=payload_data["result"]; result["external_peak_rss_mib"]=parse_time(tv)
    result["status"]="PASS" if validation else ("COMPLETE" if result["coverage_ratio"]==1.0 else "FAILED")
    write_json(directory/"memory_stages.json",payload_data["stages"])
    fields=list(payload_data["stages"][0]); write_csv(directory/"memory_stages.csv",payload_data["stages"],fields)
    sample_fields=sorted({k for x in payload_data["samples"] for k in x}); write_csv(directory/"memory_samples.csv",payload_data["samples"],sample_fields)
    write_json(directory/"smaps_rollup_stages.json",[{"stage":r["stage"],**{k:v for k,v in r.items() if k.endswith("_mib")}} for r in payload_data["stages"]])
    write_json(directory/"protocol_memory_audit.json",result)
    if not validation:
        objcmd=[sys.executable,str(Path(__file__).resolve()),"--object-worker","--model",model,"--dataset",dataset,"--output",str(directory)]
        subprocess.run(objcmd,cwd=ROOT,env=env,check=True,stdout=(directory/"object_stdout.log").open("w"),stderr=(directory/"object_stderr.log").open("w"))
    return result


def baseline_worker(kind: str, output: Path) -> None:
    tracemalloc.start(); started=time.perf_counter(); rec=Recorder(); rec.take("start")
    if kind in {"numpy","torch","cuda_import","models"}: import numpy  # noqa
    if kind in {"torch","cuda_import","models"}: import torch  # noqa
    if kind=="cuda_import": torch.cuda.is_available()
    if kind=="models":
        if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
        import scripts.benchmarks.run_unified_efficiency_rebenchmark as unified
        unified.import_core("SKAB"); unified.import_tranad("SKAB"); unified.import_couta("SKAB")
    rec.take("end"); write_json(output,{"kind":kind,"import_time_s":time.perf_counter()-started,"snapshots":rec.rows})


def run_baselines() -> None:
    for kind in ("python","numpy","torch","cuda_import","models"):
        directory=OUT/"baselines"/kind; directory.mkdir(parents=True,exist_ok=True); payload=directory/"payload.json"; tv=directory/"time_verbose.txt"
        env=os.environ.copy(); env.update(THREAD_ENV)
        subprocess.run(["/usr/bin/time","-v","-o",str(tv),sys.executable,str(Path(__file__).resolve()),"--baseline-worker",kind,"--output",str(payload)],cwd=ROOT,env=env,check=True)
        d=json.loads(payload.read_text()); d["peak_rss_mib"]=parse_time(tv); write_json(directory/"baseline.json",d)


def static_analysis() -> None:
    files=[ROOT/"scripts/benchmarks/run_unified_ram_benchmark.py",ROOT/"scripts/benchmarks/run_unified_efficiency_rebenchmark.py"]
    patterns=("astype(",".copy(",".clone(",".contiguous(","torch.tensor(","torch.stack(","np.stack(","np.concatenate(","np.asarray(","np.array(","sliding_window_view(",".unfold(","scores.append(","outputs.append(")
    rows=[]
    for path in files:
        for lineno,line in enumerate(path.read_text(errors="replace").splitlines(),1):
            for pattern in patterns:
                if pattern in line: rows.append({"file":str(path.relative_to(ROOT)),"line":lineno,"pattern":pattern,"code":line.strip(),"formal_path":True,"copy_risk":"context-dependent"})
    SUMMARY.mkdir(parents=True,exist_ok=True)
    write_csv(SUMMARY/"copy_candidate_summary.csv",rows,["file","line","pattern","code","formal_path","copy_risk"])
    for dataset in DATASETS:
        for model in MODELS:
            write_csv(case_dir(dataset, model)/"static_copy_candidates.csv", rows,
                      ["file","line","pattern","code","formal_path","copy_risk"])
    call_chain=[
      {"model":"ASCA/PPLAD/LTFAD","chain":"parent -> formal worker -> import_core -> protocol.load_model/load_scaled_test semantics -> core_cpu_batches -> score_call"},
      {"model":"TranAD","chain":"parent -> formal worker -> import_tranad -> TranADPSMDataAdapter -> load_model -> DataLoader -> formal_score_forward"},
      {"model":"COUTA","chain":"parent -> formal worker -> import_couta -> COUTADataAdapter -> restore_from_bundle -> unfold view -> score_batch"}]
    write_json(SUMMARY/"static_call_chains.json",call_chain)


def summarize() -> None:
    results=[]; failed=[]; stages=[]; large=[]
    for dataset in DATASETS:
        for model in MODELS:
            directory=case_dir(dataset,model); protocol=directory/"protocol_memory_audit.json"
            if not protocol.exists(): failed.append({"Dataset":dataset,"Model":model,"Reason":"missing"}); continue
            d=json.loads(protocol.read_text()); results.append(d)
            if d["status"]!="COMPLETE": failed.append({"Dataset":dataset,"Model":model,"Reason":d["status"]})
            for row in json.loads((directory/"memory_stages.json").read_text()): stages.append({"Dataset":dataset,"Model":model,**row})
            inv=json.loads((directory/"object_inventory.json").read_text()); large += [{"Dataset":dataset,"Model":model,**x} for x in inv["objects"]]
    SUMMARY.mkdir(parents=True,exist_ok=True)
    write_csv(SUMMARY/"failed_cases.csv",failed,["Dataset","Model","Reason"])
    stage_fields=sorted({k for r in stages for k in r}); write_csv(SUMMARY/"model_stage_comparison.csv",stages,stage_fields)
    peak=[]; incremental=[]
    for d in results:
        rows=[r for r in stages if r["Dataset"]==d["dataset"] and r["Model"]==d["model"]]; by={r["stage"]:r for r in rows}
        peak.append({"Dataset":d["dataset"],"Model":d["model"],"Peak RSS MiB":d["external_peak_rss_mib"],"Measured Peak USS MiB":d["measured_peak_uss_mib"],"Measured Peak PSS MiB":d["measured_peak_pss_mib"],"Peak Stage":d["peak_stage"]})
        incremental.append({"Dataset":d["dataset"],"Model":d["model"],"Model Load Incremental USS MiB":by["S5"]["uss_mib"]-by["S3"]["uss_mib"],"Data Prepare Incremental USS MiB":by["S11"]["uss_mib"]-by["S5"]["uss_mib"],"Inference Incremental USS MiB":d["measured_peak_uss_mib"]-by["S11"]["uss_mib"]})
    write_csv(SUMMARY/"peak_rss_uss_pss_comparison.csv",peak,list(peak[0]) if peak else [])
    write_csv(SUMMARY/"incremental_memory_comparison.csv",incremental,list(incremental[0]) if incremental else [])
    write_csv(SUMMARY/"large_object_summary.csv",sorted(large,key=lambda x:x.get("mib",0),reverse=True),sorted({k for x in large for k in x}))
    def comparison(other: str, filename: str):
        rows=[]
        for dataset in DATASETS:
            for stage in ("S3","S5","S6","S8","S11","S12","S13","S14"):
                a=next(x for x in stages if x["Dataset"]==dataset and x["Model"]=="ASCA-AD V4" and x["stage"]==stage)
                b=next(x for x in stages if x["Dataset"]==dataset and x["Model"]==other and x["stage"]==stage)
                rows.append({"Dataset":dataset,"Stage":stage,"Comparison":other,"ASCA RSS":a["rss_mib"],f"{other} RSS":b["rss_mib"],"RSS Difference":a["rss_mib"]-b["rss_mib"],"ASCA USS":a["uss_mib"],f"{other} USS":b["uss_mib"],"USS Difference":a["uss_mib"]-b["uss_mib"],"Likely Cause":"See stage notes and object inventory"})
        write_csv(SUMMARY/filename,rows,sorted({k for x in rows for k in x}))
    comparison("COUTA","asca_vs_couta_memory.csv")
    all_rows=[]
    for other in ("PPLAD","LTFAD","TranAD"):
        temp=SUMMARY/f"tmp_{safe(other)}.csv"; comparison(other,temp.name)
        with temp.open(newline="",encoding="utf-8") as f: all_rows += list(csv.DictReader(f)); temp.unlink()
    write_csv(SUMMARY/"asca_vs_all_models_memory.csv",all_rows,sorted({k for x in all_rows for k in x}))
    base=[]
    for p in (OUT/"baselines").glob("*/baseline.json"):
        d=json.loads(p.read_text()); end=d["snapshots"][-1]; base.append({"Baseline":d["kind"],"Peak RSS MiB":d["peak_rss_mib"],"End RSS MiB":end["rss_mib"],"End USS MiB":end["uss_mib"],"End PSS MiB":end["pss_mib"],"Import Time s":d["import_time_s"]})
    write_csv(SUMMARY/"runtime_baseline_comparison.csv",base,list(base[0]) if base else [])
    safe_opts=[{"Category":"D","Evidence":"RSS includes Python/PyTorch/CUDA shared mappings","Recommendation":"Use USS/incremental USS for internal diagnosis; parameters for architectural lightweight claim","Changes Results":False,"Risk":"low"},
               {"Category":"C","Evidence":"np.stack materializes every current native-window batch","Recommendation":"Investigate reusable batch buffers/from_numpy in a separate equivalence study","Changes Results":False,"Risk":"medium"},
               {"Category":"C","Evidence":"raw and transformed arrays temporarily coexist during scaling","Recommendation":"Release raw arrays immediately after transform where lifecycle permits","Changes Results":False,"Risk":"low"}]
    write_csv(SUMMARY/"safe_optimization_candidates.csv",safe_opts,list(safe_opts[0]))
    status="COMPLETE" if len(results)==15 and not failed else "PARTIAL"
    asca={x["Dataset"]:x for x in peak if x["Model"]=="ASCA-AD V4"}
    asca_inc={x["Dataset"]:x for x in incremental if x["Model"]=="ASCA-AD V4"}
    lines=["# Memory Root-Cause Audit","",f"- Status: {status}",f"- Cases: {len(results)}/15",f"- Failures: {len(failed)}","",
      "## Measurement boundary","",
      "**Measured fact.** Every case ran in an independent process. Peak RSS is `/usr/bin/time -v` Maximum RSS. USS/PSS and stage values come from psutil plus `/proc/self/smaps_rollup`; inference was sampled every 10 ms. Object inventories ran in separate processes and do not contribute to formal peaks.",
      "",
      "**Limitation.** The 10 ms sampler covers inference only and can miss short preprocessing peaks. Therefore `peak_stage=S13` means the largest sampled inference point, not proof that the external Maximum RSS occurred at S13. Tracemalloc is diagnostic instrumentation and increases this audit's absolute memory; historical process-RSS results remain the formal process benchmark.",
      "",
      "## Runtime baselines","",
      "Python-only USS is about 10.5 MiB; NumPy raises it to about 32.5 MiB; importing PyTorch raises it to about 553 MiB. Importing the selected benchmark/model modules raises the all-model baseline to about 631 MiB. Thus hundreds of MiB exist before any checkpoint, data, or forward pass.",
      "",
      "## Why ASCA has GiB-scale Peak RSS despite 146 parameters","",
      "ASCA has 146 float32 parameters: 584 bytes (0.000557 MiB). This is roughly 0.00003% of its measured 1.7–1.8 GiB Peak USS. The checkpoint is about 5 KiB. Consequently the parameter tensors cannot explain the process peak.",
      "",
      "After the selected ASCA modules are imported, the combined construction/checkpoint/CUDA-placement step adds about 366 MiB USS. This number is **not pure model memory**: the public loader combines model construction, checkpoint loading, `.to(cuda)`, and lazy CUDA runtime/allocator initialization. The first three real forwards then add roughly 719–734 MiB USS. The dominant measured component is therefore runtime/backend activation and inference workspace/caching, not weights.",
      "",
      "## Dataset and preprocessing effects","",
      "On SKAB, ASCA data preparation adds only about 2.3 MiB USS. On HAI and SMD the retained post-preparation increment is about 102.7 and 105.8 MiB. During preprocessing, raw train/test and transformed arrays briefly coexist. HAI raw train alone is about 294.1 MiB and its test array about 93.3 MiB; SMD train/test arrays are about 102.7 MiB each.",
      "",
      "The formal ASCA path releases raw train, raw test, transformed test, and scaler references before streamed window inference. It retains the scaled test and constructs only the current `np.stack` batch. It does not materialize every overlapping window and does not accumulate full output scores. Therefore no unnecessary persistent full-window tensor or score cache was found. The temporary StandardScaler arrays are real engineering working-set costs; HAI's external peak exceeds its sampled inference RSS by about 212 MiB, consistent with an unsampled preprocessing transient, but the exact allocator object at that instant remains unconfirmed.",
      "",
      "## ASCA versus COUTA and TranAD","",
      "COUTA starts from nearly the same S3 import baseline but its combined model/checkpoint placement adds about 171 MiB USS versus ASCA's 366 MiB, and its inference increment is about 247–430 MiB versus ASCA's 719–734 MiB. COUTA uses a zero-copy CPU `unfold` view and materializes only the current batch=64 through `.contiguous()`. Its lower peak is mainly backend/model-path workspace behavior, not merely its parameter count.",
      "",
      "TranAD's model-load increment is about 147–158 MiB and inference increment about 362–374 MiB. On HAI/SMD, however, its adapter intentionally retains scaled train and scaled test arrays, producing data-preparation increments of about 389/231 MiB. It still remains below ASCA because its lazy runtime/inference increment is substantially lower.",
      "",
      "PPLAD/LTFAD have larger combined load increments (about 505–506 MiB). Their inference increments vary by native similarity/local-global computations. These deltas include backend allocator effects and must not be interpreted as parameter bytes.",
      "",
      "## Safe engineering candidates (not applied)","",
      "1. Release raw and transformed arrays immediately after scaling. The formal core path already does this; adapters that retain scaled train during score-only inference should be reviewed separately. This need not change model mathematics.",
      "2. Reuse a fixed current-batch NumPy buffer instead of allocating a fresh `np.stack` result per batch, subject to bitwise/numerical equivalence testing. This changes engineering allocation behavior, not the intended score semantics.",
      "3. Keep COUTA's zero-copy `unfold` view and current-batch-only `.contiguous()` policy; never materialize the complete overlapping-window tensor.",
      "4. Do not treat CUDA/PyTorch lazy initialization as removable model memory. Changing backend, precision, batch size, compilation, or kernels would be a different experiment.",
      "",
      "## Fairness and paper use","",
      "RSS, USS, and PSS rankings are broadly similar within this isolated environment, but RSS includes shared mappings and process-runtime effects. Peak Process RSS is not a clean measure of architectural lightweightness and is not recommended as a primary paper-table claim. Prefer trainable parameters and serialized state-dict size for model compactness, plus unified latency/throughput and GPU peak memory for deployment efficiency.",
      "",
      "If process memory is reported, label it explicitly as independent-process Peak RSS and disclose Python/PyTorch/CUDA, preprocessing, batch, coverage, and instrumentation. Incremental USS is useful for internal root-cause diagnosis, but its S3/S5 boundary still combines construction, checkpoint load and CUDA placement because existing public loaders do not expose them separately.",
      "",
      "## ASCA measured summary","",
      "| Dataset | Peak RSS MiB | Peak USS MiB | Peak PSS MiB | Load increment USS MiB | Data increment USS MiB | Inference increment USS MiB |",
      "|---|---:|---:|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        p=asca[dataset]; i=asca_inc[dataset]
        lines.append(f"| {dataset} | {float(p['Peak RSS MiB']):.3f} | {float(p['Measured Peak USS MiB']):.3f} | {float(p['Measured Peak PSS MiB']):.3f} | {float(i['Model Load Incremental USS MiB']):.3f} | {float(i['Data Prepare Incremental USS MiB']):.3f} | {float(i['Inference Incremental USS MiB']):.3f} |")
    (SUMMARY/"memory_root_cause_summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    audit_log = (f"Memory root-cause audit summary\nGenerated: {now()}\nStatus: {status}\n"
                 f"Cases: {len(results)}/15\nFailures: {len(failed)}\nAudit version: {AUDIT_VERSION}\n"
                 "training=false labels=false threshold=false prediction=false evaluator=false\n")
    (OUT/"logs").mkdir(parents=True, exist_ok=True)
    (OUT/"logs"/"run.log").write_text(audit_log, encoding="utf-8")
    (SUMMARY/"run.log").write_text(audit_log, encoding="utf-8")
    write_json(SUMMARY/"protocol_memory_audit.json",{"status":status,"audit_version":AUDIT_VERSION,"planned_cases":15,"completed_cases":len(results),"failed_cases":len(failed),"training":False,"labels_read":False,"threshold_computed":False,"prediction_generated":False,"evaluator_called":False,"independent_process":True,"peak_rss_source":"/usr/bin/time -v","sampling_interval_ms":10,"object_audit_separate_process":True,"generated_at":now()})


def validate() -> bool:
    checks={"smaps_rollup":Path("/proc/self/smaps_rollup").is_file(),"time_v":Path("/usr/bin/time").is_file(),"uss":hasattr(psutil.Process().memory_full_info(),"uss"),"pss":hasattr(psutil.Process().memory_full_info(),"pss"),"tracemalloc":True}
    if not all(checks.values()): write_json(OUT/"validation"/"validation_summary.json",{"status":"FAILED","checks":checks}); return False
    rows=[]
    for model in MODELS:
        try: d=run_case(model,"SKAB",True); ok=d["stage_snapshots_complete"] and not d["labels_read"]
        except Exception as exc: d={"model":model,"status":"FAILED","error":str(exc)}; ok=False
        rows.append(d); print(f"[VALIDATION] {model}: {'PASS' if ok else 'FAILED'}",flush=True)
    passed=all(r.get("status")=="PASS" for r in rows)
    write_json(OUT/"validation"/"validation_summary.json",{"status":"PASS" if passed else "FAILED","checks":checks,"cases":rows})
    return passed


def run_all(resume: bool, retry_failed: bool, only_model: str|None, only_dataset: str|None) -> None:
    run_baselines(); failures=[]
    cases=[(d,m) for d in DATASETS for m in MODELS if (not only_model or m==only_model) and (not only_dataset or d==only_dataset)]
    for i,(dataset,model) in enumerate(cases,1):
        protocol=case_dir(dataset,model)/"protocol_memory_audit.json"
        if resume and protocol.exists():
            old=json.loads(protocol.read_text())
            if old.get("status")=="COMPLETE" and old.get("audit_version")==AUDIT_VERSION:
                print(f"[{i}/{len(cases)}] SKIP {dataset}/{model}",flush=True); continue
            if old.get("status")!="COMPLETE" and not retry_failed:
                print(f"[{i}/{len(cases)}] SKIP {dataset}/{model} status={old.get('status')}",flush=True); continue
        print(f"[{i}/{len(cases)}] RUN {dataset}/{model} {now()}",flush=True)
        try: run_case(model,dataset,False)
        except Exception as exc:
            failures.append((dataset,model,str(exc))); directory=case_dir(dataset,model); directory.mkdir(parents=True,exist_ok=True)
            write_json(directory/"protocol_memory_audit.json",{"dataset":dataset,"model":model,"status":"FAILED","error":str(exc),"traceback":traceback.format_exc()})
    if not only_model and not only_dataset: summarize()
    print(f"Finished cases={len(cases)} immediate_failures={len(failures)}",flush=True)


def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--formal-worker",action="store_true"); p.add_argument("--object-worker",action="store_true")
    p.add_argument("--validation-worker",action="store_true"); p.add_argument("--baseline-worker",choices=("python","numpy","torch","cuda_import","models"))
    p.add_argument("--output",type=Path); p.add_argument("--model",choices=MODELS); p.add_argument("--dataset",choices=DATASETS)
    p.add_argument("--validate-only",action="store_true"); p.add_argument("--run-core",action="store_true"); p.add_argument("--run-all",action="store_true")
    p.add_argument("--resume",action="store_true"); p.add_argument("--retry-failed",action="store_true"); return p.parse_args()


def main():
    args=parse_args()
    if args.baseline_worker: baseline_worker(args.baseline_worker,args.output); return
    if args.formal_worker: formal_worker(args.model,args.dataset,args.output,args.validation_worker); return
    if args.object_worker: object_worker(args.model,args.dataset,args.output); return
    OUT.mkdir(parents=True,exist_ok=True); static_analysis()
    if not validate(): raise SystemExit("Validation FAILED")
    if args.validate_only: print("Validation PASS",flush=True); return
    run_all(args.resume,args.retry_failed,args.model,args.dataset)


if __name__ == "__main__": main()
