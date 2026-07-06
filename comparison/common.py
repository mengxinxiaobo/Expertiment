
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def resolve_from(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_config(path: Path) -> Tuple[Dict[str, Any], Path]:
    path = path.expanduser().resolve()
    cfg = json.loads(path.read_text(encoding="utf-8"))
    root = resolve_from(path.parent, cfg.get("project_root", "."))
    return cfg, root


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def flatten_dict(value: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, child in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(child, dict):
            out.update(flatten_dict(child, name))
        elif isinstance(child, (str, int, float, bool)) or child is None:
            out[name] = child
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def git_commit(path: Path) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def command_text(command: Sequence[str]) -> str:
    import shlex
    return " ".join(shlex.quote(str(x)) for x in command)


def file_sha256(path: Path, chunk: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def dataset_metadata(dataset_dir: Path, prefix: str, hash_files: bool = False) -> Dict[str, Any]:
    output: Dict[str, Any] = {"directory": str(dataset_dir)}
    for role, suffix in (
        ("train", "train.npy"),
        ("test", "test.npy"),
        ("label", "test_label.npy"),
    ):
        path = dataset_dir / f"{prefix}_{suffix}"
        if not path.exists():
            raise FileNotFoundError(path)
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        info = {
            "path": str(path),
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "bytes": int(path.stat().st_size),
            "mtime": path.stat().st_mtime,
        }
        if hash_files:
            info["sha256"] = file_sha256(path)
        if role == "label":
            flat = np.asarray(arr).reshape(-1)
            info["positive_count"] = int((flat > 0).sum())
            info["positive_ratio"] = float((flat > 0).mean())
        output[role] = info
    return output


def environment_metadata(project_root: Path, v4_root: Path, original_root: Path) -> Dict[str, Any]:
    import torch
    metadata: Dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "project_root": str(project_root),
        "v4_root": str(v4_root),
        "original_root": str(original_root),
        "project_git_commit": git_commit(project_root),
        "v4_git_commit": git_commit(v4_root),
        "original_git_commit": git_commit(original_root),
    }
    if torch.cuda.is_available():
        metadata["gpu_name"] = torch.cuda.get_device_name(0)
        metadata["gpu_count"] = torch.cuda.device_count()
    try:
        metadata["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=10,
        ).strip()
    except Exception as exc:
        metadata["nvidia_smi_error"] = str(exc)
    return metadata


def process_tree_pids(root_pid: int) -> List[int]:
    try:
        import psutil
        root = psutil.Process(root_pid)
        return [root.pid] + [p.pid for p in root.children(recursive=True)]
    except Exception:
        return [root_pid]


def rss_tree_bytes(root_pid: int) -> int:
    try:
        import psutil
        total = 0
        root = psutil.Process(root_pid)
        procs = [root] + root.children(recursive=True)
        for proc in procs:
            try:
                total += int(proc.memory_info().rss)
            except Exception:
                pass
        return total
    except Exception:
        status = Path(f"/proc/{root_pid}/status")
        if not status.exists():
            return 0
        for line in status.read_text(errors="ignore").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
        return 0


def gpu_memory_for_pids(pids: Iterable[int]) -> int:
    wanted = set(int(x) for x in pids)
    try:
        text = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except Exception:
        return 0
    total_mib = 0
    for line in text.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
            mem = int(float(parts[1]))
        except ValueError:
            continue
        if pid in wanted:
            total_mib += mem
    return total_mib * 1024 * 1024


class ResourceMonitor:
    def __init__(self, pid: int, interval: float = 0.1) -> None:
        self.pid = int(pid)
        self.interval = float(interval)
        self.peak_rss_bytes = 0
        self.peak_gpu_bytes = 0
        self.samples = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            pids = process_tree_pids(self.pid)
            self.peak_rss_bytes = max(self.peak_rss_bytes, rss_tree_bytes(self.pid))
            self.peak_gpu_bytes = max(self.peak_gpu_bytes, gpu_memory_for_pids(pids))
            self.samples += 1
            self._stop.wait(self.interval)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(2.0, self.interval * 5))

    def as_dict(self) -> Dict[str, Any]:
        mib = 1024 ** 2
        return {
            "peak_process_tree_rss_bytes": int(self.peak_rss_bytes),
            "peak_process_tree_rss_mib": float(self.peak_rss_bytes / mib),
            "peak_process_tree_gpu_bytes": int(self.peak_gpu_bytes),
            "peak_process_tree_gpu_mib": float(self.peak_gpu_bytes / mib),
            "resource_samples": int(self.samples),
            "resource_poll_seconds": self.interval,
        }


def run_logged(
    command: Sequence[str],
    cwd: Path,
    log_path: Path,
    env: Dict[str, str],
    poll_seconds: float,
) -> Dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    process = subprocess.Popen(
        [str(x) for x in command],
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    monitor = ResourceMonitor(process.pid, poll_seconds)
    monitor.start()
    chunks: List[str] = []
    try:
        assert process.stdout is not None
        with log_path.open("w", encoding="utf-8") as log:
            for line in process.stdout:
                print(line, end="")
                log.write(line)
                chunks.append(line)
        return_code = process.wait()
    finally:
        monitor.stop()
    elapsed = time.perf_counter() - start
    result = monitor.as_dict()
    result.update(
        {
            "command": command_text(command),
            "cwd": str(cwd),
            "return_code": int(return_code),
            "wall_seconds": float(elapsed),
            "log_path": str(log_path),
            "stdout": "".join(chunks),
        }
    )
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    return result


def parse_threshold(text: str) -> float:
    patterns = [
        r"Threshold\s*:\s*([-+0-9.eE]+)",
        r"Threshold\s*=\s*([-+0-9.eE]+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return float(matches[-1])
    raise RuntimeError("没有在日志中解析到 Threshold。")


def parse_training_times(text: str) -> Dict[str, Any]:
    original = [
        float(x)
        for x in re.findall(r"Epoch\s*:\s*\d+\s*,\s*Cost time\s*:\s*([0-9.]+)s", text)
    ]
    v4 = [
        float(x)
        for x in re.findall(r"Epoch\s*\[[^\]]+\].*?time=([0-9.]+)s", text)
    ]
    values = original or v4
    return {
        "epoch_times_s": values,
        "epoch_count_parsed": len(values),
        "training_epoch_time_total_s": float(sum(values)) if values else None,
        "training_epoch_time_mean_s": float(sum(values) / len(values)) if values else None,
    }


def point_adjust(raw_pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred = np.asarray(raw_pred, dtype=np.int64).copy()
    gt = np.asarray(gt, dtype=np.int64)
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            j = i
            while j >= 0 and gt[j] == 1:
                pred[j] = 1
                j -= 1
            j = i
            while j < len(gt) and gt[j] == 1:
                pred[j] = 1
                j += 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return pred


def evaluate_scores(
    score_path: Path,
    label_path: Path,
    threshold: float,
    project_root: Path,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        matthews_corrcoef,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    score = np.loadtxt(score_path, dtype=np.float64).reshape(-1)
    gt = np.loadtxt(label_path, dtype=np.int64).reshape(-1)
    if len(score) != len(gt):
        raise ValueError(f"分数与标签长度不一致：{len(score)} != {len(gt)}")
    if not np.isfinite(score).all():
        raise ValueError("异常分数包含 NaN 或 Inf。")

    raw = (score > float(threshold)).astype(np.int64)
    pa = point_adjust(raw, gt)

    def classification(prefix: str, pred: np.ndarray) -> Dict[str, Any]:
        precision, recall, f1, _ = precision_recall_fscore_support(
            gt, pred, average="binary", zero_division=0
        )
        tn, fp, fn, tp = confusion_matrix(gt, pred, labels=[0, 1]).ravel()
        return {
            f"{prefix}_accuracy": float(accuracy_score(gt, pred)),
            f"{prefix}_precision": float(precision),
            f"{prefix}_recall": float(recall),
            f"{prefix}_f1": float(f1),
            f"{prefix}_mcc": float(matthews_corrcoef(gt, pred)),
            f"{prefix}_tn": int(tn),
            f"{prefix}_fp": int(fp),
            f"{prefix}_fn": int(fn),
            f"{prefix}_tp": int(tp),
            f"{prefix}_predicted_anomaly_ratio": float(pred.mean()),
        }

    result: Dict[str, Any] = {
        "threshold": float(threshold),
        "score_count": int(len(score)),
        "ground_truth_anomaly_ratio": float(gt.mean()),
        "score_min": float(score.min()),
        "score_max": float(score.max()),
        "score_mean": float(score.mean()),
        "score_std": float(score.std()),
    }
    result.update(classification("raw", raw))
    result.update(classification("pa", pa))

    try:
        result["roc_auc"] = float(roc_auc_score(gt, score))
    except Exception:
        result["roc_auc"] = None
    try:
        result["pr_auc_average_precision"] = float(average_precision_score(gt, score))
    except Exception:
        result["pr_auc_average_precision"] = None

    # 调用当前项目已有的 Affiliation、R-AUC、VUS 等实现。
    sys.path.insert(0, str(project_root))
    try:
        from metrics.metrics import combine_all_evaluation_scores
        extra = combine_all_evaluation_scores(raw, gt, score)
        for key, value in extra.items():
            normalized = (
                key.strip().lower().replace(" ", "_").replace("-", "_")
            )
            result[f"project_{normalized}"] = (
                float(value) if value is not None and np.isscalar(value) else value
            )
    except Exception as exc:
        result["project_metrics_error"] = repr(exc)
    finally:
        try:
            sys.path.remove(str(project_root))
        except ValueError:
            pass

    return result, raw, pa


def link_dataset(source: Path, original_root: Path, link_name: str) -> Path:
    target_parent = original_root / "dataset"
    target_parent.mkdir(parents=True, exist_ok=True)
    target = target_parent / link_name
    if target.is_symlink():
        if target.resolve() == source.resolve():
            return target
        target.unlink()
    elif target.exists():
        required = list(target.glob("*_train.npy"))
        if required:
            return target
        raise RuntimeError(
            f"{target} 已存在但不是可用数据目录。请手动删除空目录后重试。"
        )
    try:
        target.symlink_to(source.resolve(), target_is_directory=True)
    except OSError as exc:
        raise RuntimeError(
            f"无法创建数据软链接 {target} -> {source}。"
            "请在 WSL 中运行，或手动创建该链接。"
        ) from exc
    return target
