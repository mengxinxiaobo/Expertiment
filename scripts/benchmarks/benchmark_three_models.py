#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def resolve_pplad_class(root: Path):
    """Find the actual PPLAD class in the current repository.

    The refactored repository's root solver.py may refer to PPLAD without
    exporting it as solver.PPLAD, so do not assume that attribute exists.
    """
    candidates = [
        root / "model" / "PPLAD.py",
        root / "model" / "pplad.py",
        root / "models" / "PPLAD.py",
        root / "models" / "pplad.py",
        root / "legacy" / "model" / "PPLAD.py",
        root / "BaselineModels" / "PPLAD-main" / "model" / "PPLAD.py",
        root / "BaselineModels" / "PPLAD" / "model" / "PPLAD.py",
    ]

    checked = []
    for path in candidates:
        checked.append(str(path))
        if not path.exists():
            continue
        module = load_module(f"resolved_pplad_{len(checked)}", path)
        if hasattr(module, "PPLAD"):
            return module.PPLAD, path

    # Fallback: scan Python files whose name or contents suggest PPLAD.
    for path in root.rglob("*.py"):
        if any(part in {".git", "__pycache__", ".venv", "venv"} for part in path.parts):
            continue
        try:
            source = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "class PPLAD" not in source:
            continue
        module = load_module(f"resolved_pplad_scan_{abs(hash(path))}", path)
        if hasattr(module, "PPLAD"):
            return module.PPLAD, path

    raise FileNotFoundError(
        "Could not locate a Python class named PPLAD.\n"
        "Checked common paths and scanned the repository.\n"
        "Run this command and send the output:\n"
        f"  grep -R \"class PPLAD\" -n {root}/model {root}/BaselineModels 2>/dev/null"
    )


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def state_dict_kib(model: nn.Module) -> float:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return len(buffer.getvalue()) / 1024.0


def find_file(folder: Path, dataset: str, suffixes: list[str]) -> Path:
    for suffix in suffixes:
        for name in (dataset, dataset.upper(), dataset.lower()):
            path = folder / f"{name}_{suffix}.npy"
            if path.exists():
                return path
    raise FileNotFoundError(f"Missing {suffixes} under {folder}")


def load_arrays(folder: Path, dataset: str):
    train_path = find_file(folder, dataset, ["train"])
    test_path = find_file(folder, dataset, ["test"])
    label_path = find_file(folder, dataset, ["test_label", "test_labels", "label", "labels"])
    train = np.nan_to_num(np.load(train_path)).astype(np.float32)
    test = np.nan_to_num(np.load(test_path)).astype(np.float32)
    labels = np.load(label_path).reshape(-1).astype(np.int64)
    if train.ndim == 1:
        train = train[:, None]
    if test.ndim == 1:
        test = test[:, None]
    scaler = StandardScaler()
    train = scaler.fit_transform(train).astype(np.float32)
    test = scaler.transform(test).astype(np.float32)
    return train, test, labels


class Windows(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray | None, win: int, step: int):
        self.x, self.y, self.win = x, y, win
        self.starts = list(range(0, len(x) - win + 1, step))
        last = len(x) - win
        if self.starts[-1] != last:
            self.starts.append(last)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        start = self.starts[idx]
        labels = np.zeros(self.win, dtype=np.int64) if self.y is None else self.y[start:start+self.win]
        return torch.from_numpy(self.x[start:start+self.win]), torch.from_numpy(labels), start


def instance_norm(x: torch.Tensor, eps: float = 1e-5):
    mean = x.mean(1, keepdim=True).detach()
    var = x.var(1, keepdim=True, unbiased=False).detach()
    return (x - mean) / torch.sqrt(var + eps)


def normalize_score(score: torch.Tensor):
    lo = score.min(-1, keepdim=True).values
    hi = score.max(-1, keepdim=True).values
    return torch.softmax((score - lo) / (hi - lo + 1e-5), dim=-1)


def overlap_average(scores, starts, total_length, win):
    total = np.zeros(total_length, dtype=np.float64)
    count = np.zeros(total_length, dtype=np.float64)
    for score, start in zip(scores, starts):
        end = min(start + win, total_length)
        total[start:end] += score[:end-start]
        count[start:end] += 1
    if np.any(count == 0):
        raise RuntimeError("Some timestamps were not scored")
    return total / count


def point_adjust(pred: np.ndarray, gt: np.ndarray):
    pred = pred.copy()
    active = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not active:
            active = True
            j = i
            while j >= 0 and gt[j] == 1:
                pred[j] = 1
                j -= 1
            j = i
            while j < len(gt) and gt[j] == 1:
                pred[j] = 1
                j += 1
        elif gt[i] == 0:
            active = False
        if active:
            pred[i] = 1
    return pred


def pa_metrics(gt, pred):
    pred = point_adjust(pred, gt)
    precision, recall, f1, _ = precision_recall_fscore_support(
        gt, pred, average="binary", zero_division=0
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pa_accuracy": float(accuracy_score(gt, pred)),
    }


def symmetric_neighborhood(x: torch.Tensor, local_size: int, global_size: int):
    B, L, C = x.shape
    total = local_size + global_size
    front = total // 2
    back = total - front
    offsets = torch.arange(-front, back, device=x.device)
    pos = torch.arange(L, device=x.device)
    idx = (pos[:, None] + offsets[None, :]).clamp(0, L - 1)
    values = x[:, idx, :].permute(0, 1, 3, 2).contiguous()
    distance = ((values - x.unsqueeze(-1)) ** 2).sum(dim=2)
    relation = torch.softmax(distance, dim=-1)
    site = total // 2
    lf = local_size // 2
    lb = local_size - lf
    local = relation[:, :, site-lf:site+lb]
    global_ = torch.cat([relation[:, :, :site-lf], relation[:, :, site+lb:]], dim=-1)
    return local, global_, relation


def ltfad_neighbors(x, local_sizes, global_sizes):
    B, L, C = x.shape
    pos = torch.arange(L, device=x.device)
    locals_, globals_ = [], []
    for ls, gs in zip(local_sizes, global_sizes):
        lf, lb = ls // 2, ls - ls // 2
        loff = torch.arange(-lf, lb, device=x.device)
        lidx = (pos[:, None] + loff[None, :]).clamp(0, L-1)
        locals_.append(x[:, lidx, :].permute(0, 1, 3, 2).contiguous())
        total = ls + gs
        gf, gb = total // 2, total - total // 2
        alloff = torch.arange(-gf, gb, device=x.device)
        mask = (alloff >= -lf) & (alloff < lb)
        goff = alloff[~mask]
        gidx = (pos[:, None] + goff[None, :]).clamp(0, L-1)
        globals_.append(x[:, gidx, :].permute(0, 1, 3, 2).contiguous())
    return locals_, globals_


class ASCAAdapter(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def loss(self, x):
        _, d = self.model(instance_norm(x))
        fit = d["local_fit"].mean() + d["global_fit"].mean()
        area = d["area_error"].mean()
        def cov(g, k):
            expected = float(k) / g.shape[-1]
            return (g.mean((0, 1)) - expected).square().mean()
        return fit + 0.1 * area + 0.05 * (
            cov(d["local_gate"], self.model.local_topk)
            + cov(d["global_gate"], self.model.global_topk)
        )

    def score(self, x):
        _, d = self.model(instance_norm(x))
        return d["score_combined"]


class LTFADAdapter(nn.Module):
    def __init__(self, model, local_sizes, global_sizes, r):
        super().__init__()
        self.model, self.local_sizes, self.global_sizes, self.r = model, local_sizes, global_sizes, r

    def outputs(self, x):
        x = instance_norm(x)
        local, global_ = ltfad_neighbors(x, self.local_sizes, self.global_sizes)
        a, b, c, d = self.model(local, global_)
        return local, global_, a, b, c, d

    def loss(self, x):
        local, global_, a, b, c, d = self.outputs(x)
        vals = [0.0] * 6
        for i in range(len(b)):
            vals[0] += ((a[i] - global_[i]) ** 2).mean()
            vals[1] += ((b[i] - local[i]) ** 2).mean()
            vals[2] += ((c[i] - global_[i]) ** 2).mean()
            vals[3] += ((d[i] - local[i]) ** 2).mean()
            vals[4] += ((a[i] - c[i]) ** 2).mean()
            vals[5] += ((b[i] - d[i]) ** 2).mean()
        n = len(b)
        return self.r * (vals[0] + vals[2] + vals[4]) / n + (1-self.r) * (vals[1] + vals[3] + vals[5]) / n

    def score(self, x):
        local, global_, a, b, c, d = self.outputs(x)
        vals = [0.0] * 6
        for i in range(len(b)):
            vals[0] += ((a[i] - global_[i]) ** 2).sum(-1)
            vals[1] += ((b[i] - local[i]) ** 2).sum(-1)
            vals[2] += ((c[i] - global_[i]) ** 2).sum(-1)
            vals[3] += ((d[i] - local[i]) ** 2).sum(-1)
            vals[4] += ((a[i] - c[i]) ** 2).sum(-1)
            vals[5] += ((b[i] - d[i]) ** 2).sum(-1)
        vals = [v.max(-1).values for v in vals]
        return self.r * (vals[0]+vals[2]+vals[4]) + (1-self.r) * (vals[1]+vals[3]+vals[5])


class PPLADAdapter(nn.Module):
    def __init__(self, model, local_size=5, global_size=13, r=0.1):
        super().__init__()
        self.model, self.local_size, self.global_size, self.r = model, local_size, global_size, r

    def outputs(self, x, op="train"):
        x = instance_norm(x)
        local, global_, relation = symmetric_neighborhood(x, self.local_size, self.global_size)
        out = self.model(x, [local], [global_], op, 0, relation)
        return local, global_, out

    def loss(self, x):
        local, global_, out = self.outputs(x, "train")
        series, prior, _, _, area_local, area_global, _, _ = out
        fit = ((series[0] - local) ** 2).mean() + ((prior[0] - global_) ** 2).mean()
        area = (1 + area_local[0]).sum() + (1 + 2 * area_global[0]).sum()
        return self.r * fit + (1-self.r) * area

    def score(self, x):
        local, global_, out = self.outputs(x, "test")
        series, prior = out[0], out[1]
        s = ((series[0] - local) ** 2).sum(-1)
        p = ((prior[0] - global_) ** 2).sum(-1)
        return s + p


def train_model(name, model, loader, optimizer, epochs, device):
    for epoch in range(epochs):
        model.train()
        sync()
        start = time.perf_counter()
        total, batches = 0.0, 0
        for x, _, _ in loader:
            x = x.float().to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        sync()
        print(f"[{name}] epoch {epoch+1}/{epochs} loss={total/max(batches,1):.6f} time={time.perf_counter()-start:.3f}s")


@torch.no_grad()
def collect_scores(model, loader, total_len, win, device):
    model.eval()
    scores, starts = [], []
    for x, _, s in loader:
        x = x.float().to(device)
        score = normalize_score(model.score(x))
        scores.extend(score.cpu().numpy())
        starts.extend(s.numpy().tolist())
    return overlap_average(scores, starts, total_len, win)


def latency(model, batch_size, win, channels, device, warmup=30, repeats=200):
    x = torch.randn(batch_size, win, channels, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model.score(x)
        sync()
        start = time.perf_counter()
        for _ in range(repeats):
            model.score(x)
        sync()
    return (time.perf_counter() - start) * 1000.0 / repeats


@torch.no_grad()
def benchmark_and_evaluate(name, model, train_eval_loader, test_loader, train_len, test_len,
                           labels, win, ratio, channels, device):
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device) / 1024**2
    else:
        baseline = float("nan")

    train_scores = collect_scores(model, train_eval_loader, train_len, win, device)

    sync()
    start = time.perf_counter()
    test_scores = collect_scores(model, test_loader, test_len, win, device)
    sync()
    full_test_time = time.perf_counter() - start

    threshold = float(np.percentile(np.concatenate([train_scores, test_scores]), 100-ratio))
    pred = (test_scores > threshold).astype(np.int64)
    metrics = pa_metrics(labels, pred)

    b1 = latency(model, 1, win, channels, device)
    b128 = latency(model, 128, win, channels, device)
    peak = torch.cuda.max_memory_allocated(device) / 1024**2 if torch.cuda.is_available() else float("nan")
    incremental = peak - baseline if torch.cuda.is_available() else float("nan")
    throughput = test_len / full_test_time

    return {
        "model": name,
        **metrics,
        "parameters": count_params(model),
        "state_dict_kib": state_dict_kib(model),
        "latency_batch1_ms": b1,
        "latency_batch128_ms": b128,
        "full_test_time_s": full_test_time,
        "throughput_points_s": throughput,
        "gpu_peak_mib": peak,
        "gpu_incremental_mib": incremental,
        "anomaly_ratio": ratio,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--anomaly-ratio", type=float, default=0.3)
    p.add_argument("--train-step", type=int, default=1)
    p.add_argument("--eval-step", type=int, default=90)
    p.add_argument("--epochs", type=int, default=2)
    args = p.parse_args()

    seed_all(args.seed)
    device = torch.device(args.device)
    train, test, labels = load_arrays(args.dataset_dir, "SKAB")
    channels = train.shape[1]
    win = 90

    train_ds = Windows(train, None, win, args.train_step)
    train_eval_ds = Windows(train, None, win, args.eval_step)
    test_ds = Windows(test, labels, win, args.eval_step)
    common = dict(batch_size=args.batch_size, num_workers=0, drop_last=False,
                  pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_ds, shuffle=True, **common)
    train_eval_loader = DataLoader(train_eval_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)

    # PPLAD: resolve the model class from its actual source file.
    PPLADClass, pplad_source = resolve_pplad_class(ROOT)
    print(f"Resolved PPLAD class from: {pplad_source}")
    print("PPLAD config: d_model=128, local_size=[3], global_size=[20]")
    pplad_base = PPLADClass(
        batch_size=args.batch_size,
        win_size=win,
        enc_in=channels,
        c_out=channels,
        d_model=128,
        local_size=[3],
        global_size=[20],
        channel=channels,
    )
    pplad = PPLADAdapter(pplad_base, 3, 20, 0.1).to(device)

    expected_pplad_params = 3201
    actual_pplad_params = count_params(pplad)
    if actual_pplad_params != expected_pplad_params:
        raise RuntimeError(
            f"PPLAD parameter mismatch: expected {expected_pplad_params}, "
            f"got {actual_pplad_params}. Refusing to run an invalid comparison."
        )

    # ASCA-AD V4
    asca_mod = load_module("asca_model_module", ROOT / "asca_ad" / "model.py")
    asca_base = asca_mod.AdaptiveSparseAnchorCompetitiveModelV4(
        [1,2,3,4,5,6,7,8], [12,16,20,24,28,32,40,48],
        2, 4, 8, 8, 0.5, 1.0, 0.03, 1.5, 1.0
    )
    asca = ASCAAdapter(asca_base).to(device)

    # LTFAD
    ltfad_mod = load_module(
        "ltfad_model_module",
        ROOT / "BaselineModels" / "LTFAD-main" / "model" / "LTFAD.py"
    )
    ltfad_base = ltfad_mod.LTFAD(win, 128, [5], [13], channels)
    ltfad = LTFADAdapter(ltfad_base, [5], [13], 0.1).to(device)

    models = [
        ("PPLAD", pplad, 1e-4),
        ("ASCA-AD", asca, 1e-3),
        ("LTFAD", ltfad, 1e-4),
    ]

    results = []
    for name, model, lr in models:
        print("\n" + "="*80)
        print(f"{name}: {count_params(model):,} parameters")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        train_model(name, model, train_loader, optimizer, args.epochs, device)
        result = benchmark_and_evaluate(
            name, model, train_eval_loader, test_loader,
            len(train), len(test), labels, win,
            args.anomaly_ratio, channels, device
        )
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        del optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    columns = list(results[0].keys())
    with (args.output_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)

    lines = [
        "| 模型 | Precision | Recall | F1 | PA-Accuracy | 参数量 | State Dict (KiB) | batch=1 (ms) | batch=128 (ms) | 完整测试集时间 (s) | Throughput (points/s) | GPU峰值显存 (MiB) | GPU增量显存 (MiB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['model']} | {r['precision']:.4f} | {r['recall']:.4f} | "
            f"{r['f1']:.4f} | {r['pa_accuracy']:.4f} | {r['parameters']:,} | "
            f"{r['state_dict_kib']:.2f} | {r['latency_batch1_ms']:.4f} | "
            f"{r['latency_batch128_ms']:.4f} | {r['full_test_time_s']:.6f} | "
            f"{r['throughput_points_s']:.2f} | {r['gpu_peak_mib']:.2f} | "
            f"{r['gpu_incremental_mib']:.2f} |"
        )
    (args.output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
