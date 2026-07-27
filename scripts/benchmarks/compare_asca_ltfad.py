#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, importlib.util, json, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()

def params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def load_module(name, path):
    spec=importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None: raise RuntimeError(f"Cannot import {path}")
    mod=importlib.util.module_from_spec(spec); sys.modules[name]=mod; spec.loader.exec_module(mod); return mod

def find_file(d, dataset, kind):
    aliases={"train":["train"],"test":["test"],"test_label":["test_label","test_labels","label","labels"]}[kind]
    for a in aliases:
        for n in (dataset, dataset.upper(), dataset.lower()):
            p=d/f"{n}_{a}.npy"
            if p.exists(): return p
    raise FileNotFoundError(f"Missing {kind} under {d}")

def load_arrays(d, dataset):
    pt, pv, pl = find_file(d,dataset,"train"), find_file(d,dataset,"test"), find_file(d,dataset,"test_label")
    tr=np.nan_to_num(np.load(pt)).astype("float32"); te=np.nan_to_num(np.load(pv)).astype("float32")
    y=np.load(pl).reshape(-1).astype("int64")
    if tr.ndim==1: tr=tr[:,None]
    if te.ndim==1: te=te[:,None]
    s=StandardScaler(); tr=s.fit_transform(tr).astype("float32"); te=s.transform(te).astype("float32")
    return tr,te,y,pt,pv,pl

class Windows(Dataset):
    def __init__(self,x,y,w,step):
        self.x=x; self.y=y; self.w=w
        self.starts=list(range(0,len(x)-w+1,step))
        last=len(x)-w
        if self.starts[-1]!=last: self.starts.append(last)
    def __len__(self): return len(self.starts)
    def __getitem__(self,i):
        s=self.starts[i]; y=np.zeros(self.w,dtype="int64") if self.y is None else self.y[s:s+self.w]
        return torch.from_numpy(self.x[s:s+self.w]), torch.from_numpy(y), s

def inst_norm(x,eps=1e-5):
    m=x.mean(1,keepdim=True).detach(); v=x.var(1,keepdim=True,unbiased=False).detach()
    return (x-m)/torch.sqrt(v+eps)

def norm_score(s):
    lo=s.min(-1,keepdim=True).values; hi=s.max(-1,keepdim=True).values
    return torch.softmax((s-lo)/(hi-lo+1e-5),dim=-1)

def point_adjust(pred,gt):
    p=pred.copy(); active=False
    for i in range(len(gt)):
        if gt[i]==1 and p[i]==1 and not active:
            active=True; j=i
            while j>=0 and gt[j]==1: p[j]=1; j-=1
            j=i
            while j<len(gt) and gt[j]==1: p[j]=1; j+=1
        elif gt[i]==0: active=False
        if active: p[i]=1
    return p

def metrics(gt,p):
    pr,rc,f1,_=precision_recall_fscore_support(gt,p,average="binary",zero_division=0)
    return {"accuracy":float(accuracy_score(gt,p)),"precision":float(pr),"recall":float(rc),"f1":float(f1),"mcc":float(matthews_corrcoef(gt,p))}

def overlap(scores,starts,n,w):
    sm=np.zeros(n); ct=np.zeros(n)
    for s,a in zip(scores,starts):
        e=min(a+w,n); sm[a:e]+=s[:e-a]; ct[a:e]+=1
    if np.any(ct==0): raise RuntimeError("Unscored timestamps")
    return sm/ct

def ltfad_neighbors(x,local_sizes,global_sizes):
    B,L,C=x.shape; pos=torch.arange(L,device=x.device); loc=[]; glob=[]
    for ls,gs in zip(local_sizes,global_sizes):
        lf=ls//2; lb=ls-lf
        off=torch.arange(-lf,lb,device=x.device); idx=(pos[:,None]+off[None,:]).clamp(0,L-1)
        loc.append(x[:,idx,:].permute(0,1,3,2).contiguous())
        total=ls+gs; gf=total//2; gb=total-gf
        alloff=torch.arange(-gf,gb,device=x.device); mask=(alloff>=-lf)&(alloff<lb); goff=alloff[~mask]
        gidx=(pos[:,None]+goff[None,:]).clamp(0,L-1)
        glob.append(x[:,gidx,:].permute(0,1,3,2).contiguous())
    return loc,glob

class ASCA(nn.Module):
    def __init__(self,m): super().__init__(); self.m=m
    def score(self,x):
        _,d=self.m(inst_norm(x)); return d["score_combined"]
    def loss(self,x,aw,bw):
        _,d=self.m(inst_norm(x)); fit=d["local_fit"].mean()+d["global_fit"].mean(); area=d["area_error"].mean()
        def cov(g,k):
            exp=float(k)/g.shape[-1]; return (g.mean((0,1))-exp).square().mean()
        return fit+aw*area+bw*(cov(d["local_gate"],self.m.local_topk)+cov(d["global_gate"],self.m.global_topk))

class LTFAD(nn.Module):
    def __init__(self,m,ls,gs,r): super().__init__(); self.m=m; self.ls=ls; self.gs=gs; self.r=r
    def out(self,x):
        l,g=ltfad_neighbors(inst_norm(x),self.ls,self.gs); a,b,c,d=self.m(l,g); return l,g,a,b,c,d
    def loss(self,x):
        l,g,a,b,c,d=self.out(x); vals=[0.,0.,0.,0.,0.,0.]
        for i in range(len(b)):
            vals[0]+=((a[i]-g[i])**2).mean(); vals[1]+=((b[i]-l[i])**2).mean()
            vals[2]+=((c[i]-g[i])**2).mean(); vals[3]+=((d[i]-l[i])**2).mean()
            vals[4]+=((a[i]-c[i])**2).mean(); vals[5]+=((b[i]-d[i])**2).mean()
        n=len(b); return self.r*(vals[0]+vals[2]+vals[4])/n+(1-self.r)*(vals[1]+vals[3]+vals[5])/n
    def score(self,x):
        l,g,a,b,c,d=self.out(x); vals=[0.,0.,0.,0.,0.,0.]
        for i in range(len(b)):
            vals[0]+=((a[i]-g[i])**2).sum(-1); vals[1]+=((b[i]-l[i])**2).sum(-1)
            vals[2]+=((c[i]-g[i])**2).sum(-1); vals[3]+=((d[i]-l[i])**2).sum(-1)
            vals[4]+=((a[i]-c[i])**2).sum(-1); vals[5]+=((b[i]-d[i])**2).sum(-1)
        vals=[v.max(-1).values for v in vals]
        return self.r*(vals[0]+vals[2]+vals[4])+(1-self.r)*(vals[1]+vals[3]+vals[5])

def train(name,m,loader,opt,epochs,dev,aw,bw):
    times=[]
    for ep in range(epochs):
        m.train(); sync(); st=time.perf_counter(); tot=0.; nb=0
        for x,_,_ in loader:
            x=x.float().to(dev,non_blocking=True); opt.zero_grad(set_to_none=True)
            loss=m.loss(x,aw,bw) if isinstance(m,ASCA) else m.loss(x)
            loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),5.0); opt.step()
            tot+=loss.item(); nb+=1
        sync(); t=time.perf_counter()-st; times.append(t)
        print(f"[{name}] epoch {ep+1}/{epochs} loss={tot/max(nb,1):.6f} time={t:.3f}s")
    return times

@torch.no_grad()
def collect(m,loader,n,w,dev):
    m.eval(); ws=[]; starts=[]
    for x,_,s in loader:
        x=x.float().to(dev,non_blocking=True); ws.extend(norm_score(m.score(x)).cpu().numpy()); starts.extend(s.numpy().tolist())
    return overlap(ws,starts,n,w)

def run_eval(name,m,tr_eval,te_loader,ntr,nte,y,w,ratio,dev,etimes):
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats(dev)
    trs=collect(m,tr_eval,ntr,w,dev); sync(); st=time.perf_counter(); tes=collect(m,te_loader,nte,w,dev); sync(); tt=time.perf_counter()-st
    th=float(np.percentile(np.concatenate([trs,tes]),100-ratio)); pred=(tes>th).astype("int64")
    raw=metrics(y,pred); pa=metrics(y,point_adjust(pred,y))
    peak=torch.cuda.max_memory_allocated(dev)/1024**2 if torch.cuda.is_available() else float("nan")
    return {"model":name,"parameters":params(m),"epoch_times":etimes,"mean_epoch_time":float(np.mean(etimes)),
            "test_time":tt,"peak_memory_mib":peak,"threshold":th,"raw":raw,"pa":pa}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="SKAB")
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--models", nargs="+", choices=["asca", "ltfad"], default=["asca", "ltfad"])

    # ASCA-AD V4 fixed architecture and oracle threshold grid.
    p.add_argument("--asca-win-size", type=int, default=100)
    p.add_argument("--asca-train-step", type=int, default=1)
    p.add_argument("--asca-eval-step", type=int, default=100)
    p.add_argument("--asca-epochs", type=int, default=10)
    p.add_argument("--asca-lr", type=float, default=1e-3)
    p.add_argument("--asca-area-weight", type=float, default=0.1)
    p.add_argument("--asca-balance-weight", type=float, default=0.05)
    p.add_argument("--asca-ratio-start", type=float, default=0.1)
    p.add_argument("--asca-ratio-stop", type=float, default=3.0)
    p.add_argument("--asca-ratio-step", type=float, default=0.1)

    # LTFAD official SKAB settings from best_parameter.txt.
    p.add_argument("--ltfad-win-size", type=int, default=90)
    p.add_argument("--ltfad-train-step", type=int, default=1)
    p.add_argument("--ltfad-eval-step", type=int, default=90)
    p.add_argument("--ltfad-epochs", type=int, default=2)
    p.add_argument("--ltfad-lr", type=float, default=1e-4)
    p.add_argument("--ltfad-d-model", type=int, default=128)
    p.add_argument("--ltfad-local-sizes", nargs="+", type=int, default=[5])
    p.add_argument("--ltfad-global-sizes", nargs="+", type=int, default=[13])
    p.add_argument("--ltfad-r", type=float, default=0.1)
    p.add_argument("--ltfad-anomaly-ratio", type=float, default=0.3)

    a = p.parse_args()
    seed_all(a.seed)
    dev = torch.device(a.device)

    tr, te, y, pt, pv, pl = load_arrays(a.dataset_dir, a.dataset)
    channels = tr.shape[1]
    print("=" * 80)
    print("SKAB comparison: ASCA-AD V4 vs LTFAD")
    print(f"train={tr.shape} test={te.shape} labels={y.shape}")
    print(f"files={pt.name},{pv.name},{pl.name}")
    print("=" * 80)

    common_loader_args = dict(
        batch_size=a.batch_size,
        num_workers=0,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )
    results = []

    for kind in a.models:
        seed_all(a.seed)

        if kind == "asca":
            win_size = a.asca_win_size
            train_step = a.asca_train_step
            eval_step = a.asca_eval_step
            epochs = a.asca_epochs

            mod = load_module("asca_model", ROOT / "asca_ad" / "model.py")
            base = mod.AdaptiveSparseAnchorCompetitiveModelV4(
                [1,2,3,4,5,6,7,8],
                [12,16,20,24,28,32,40,48],
                2, 4, 8, 8, 0.5, 1.0, 0.03, 1.5, 1.0,
            )
            model = ASCA(base).to(dev)
            optimizer = torch.optim.Adam(model.parameters(), lr=a.asca_lr)
            name = "ASCA-AD V4"
        else:
            win_size = a.ltfad_win_size
            train_step = a.ltfad_train_step
            eval_step = a.ltfad_eval_step
            epochs = a.ltfad_epochs

            mod = load_module(
                "ltfad_model",
                ROOT / "BaselineModels" / "LTFAD-main" / "model" / "LTFAD.py",
            )
            base = mod.LTFAD(
                win_size,
                a.ltfad_d_model,
                a.ltfad_local_sizes,
                a.ltfad_global_sizes,
                channels,
            )
            model = LTFAD(
                base,
                a.ltfad_local_sizes,
                a.ltfad_global_sizes,
                a.ltfad_r,
            ).to(dev)
            optimizer = torch.optim.Adam(model.parameters(), lr=a.ltfad_lr)
            name = "LTFAD"

        train_ds = Windows(tr, None, win_size, train_step)
        train_eval_ds = Windows(tr, None, win_size, eval_step)
        test_ds = Windows(te, y, win_size, eval_step)

        train_loader = DataLoader(train_ds, shuffle=True, **common_loader_args)
        train_eval_loader = DataLoader(train_eval_ds, shuffle=False, **common_loader_args)
        test_loader = DataLoader(test_ds, shuffle=False, **common_loader_args)

        print()
        print(f"{name}: {params(model):,} parameters")
        print(f"{name}: win={win_size}, epochs={epochs}, train_step={train_step}, eval_step={eval_step}")

        epoch_times = train(
            name,
            model,
            train_loader,
            optimizer,
            epochs,
            dev,
            a.asca_area_weight,
            a.asca_balance_weight,
        )

        if kind == "asca":
            best = None
            ratio = a.asca_ratio_start
            while ratio <= a.asca_ratio_stop + 1e-12:
                candidate = run_eval(
                    name,
                    model,
                    train_eval_loader,
                    test_loader,
                    len(tr),
                    len(te),
                    y,
                    win_size,
                    ratio,
                    dev,
                    epoch_times,
                )
                candidate["anomaly_ratio"] = float(round(ratio, 10))
                candidate["threshold_policy"] = "oracle-best PA-F1 grid on SKAB test labels"
                if best is None or candidate["pa"]["f1"] > best["pa"]["f1"]:
                    best = candidate
                ratio += a.asca_ratio_step
            result = best
        else:
            result = run_eval(
                name,
                model,
                train_eval_loader,
                test_loader,
                len(tr),
                len(te),
                y,
                win_size,
                a.ltfad_anomaly_ratio,
                dev,
                epoch_times,
            )
            result["anomaly_ratio"] = a.ltfad_anomaly_ratio
            result["threshold_policy"] = "official SKAB anomaly_ratio=0.3"

        results.append(result)
        print(
            f"{name}: ratio={result['anomaly_ratio']}, "
            f"RAW-F1={result['raw']['f1']:.4f}, "
            f"PA-F1={result['pa']['f1']:.4f}, "
            f"test={result['test_time']:.3f}s, "
            f"peak={result['peak_memory_mib']:.2f}MiB"
        )

        del model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    a.output_dir.mkdir(parents=True, exist_ok=True)
    (a.output_dir / "results.json").write_text(
        json.dumps({"config": vars(a), "results": results}, indent=2, default=str),
        encoding="utf-8",
    )

    with (a.output_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        cols = [
            "model", "parameters", "mean_epoch_time", "test_time",
            "peak_memory_mib", "anomaly_ratio", "raw_f1", "pa_f1",
        ]
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "model": r["model"],
                "parameters": r["parameters"],
                "mean_epoch_time": r["mean_epoch_time"],
                "test_time": r["test_time"],
                "peak_memory_mib": r["peak_memory_mib"],
                "anomaly_ratio": r["anomaly_ratio"],
                "raw_f1": r["raw"]["f1"],
                "pa_f1": r["pa"]["f1"],
            })

    lines = [
        "# SKAB comparison",
        "",
        "|Model|Params|Epoch(s)|Test(s)|Peak MiB|Ratio|RAW F1|PA-F1|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"|{r['model']}|{r['parameters']}|{r['mean_epoch_time']:.3f}|"
            f"{r['test_time']:.3f}|{r['peak_memory_mib']:.2f}|"
            f"{r['anomaly_ratio']}|{r['raw']['f1']:.4f}|{r['pa']['f1']:.4f}|"
        )
    (a.output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved to {a.output_dir}")

if __name__ == "__main__":
    main()
