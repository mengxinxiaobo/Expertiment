
import argparse, json
from pathlib import Path
import numpy as np

def point_adjust(pred, gt):
    pred = pred.copy()
    state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not state:
            state = True
            for j in range(i, -1, -1):
                if gt[j] == 0:
                    break
                pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                pred[j] = 1
        elif gt[i] == 0:
            state = False
        if state:
            pred[i] = 1
    return pred

def metrics(gt, pred):
    tp=((gt==1)&(pred==1)).sum()
    fp=((gt==0)&(pred==1)).sum()
    fn=((gt==1)&(pred==0)).sum()
    tn=((gt==0)&(pred==0)).sum()
    p=tp/max(tp+fp,1)
    r=tp/max(tp+fn,1)
    f=2*p*r/max(p+r,1e-12)
    return {"accuracy":float((tp+tn)/len(gt)),
            "precision":float(p),
            "recall":float(r),
            "f1":float(f)}

def evaluate(train_energy,test_energy,labels,ratio):
    energy=np.concatenate([train_energy,test_energy])
    threshold=np.percentile(energy,100-ratio)
    pred=(test_energy>threshold).astype(int)
    return {
        "threshold":float(threshold),
        "raw":metrics(labels,pred),
        "pa":metrics(labels,point_adjust(pred,labels))
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--score-dir",type=Path,required=True)
    ap.add_argument("--label",type=Path,required=True)
    ap.add_argument("--output",type=Path,default="results.json")
    ap.add_argument("--anomaly-ratio",type=float,default=0.3)
    args=ap.parse_args()

    labels=np.load(args.label).reshape(-1)
    result={}

    for name in ["PPLAD","ASCA-AD","LTFAD"]:
        tr=np.load(args.score_dir/f"{name}_train_energy.npy")
        te=np.load(args.score_dir/f"{name}_test_energy.npy")
        result[name]=evaluate(tr,te,labels,args.anomaly_ratio)
        print("="*60)
        print(name)
        print(json.dumps(result[name],indent=2))

    args.output.write_text(json.dumps(result,indent=2))

if __name__=="__main__":
    main()
