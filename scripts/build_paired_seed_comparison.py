"""Rebuild the primary cross-family final-score comparison on seeds {42,2,3}.

This script uses only the canonical numerical results in docs/results/latest_results.json.
No raw training logs or model weights are required.
"""
from pathlib import Path
import csv, json
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
COMMON_SEEDS = [42, 2, 3]

def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def stats(ps):
    vals=[float(ps[str(s)]) for s in COMMON_SEEDS]
    return mean(vals), stdev(vals)

def add_row(rows,family,variant,ps,baseline):
    m,s=stats(ps)
    deltas={str(seed):float(ps[str(seed)])-float(baseline[str(seed)]) for seed in COMMON_SEEDS}
    dm,ds=stats(deltas) if any(abs(v)>0 for v in deltas.values()) else (0.0,0.0)
    rows.append({"family":family,"variant":variant,"seed_42":float(ps["42"]),"seed_2":float(ps["2"]),"seed_3":float(ps["3"]),"mean":m,"std_across_training_seeds":s,"mean_paired_delta_vs_baseline":dm,"std_paired_delta_vs_baseline":ds,"paired_delta_per_seed":deltas})

def main():
    d=load_json(ROOT/"docs/results/latest_results.json")
    baseline={str(k):float(v) for k,v in d["baseline"]["per_seed"].items()}
    ddqn={str(k):float(v) for k,v in d["ddqn"]["per_seed"].items()}
    rows=[]
    add_row(rows,"baseline","Vanilla DQN",baseline,baseline)
    add_row(rows,"ddqn","Double DQN",ddqn,baseline)
    names={"epsilon":"Epsilon","rally":"Rally","potential":"PBRS","contact":"Contact"}
    for fam in ["epsilon","rally","potential","contact"]:
        for key,v in d[fam]["variants"].items():
            add_row(rows,names[fam],v.get("label",key),{str(k):float(x) for k,x in v["per_seed"].items()},baseline)
    out={"provenance":"Primary comparison on common training seeds {42,2,3}; canonical checkpoint numbers only.","common_training_seeds":COMMON_SEEDS,"rows":rows}
    od=ROOT/"docs/results"
    (od/"paired_seed_results.json").write_text(json.dumps(out,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    fields=["family","variant","seed_42","seed_2","seed_3","mean","std_across_training_seeds","mean_paired_delta_vs_baseline","std_paired_delta_vs_baseline"]
    with (od/"paired_seed_final_scores.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); [w.writerow({k:r[k] for k in fields}) for r in rows]
    print(od/"paired_seed_results.json")
    print(od/"paired_seed_final_scores.csv")

if __name__=="__main__": main()
