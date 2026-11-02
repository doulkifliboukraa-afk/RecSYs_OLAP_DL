import csv, json, numpy as np
src=open('run_final.py').read().split('# ── Main')[0]
exec(compile(src,'run_final_head','exec'))
from scipy.stats import wilcoxon
D=OUT
def load(name):
    rows=[]
    for r in csv.DictReader(open(f'{D}/{name}.csv')):
        for k in KS: r[f'ndcg@{k}']=float(r[f'ndcg@{k}'])
        rows.append(r)
    return rows
def vals(rows,s,k,role=None):
    return np.array([r[f'ndcg@{k}'] for r in rows if r['strategy']==s and (role is None or r['role']==role)])
def pmeans(rows,s,k):
    out={}
    for p in NON_COLD:
        v=[r[f'ndcg@{k}'] for r in rows if r['strategy']==s and r['persona']==p]
        if v: out[p]=np.mean(v)
    return out
def wil(rows,a,b,k=1):
    A=pmeans(rows,a,k); B=pmeans(rows,b,k)
    ps=[p for p in A if p in B]
    x=np.array([A[p] for p in ps]); y=np.array([B[p] for p in ps])
    try: p=float(wilcoxon(x,y,alternative='two-sided',zero_method='wilcox').pvalue)
    except Exception: p=float('nan')
    return dict(p=p,n=len(ps),wins=int((x>y).sum()),losses=int((x<y).sum()),ties=int((x==y).sum()),gap=float(x.mean()-y.mean()))
def stat(rows,s,k=1):
    v=vals(rows,s,k); ci=bci(v)
    return dict(m=float(v.mean()),sd=float(v.std()),lo=float(ci[0]),hi=float(ci[1]),n5=float(vals(rows,s,5).mean()),n3=float(vals(rows,s,3).mean()),
                p1=float(np.mean(v>0)),p3=float(np.mean(vals(rows,s,3)>0)/3),p5=float(np.mean(vals(rows,s,5)>0)/5))
res={}
S=["random","popularity","knn","mf","own_only","hybrid","collaborative","sim_only"]
for tag,c in [('high','A'),('low','B')]:
    L=load(f'loo_{tag}'); C=load(f'chron_{tag}'); K=load(f'cs_{tag}')
    r={}
    r['loo']={s:stat(L,s) for s in S}
    r['chron']={s:stat(C,s) for s in S}
    r['cold']={s:stat(K,s) for s in ["metadata_fallback","knn","mf","collaborative","hybrid","sim_only"]}
    base=r['cold']['metadata_fallback']['m']
    for s in r['cold']: r['cold'][s]['gain']=r['cold'][s]['m']-base
    r['role']={ro:{s:float(vals(L,s,1,ro).mean()) for s in S} for ro in "FDMP"}
    r['wil_loo']={f'{b}@{k}':wil(L,'hybrid',b,k) for b in S if b!='hybrid' for k in KS}
    r['wil_chron']={f'{b}@{k}':wil(C,'hybrid',b,k) for b in ['own_only','mf'] for k in KS}
    r['wil_cold']={b:wil(K,'hybrid',b,1) for b in ['metadata_fallback','collaborative','mf','knn','sim_only']}
    res[c]=r
# MF rank sweep (condition A), hybrid persona means from CSV
LA=load('loo_high'); res['mfrank']={}
for nf in [2,3,5,8]:
    train_mf.__defaults__=(nf,25,0.02,0.1,0)
    rows=[]
    for seed in SEEDS:
        ev=gen_events(seed=seed,consistency=0.85)
        ds_c,at_c,sd,sa=build_matrices(ev)
        mfd,mfa=build_mf_scorers(ds_c,at_c,seed=seed)
        rows.extend(run_loo(ev,seed,ds_c,at_c,mfd,mfa))
    A=pmeans(LA,'hybrid',1); B=pmeans(rows,'mf',1)
    ps=[p for p in A if p in B]
    x=np.array([A[p] for p in ps]); y=np.array([B[p] for p in ps])
    res['mfrank'][nf]=dict(p1=float(vals(rows,'mf',1).mean()),p=float(wilcoxon(x,y).pvalue),wins=int((x>y).sum()))
json.dump(res,open('analysis2.json','w'),indent=1)
print('ok')
