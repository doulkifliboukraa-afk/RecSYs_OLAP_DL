"""
run_final.py — self-contained, MF trained once per seed (not per fold).
Writes intermediate results to disk after each seed so nothing is lost.
"""
import random, math, os, sys, csv, json, time
import numpy as np
from scipy.stats import wilcoxon
sys.path.insert(0, os.path.dirname(__file__))

# ── Schema ────────────────────────────────────────────────────────────────────
from build_usage_log import (
    DATASETS, ATTRS, PERSONAS as P15, COLD, find, attrs_of
)
from build_usage_log_30 import PERSONAS_30 as PERSONAS, NEW_PERSONAS

ANALYSTS = [a for a in PERSONAS.keys()] + list(COLD)
NON_COLD = [a for a in PERSONAS.keys()]
SEEDS    = [42, 7, 13, 99, 123]
KS       = [1, 3, 5]
OUT      = os.environ.get("EVAL_OUT", "eval_out_final")   # output directory (relative by default)
os.makedirs(OUT, exist_ok=True)

def dataset_of_attr(attr):
    for d in DATASETS:
        if attr.startswith(d + "."):
            return d
    return None

# ── Event generator ───────────────────────────────────────────────────────────
def gen_events(seed=42, consistency=0.85):
    rng = random.Random(seed)
    events = []
    sid = 0
    for PSET in [P15, NEW_PERSONAS]:
        for persona, spec in PSET.items():
            for rnd in range(1, 5):
                sid += 1
                session = f"S{sid:04d}"
                fact = spec["fact"]
                if spec["alt_facts"] and rng.random() < (1-consistency):
                    fact = rng.choice(spec["alt_facts"])
                events.append((session, persona, rnd, "Dataset", fact, "F"))
                mp = spec["measures"] if fact==spec["fact"] else (attrs_of(fact) or spec["measures"])
                dp = list(spec["dims"])
                if spec["dim_alt"] and rng.random() < 0.25:
                    dp = dp + [rng.choice(spec["dim_alt"])]
                nd = min(len(dp), rng.randint(1, max(1, len(dp))))
                dims = rng.sample(dp, nd)
                for d in dims:
                    events.append((session, persona, rnd, "Dataset", d, "D"))
                nm = min(len(mp), rng.randint(1, max(1, len(mp))))
                for m in rng.sample(mp, nm):
                    events.append((session, persona, rnd, "Attribute", m, "M"))
                for d in dims:
                    pl = spec["params"].get(d, attrs_of(d)[:1])
                    np_ = min(len(pl), rng.randint(1, max(1, len(pl))))
                    for p in rng.sample(pl, np_):
                        events.append((session, persona, rnd, "Attribute", p, "P"))
                if rng.random() < 0.10:
                    pool = attrs_of(fact)
                    if pool:
                        events.append((session, persona, rnd, "Attribute", rng.choice(pool), "M"))
    return events

# ── Counts + similarity ───────────────────────────────────────────────────────
def build_counts(evlist):
    ds = {a: {d: {'F':0,'D':0} for d in DATASETS} for a in NON_COLD}
    at = {a: {x: {'M':0,'P':0} for x in ATTRS}    for a in NON_COLD}
    for (sid, persona, rnd, itype, name, role) in evlist:
        if persona in COLD: continue
        if itype=="Dataset" and name in DATASETS:
            if persona in ds: ds[persona][name][role] += 1
        elif itype=="Attribute" and name in ATTRS:
            if persona in at: at[persona][name][role] += 1
    return ds, at

def cosine_sim(items, vec_fn):
    vecs  = {it: vec_fn(it) for it in items}
    norms = {it: np.linalg.norm(v) for it,v in vecs.items()}
    sim   = {}
    for i in items:
        for j in items:
            ni,nj = norms[i],norms[j]
            sim[(i,j)] = float(np.dot(vecs[i],vecs[j])/(ni*nj)) if ni>0 and nj>0 else 0.0
    return sim

def build_matrices(evlist):
    ds_c, at_c = build_counts(evlist)
    def dsv(d): return np.array([x for a in NON_COLD for x in [ds_c[a][d]['F'],ds_c[a][d]['D']]], dtype=float)
    def atv(x): return np.array([v for a in NON_COLD for v in [at_c[a][x]['M'],at_c[a][x]['P']]], dtype=float)
    return ds_c, at_c, cosine_sim(DATASETS,dsv), cosine_sim(ATTRS,atv)

# ── Fast MF (train once per seed on full data) ────────────────────────────────
def train_mf(matrix, n_factors=8, n_steps=25, lr=0.02, reg=0.1, seed=0):
    rng = np.random.default_rng(seed)
    P = rng.normal(0,0.1,(matrix.shape[0],n_factors))
    Q = rng.normal(0,0.1,(matrix.shape[1],n_factors))
    mask = matrix > 0
    for _ in range(n_steps):
        err = (matrix - P@Q.T)*mask
        P  += lr*(err@Q - reg*P)
        Q  += lr*(err.T@P - reg*Q)
    return P, Q

def build_mf_scorers(ds_c, at_c, seed=0):
    a_idx = {a:i for i,a in enumerate(NON_COLD)}
    d_idx = {d:i for i,d in enumerate(DATASETS)}
    x_idx = {x:i for i,x in enumerate(ATTRS)}
    R_ds = np.array([[ds_c[a][d]['F']+ds_c[a][d]['D'] for d in DATASETS] for a in NON_COLD], dtype=float)
    R_at = np.array([[at_c[a][x]['M']+at_c[a][x]['P'] for x in ATTRS]    for a in NON_COLD], dtype=float)
    P_ds,Q_ds = train_mf(R_ds, seed=seed)
    P_at,Q_at = train_mf(R_at, seed=seed+1)
    def sc_ds(analyst, cands):
        if analyst not in a_idx: return {c:0.0 for c in cands}
        ai=a_idx[analyst]; return {c: float(P_ds[ai]@Q_ds[d_idx[c]]) for c in cands if c in d_idx}
    def sc_at(analyst, cands):
        if analyst not in a_idx: return {c:0.0 for c in cands}
        ai=a_idx[analyst]; return {c: float(P_at[ai]@Q_at[x_idx[c]]) for c in cands if c in x_idx}
    return sc_ds, sc_at

# ── Ranking ───────────────────────────────────────────────────────────────────
def ndcg(ranked, truth, k):
    try: r=ranked.index(truth)+1
    except ValueError: return 0.0
    return 1.0/math.log2(r+1) if r<=k else 0.0

def rank_all(candidates, persona, role, own_fn, others_fn, sim, all_items,
             ds_c, at_c, mf_ds, mf_at, rng, strat, k_nb=5):
    c = list(candidates)
    if strat=="random":
        rng.shuffle(c); return c
    if strat=="popularity":
        if role in ('F','D'):
            sc={x:sum(ds_c[a][x][role] for a in NON_COLD) for x in c}
        else:
            sc={x:sum(at_c[a][x][role] for a in NON_COLD) for x in c}
        return sorted(c, key=lambda x:(-sc[x], rng.random()))
    if strat=="own_only":
        return sorted(c, key=lambda x:(-own_fn(x), rng.random()))
    if strat=="knn":
        sc={}
        for x in c:
            nb=sorted([y for y in all_items if y!=x], key=lambda y:-sim.get((x,y),0))[:k_nb]
            num=sum(sim.get((x,j),0)*own_fn(j) for j in nb)
            den=sum(abs(sim.get((x,j),0)) for j in nb)
            sc[x]=num/den if den>0 else 0.0
        return sorted(c, key=lambda x:(-sc[x], rng.random()))
    if strat=="mf":
        mf_fn = mf_ds if role in ('F','D') else mf_at
        sc = mf_fn(persona, c)
        return sorted(c, key=lambda x:(-sc.get(x,0.0), rng.random()))
    # similarity level: maximum cosine similarity to the items of this role that
    # have direct usage evidence (own or others' usage), as in Section 6.
    def sim_known(x):
        best=0.0
        for y in all_items:
            if y==x: continue
            if own_fn(y)+others_fn(y)>0:
                v=sim.get((x,y),0.0)
                if v>best: best=v
        return best
    if strat=="sim_only":
        sc={x:sim_known(x) for x in c}
        return sorted(c, key=lambda x:(-sc[x], rng.random()))
    # hybrid (paper order): own usage > others' usage > similarity > random
    sc={}
    for x in c:
        sc[x]=(own_fn(x), others_fn(x), sim_known(x), rng.random())
    return sorted(c, key=lambda x:(-sc[x][0], -sc[x][1], -sc[x][2], sc[x][3]))

# ── Fold executor ─────────────────────────────────────────────────────────────
FULL_STRATS = ["random","popularity","knn","mf","own_only","hybrid","collaborative","sim_only"]
COLD_STRATS = ["metadata_fallback","knn","mf","collaborative","hybrid","sim_only"]

def score_fold(persona, role, name, fact, candidates, ds_c, at_c, sim_ds, sim_at,
               mf_ds, mf_at, rng_map, strat_list, cs=False):
    if role in ('F','D'):
        sim=sim_ds; all_items=list(DATASETS)
    else:
        sim=sim_at; all_items=candidates
    def own_fn(c):
        if cs: return 0
        return (ds_c[persona][c]['F'] if role=='F' else
                ds_c[persona][c]['D'] if role=='D' else
                at_c[persona][c]['M'] if role=='M' else
                at_c[persona][c]['P']) if persona in ds_c else 0
    def oth_fn(c):
        if role in ('F','D'):
            return sum(ds_c[a][c][role] for a in NON_COLD if a!=persona)
        return sum(at_c[a][c][role] for a in NON_COLD if a!=persona)
    results=[]
    for strat in strat_list:
        rng=rng_map[strat]
        if strat=="metadata_fallback":
            ranked=list(candidates); rng.shuffle(ranked)
        elif strat=="collaborative":
            ranked=sorted(candidates, key=lambda x:(-oth_fn(x), rng.random()))
        else:
            ranked=rank_all(candidates, persona, role, own_fn, oth_fn,
                            sim, all_items, ds_c, at_c, mf_ds, mf_at, rng, strat)
        rank=ranked.index(name)+1 if name in ranked else None
        row=dict(persona=persona, role=role, strategy=strat)
        for k in KS: row[f"ndcg@{k}"]=ndcg(ranked,name,k)
        results.append(row)
    return results

# ── Protocol A: LOO ───────────────────────────────────────────────────────────
def run_loo(events, seed, ds_c_full, at_c_full, mf_ds, mf_at):
    rng_map={s:random.Random(seed+100+i) for i,s in enumerate(FULL_STRATS)}
    results=[]
    for idx in range(len(events)):
        sid,persona,rnd,itype,name,role=events[idx]
        if persona in COLD: continue
        masked=events[:idx]+events[idx+1:]
        ctx=[e for e in masked if e[0]==sid]
        # Use cached counts but rebuild sim from masked (fast since no MF)
        ds_c,at_c=build_counts(masked)
        def dsv(d): return np.array([x for a in NON_COLD for x in [ds_c[a][d]['F'],ds_c[a][d]['D']]], dtype=float)
        def atv(x): return np.array([v for a in NON_COLD for v in [at_c[a][x]['M'],at_c[a][x]['P']]], dtype=float)
        sim_ds=cosine_sim(DATASETS,dsv); sim_at=cosine_sim(ATTRS,atv)
        fact=next((e[4] for e in ctx if e[3]=="Dataset" and e[5]=="F"),None)
        if role=="F":
            cands=list(DATASETS)
        elif role=="D":
            odims={e[4] for e in ctx if e[3]=="Dataset" and e[5]=="D"}
            cands=[d for d in DATASETS if d!=fact and d not in odims]
        elif role=="M":
            if not fact: continue
            omeas={e[4] for e in ctx if e[3]=="Attribute" and e[5]=="M"}
            cands=[a for a in ATTRS if dataset_of_attr(a)==fact and a not in omeas]
        elif role=="P":
            tds=dataset_of_attr(name)
            if not tds: continue
            opar={e[4] for e in ctx if e[3]=="Attribute" and e[5]=="P" and dataset_of_attr(e[4])==tds}
            cands=[a for a in ATTRS if dataset_of_attr(a)==tds and a not in opar]
        else: continue
        if not cands or name not in cands: continue
        results.extend(score_fold(persona,role,name,fact,cands,
                                   ds_c,at_c,sim_ds,sim_at,mf_ds,mf_at,
                                   rng_map,FULL_STRATS))
    return results

# ── Protocol B: Chronological ─────────────────────────────────────────────────
def run_chron(events, seed, mf_ds, mf_at):
    train=[e for e in events if e[2]<=3 and e[1] not in COLD]
    test =[e for e in events if e[2]==4  and e[1] not in COLD]
    ds_c,at_c,sim_ds,sim_at=build_matrices(train)
    rng_map={s:random.Random(seed+200+i) for i,s in enumerate(FULL_STRATS)}
    results=[]
    sess={}
    for e in test: sess.setdefault(e[0],[]).append(e)
    for sid,sevts in sess.items():
        persona=sevts[0][1]
        fact=next((e[4] for e in sevts if e[3]=="Dataset" and e[5]=="F"),None)
        for e_idx,(esid,ep,ernd,etype,name,role) in enumerate(sevts):
            if role=="F": cands=list(DATASETS)
            elif role=="D": cands=[d for d in DATASETS if d!=fact]
            elif role=="M":
                if not fact: continue
                cands=[a for a in ATTRS if dataset_of_attr(a)==fact]
            elif role=="P":
                tds=dataset_of_attr(name)
                if not tds: continue
                cands=[a for a in ATTRS if dataset_of_attr(a)==tds]
            else: continue
            if not cands or name not in cands: continue
            results.extend(score_fold(persona,role,name,fact,cands,
                                       ds_c,at_c,sim_ds,sim_at,mf_ds,mf_at,
                                       rng_map,FULL_STRATS))
    return results

# ── Protocol C: Cold-start ────────────────────────────────────────────────────
def run_cold(events, seed):
    rng_map={s:random.Random(seed+300+i) for i,s in enumerate(COLD_STRATS)}
    results=[]
    for persona in list(PERSONAS.keys()):
        cold_ev=[e for e in events if e[1]!=persona and e[1] not in COLD]
        ds_c,at_c,sim_ds,sim_at=build_matrices(cold_ev)
        mf_ds,mf_at=build_mf_scorers(ds_c,at_c,seed=seed)
        test=[e for e in events if e[1]==persona and e[2]==1]
        fact=next((e[4] for e in test if e[3]=="Dataset" and e[5]=="F"),None)
        for (esid,ep,ernd,etype,name,role) in test:
            if role=="F": cands=list(DATASETS)
            elif role=="D": cands=[d for d in DATASETS if d!=fact]
            elif role=="M":
                if not fact: continue
                cands=[a for a in ATTRS if dataset_of_attr(a)==fact]
            elif role=="P":
                tds=dataset_of_attr(name)
                if not tds: continue
                cands=[a for a in ATTRS if dataset_of_attr(a)==tds]
            else: continue
            if not cands or name not in cands: continue
            results.extend(score_fold(persona,role,name,fact,cands,
                                       ds_c,at_c,sim_ds,sim_at,mf_ds,mf_at,
                                       rng_map,COLD_STRATS,cs=True))
    return results

# ── Aggregation helpers ────────────────────────────────────────────────────────
def pm(results,strat,k):
    out=[]
    for p in list(PERSONAS.keys()):
        vals=[r[f"ndcg@{k}"] for r in results if r["strategy"]==strat and r["persona"]==p]
        out.append(np.mean(vals) if vals else 0.0)
    return np.array(out)

def bci(arr,n=2000):
    rng=np.random.default_rng(0)
    b=[np.mean(rng.choice(arr,len(arr),replace=True)) for _ in range(n)]
    return tuple(np.percentile(b,[2.5,97.5]))

def safe_wlcx(a,b):
    try: s,p=wilcoxon(a,b,alternative='two-sided',zero_method='wilcox'); return s,p
    except: return float('nan'),float('nan')

def write_csv(rows,path):
    if not rows: return
    with open(path,"w",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows: w.writerow(r)

# ── Print summary ─────────────────────────────────────────────────────────────
def print_summary(loo_all,chron_all,cs_all,label):
    print(f"\n{'='*68}")
    print(f"CONDITION: {label}")
    print(f"{'='*68}")
    print(f"\nA. Leave-one-out ({len(SEEDS)} seeds × 30 personas):")
    hyb_pm=pm(loo_all,"hybrid",1)
    for s in FULL_STRATS:
        n1=np.array([r["ndcg@1"] for r in loo_all if r["strategy"]==s])
        n5=np.array([r["ndcg@5"] for r in loo_all if r["strategy"]==s])
        if not len(n1): continue
        ci=bci(n1)
        xpm=pm(loo_all,s,1)
        _,p=safe_wlcx(hyb_pm,xpm)
        sig="*" if p<0.05 else " "
        print(f"  {sig}{s:<12} NDCG@1={np.mean(n1):.4f}±{np.std(n1):.4f} "
              f"[{ci[0]:.4f},{ci[1]:.4f}]  NDCG@5={np.mean(n5):.4f}  "
              f"gap vs hybrid={np.mean(hyb_pm)-np.mean(xpm):+.5f}  p={p:.4f}")
    print(f"\nB. Chronological split:")
    for s in FULL_STRATS:
        n1=np.array([r["ndcg@1"] for r in chron_all if r["strategy"]==s])
        n5=np.array([r["ndcg@5"] for r in chron_all if r["strategy"]==s])
        if not len(n1): continue
        print(f"  {s:<12} NDCG@1={np.mean(n1):.4f}±{np.std(n1):.4f}  NDCG@5={np.mean(n5):.4f}")
    print(f"\nC. Cold-start:")
    base=np.mean([r["ndcg@1"] for r in cs_all if r["strategy"]=="metadata_fallback"])
    for s in COLD_STRATS:
        n1=np.array([r["ndcg@1"] for r in cs_all if r["strategy"]==s])
        if not len(n1): continue
        print(f"  {s:<22} NDCG@1={np.mean(n1):.4f}±{np.std(n1):.4f}  gain={np.mean(n1)-base:+.4f}")
    print(f"\nD. By-role (LOO, NDCG@1):")
    print(f"  {'Role':<4} {'own':>8} {'hybrid':>8} {'knn':>8} {'mf':>8} {'pop':>8}")
    for role in ["F","D","M","P"]:
        def rm(s): return np.mean([r["ndcg@1"] for r in loo_all if r["strategy"]==s and r["role"]==role])
        print(f"  {role:<4} {rm('own_only'):>8.4f} {rm('hybrid'):>8.4f} {rm('knn'):>8.4f} {rm('mf'):>8.4f} {rm('popularity'):>8.4f}")

# ── Main ──────────────────────────────────────────────────────────────────────
print("FINAL EVALUATION — 30 personas, 5 strategies, 3 protocols, 2 conditions")
print(f"Seeds: {SEEDS}  |  Start: {time.strftime('%H:%M:%S')}\n")

all_data={}
for consistency,label,tag in [(0.85,"High self-consistency (A)","high"),
                               (0.40,"Low self-consistency (B)", "low")]:
    loo_all=[]; chron_all=[]; cs_all=[]
    for seed in SEEDS:
        t0=time.time()
        events=gen_events(seed=seed,consistency=consistency)
        ds_c,at_c,sim_ds,sim_at=build_matrices(events)
        mf_ds,mf_at=build_mf_scorers(ds_c,at_c,seed=seed)
        loo_all.extend(run_loo(events,seed,ds_c,at_c,mf_ds,mf_at))
        chron_all.extend(run_chron(events,seed,mf_ds,mf_at))
        cs_all.extend(run_cold(events,seed))
        print(f"  [{label}] seed={seed} done in {time.time()-t0:.1f}s")
    print_summary(loo_all,chron_all,cs_all,label)
    write_csv(loo_all,   f"{OUT}/loo_{tag}.csv")
    write_csv(chron_all, f"{OUT}/chron_{tag}.csv")
    write_csv(cs_all,    f"{OUT}/cs_{tag}.csv")
    all_data[tag]=dict(loo=loo_all,chron=chron_all,cs=cs_all)

print(f"\nAll CSVs written to {OUT}/")
print(f"Done: {time.strftime('%H:%M:%S')}")
