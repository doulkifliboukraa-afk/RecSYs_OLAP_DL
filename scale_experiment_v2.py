"""
scale_experiment_v2.py -- EXPLORATORY scale-up check (Supplementary Material S1).
Question: does adding more analysts let cross-analyst (collaborative) ranking overtake own-history?

Updated to the current evaluation pipeline (run_final.py):
  * 30 personas (build_usage_log + build_usage_log_30), same event generator,
  * five seeds, two consistency conditions (A = 0.85, B = 0.40),
  * strategies: random, popularity (all analysts' counts incl. the target analyst's own),
    own_only, others_only (all OTHER analysts' counts), sim_only, hybrid
    (own > others > similarity > random, the order of Section 6).
Leave-one-out is exact for own/others/popularity counts (removing a held-out event only changes the
target analyst's own count) and approximated for the similarity matrix (computed once on the full log).

Experiment 1 (--exp rep):  replicate each persona into N independent analysts.
Experiment 2 (--exp pool): overlap-engineered shared small pool (reconstructed from the description in S1).
Usage: python3 scale_experiment_v2.py --exp rep|pool --cons 0.85 [--seeds 42,7,13,99,123]
"""
import random, math, csv, sys, argparse, json
import numpy as np
from build_usage_log import DATASETS, ATTRS, PERSONAS as P15, attrs_of
from build_usage_log_30 import NEW_PERSONAS

KS = [1, 3, 5]
STRATS = ["random", "popularity", "own_only", "others_only", "sim_only", "hybrid"]

def dataset_of_attr(a):
    for d in DATASETS:
        if a.startswith(d + "."): return d
    return None

# ---------- generators ----------
def gen_persona_events(spec, rng, analyst, sid0, consistency):
    ev = []; sid = sid0
    for rnd in range(1, 5):
        sid += 1; session = f"S{sid:06d}"
        fact = spec["fact"]
        if spec["alt_facts"] and rng.random() < (1 - consistency): fact = rng.choice(spec["alt_facts"])
        ev.append((session, analyst, rnd, "Dataset", fact, "F"))
        mp = spec["measures"] if fact == spec["fact"] else (attrs_of(fact) or spec["measures"])
        dp = list(spec["dims"])
        if spec["dim_alt"] and rng.random() < 0.25: dp = dp + [rng.choice(spec["dim_alt"])]
        dims = rng.sample(dp, min(len(dp), rng.randint(1, max(1, len(dp)))))
        for d in dims: ev.append((session, analyst, rnd, "Dataset", d, "D"))
        for m in rng.sample(mp, min(len(mp), rng.randint(1, max(1, len(mp))))):
            ev.append((session, analyst, rnd, "Attribute", m, "M"))
        for d in dims:
            pl = spec["params"].get(d, attrs_of(d)[:1])
            for p in rng.sample(pl, min(len(pl), rng.randint(1, max(1, len(pl))))):
                ev.append((session, analyst, rnd, "Attribute", p, "P"))
        if rng.random() < 0.10:
            pool = attrs_of(fact)
            if pool: ev.append((session, analyst, rnd, "Attribute", rng.choice(pool), "M"))
    return ev, sid

def gen_rep(seed, n_rep, consistency):
    PERS = {**P15, **NEW_PERSONAS}
    events, analysts, sid = [], [], 0
    for persona, spec in PERS.items():
        for rep in range(n_rep):
            a = f"{persona}#{rep}" if n_rep > 1 else persona
            analysts.append(a)
            rng = random.Random(f"{seed}-{persona}-{rep}")
            ev, sid = gen_persona_events(spec, rng, a, sid, consistency)
            events += ev
    return events, analysts

def gen_pool(seed, n_per_family, n_fam=4, noise=0.10):
    """Four families, each with a fixed fact and a small shared pool (3 dimensions, 2 measures);
    every analyst keeps a fixed personal subset of the pool, with light session-to-session noise."""
    rng0 = random.Random(f"pool-{seed}")
    facts = ["SE_mysql.postes", "SE_mongodb.Comments", "SE_neo4j.Posts", "SE_mysql.Votes"][:n_fam]
    fams = []
    for i, f in enumerate(facts):
        others = [d for d in DATASETS if d != f]
        dims = [others[(i * 3 + j) % len(others)] for j in range(3)]
        meas = attrs_of(f)[:2]
        fams.append((f, dims, meas))
    events, analysts, sid = [], [], 0
    for fi, (f, dpool, mpool) in enumerate(fams):
        for k in range(n_per_family):
            a = f"F{fi}#{k}"; analysts.append(a)
            rng = random.Random(f"{seed}-pool-{fi}-{k}")
            my_d = rng.sample(dpool, rng.randint(1, len(dpool)))
            my_m = rng.sample(mpool, rng.randint(1, len(mpool)))
            for rnd in range(1, 5):
                sid += 1; s = f"S{sid:06d}"
                events.append((s, a, rnd, "Dataset", f, "F"))
                dims = list(my_d)
                if rng.random() < noise:
                    extra = [d for d in dpool if d not in dims]
                    if extra: dims.append(rng.choice(extra))
                for d in dims: events.append((s, a, rnd, "Dataset", d, "D"))
                ms = list(my_m)
                if rng.random() < noise:
                    extra = [m for m in mpool if m not in ms]
                    if extra: ms.append(rng.choice(extra))
                for m in ms: events.append((s, a, rnd, "Attribute", m, "M"))
    return events, analysts

# ---------- evaluation ----------
def evaluate(events, analysts, seed):
    didx = {d: i for i, d in enumerate(DATASETS)}; aidx = {x: i for i, x in enumerate(ATTRS)}
    own_ds = {a: {} for a in analysts}; own_at = {a: {} for a in analysts}
    tot = {("F", d): 0 for d in DATASETS}; tot.update({("D", d): 0 for d in DATASETS})
    tot.update({("M", x): 0 for x in ATTRS}); tot.update({("P", x): 0 for x in ATTRS})
    for (s, a, r, t, n, role) in events:
        (own_ds if t == "Dataset" else own_at)[a][(role, n)] = (own_ds if t == "Dataset" else own_at)[a].get((role, n), 0) + 1
        tot[(role, n)] += 1
    na = len(analysts); aid = {a: i for i, a in enumerate(analysts)}
    Vd = np.zeros((len(DATASETS), 2 * na)); Va = np.zeros((len(ATTRS), 2 * na))
    for (s, a, r, t, n, role) in events:
        if t == "Dataset": Vd[didx[n], 2 * aid[a] + (0 if role == "F" else 1)] += 1
        else: Va[aidx[n], 2 * aid[a] + (0 if role == "M" else 1)] += 1
    def cos(V, items, idx):
        nr = np.linalg.norm(V, axis=1); S = V @ V.T
        return {(i, j): (float(S[idx[i], idx[j]] / (nr[idx[i]] * nr[idx[j]])) if nr[idx[i]] > 0 and nr[idx[j]] > 0 else 0.0)
                for i in items for j in items}
    sim_ds = cos(Vd, DATASETS, didx); sim_at = cos(Va, ATTRS, aidx)
    by_sess = {}
    for i, e in enumerate(events): by_sess.setdefault(e[0], []).append(i)
    rngs = {s: random.Random(seed + 1000 + i) for i, s in enumerate(STRATS)}
    hits = {s: [] for s in STRATS}
    for idx, (sid, persona, rnd, itype, name, role) in enumerate(events):
        ctx = [events[j] for j in by_sess[sid] if j != idx]
        fact = next((e[4] for e in ctx if e[3] == "Dataset" and e[5] == "F"), None)
        if role == "F": cands = list(DATASETS)
        elif role == "D":
            od = {e[4] for e in ctx if e[3] == "Dataset" and e[5] == "D"}
            cands = [d for d in DATASETS if d != fact and d not in od]
        elif role == "M":
            if not fact: continue
            om = {e[4] for e in ctx if e[3] == "Attribute" and e[5] == "M"}
            cands = [x for x in ATTRS if dataset_of_attr(x) == fact and x not in om]
        else:
            tds = dataset_of_attr(name)
            if not tds: continue
            op = {e[4] for e in ctx if e[3] == "Attribute" and e[5] == "P" and dataset_of_attr(e[4]) == tds}
            cands = [x for x in ATTRS if dataset_of_attr(x) == tds and x not in op]
        if name not in cands: continue
        own_store = own_ds[persona] if role in "FD" else own_at[persona]
        def own(c): return own_store.get((role, c), 0) - (1 if c == name else 0)
        def oth(c): return tot[(role, c)] - own_store.get((role, c), 0)
        def pop(c): return tot[(role, c)] - (1 if c == name else 0)
        sim = sim_ds if role in "FD" else sim_at
        all_items = list(DATASETS) if role in "FD" else cands
        def sim_known(x):
            best = 0.0
            for y in all_items:
                if y != x and own(y) + oth(y) > 0:
                    v = sim.get((x, y), 0.0)
                    if v > best: best = v
            return best
        for st in STRATS:
            rng = rngs[st]
            if st == "random": r = cands[:]; rng.shuffle(r)
            elif st == "popularity": r = sorted(cands, key=lambda x: (-pop(x), rng.random()))
            elif st == "own_only": r = sorted(cands, key=lambda x: (-own(x), rng.random()))
            elif st == "others_only": r = sorted(cands, key=lambda x: (-oth(x), rng.random()))
            elif st == "sim_only": r = sorted(cands, key=lambda x: (-sim_known(x), rng.random()))
            else:
                sc = {x: (own(x), oth(x), sim_known(x), rng.random()) for x in cands}
                r = sorted(cands, key=lambda x: (-sc[x][0], -sc[x][1], -sc[x][2], sc[x][3]))
            hits[st].append(1.0 if r[0] == name else 0.0)
    return {s: float(np.mean(v)) for s, v in hits.items()}, len(hits["own_only"])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="rep"); ap.add_argument("--cons", type=float, default=0.85)
    ap.add_argument("--seeds", default="42,7,13,99,123"); ap.add_argument("--out", default="scale_v2")
    args = ap.parse_args(); seeds = [int(x) for x in args.seeds.split(",")]
    plan = [1, 3, 10, 30] if args.exp == "rep" else [5, 50, 150]
    res = []
    for n in plan:
        per = {s: [] for s in STRATS}; nfold = []
        for sd in seeds:
            ev, an = gen_rep(sd, n, args.cons) if args.exp == "rep" else gen_pool(sd, n)
            p1, nf = evaluate(ev, an, sd)
            for s in STRATS: per[s].append(p1[s])
            nfold.append(nf)
        row = dict(exp=args.exp, cons=args.cons, n_unit=n, n_analysts=len(an), folds=int(np.mean(nfold)))
        for s in STRATS: row[s] = (float(np.mean(per[s])), float(np.std(per[s])))
        res.append(row)
        print(args.exp, args.cons, n, len(an), {s: round(row[s][0], 3) for s in STRATS}, flush=True)
    json.dump(res, open(f"{args.out}_{args.exp}_{args.cons}.json", "w"), indent=1)
