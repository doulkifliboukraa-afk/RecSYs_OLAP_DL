# Reproducing the evaluation

Requirements: Python 3, numpy, scipy, openpyxl.

Run order (from this folder):

1. `python3 run_final.py`   -> writes loo_{high,low}.csv, chron_{high,low}.csv,
   cs_{high,low}.csv to ./eval_out_final (set EVAL_OUT to change the directory).
   30 personas x 5 seeds (42, 7, 13, 99, 123) x 2 consistency conditions
   (A = 0.85, B = 0.40). Prints the leave-one-out, chronological, cold-start
   and by-role tables.
2. `python3 analysis2.py`   -> reads the CSVs and writes analysis2.json:
   mean/std and bootstrap 95% CIs, P@k / NDCG@k, persona-level Wilcoxon
   tests (n = 30), by-role NDCG@1, and the MF latent-factor sweep (2, 3, 5, 8).

All runs are seeded; a clean re-run reproduces analysis2.json exactly.

Files: build_usage_log.py (original 15 personas, schema, event generator),
build_usage_log_30.py (15 added personas), run_final.py (strategies and
protocols), analysis2.py (statistics).

Hybrid ranking = own usage > other analysts' usage > similarity (maximum
cosine similarity to items of the role with direct usage evidence) > random.
Ablation variants: own_only, collaborative (others only), sim_only.

Scale-up check (Supplementary S1):
3. `python3 scale_experiment_v2.py --exp rep --cons 0.85` (and `--cons 0.4`), `--exp pool`
   -> scale_v2_*.json (own-only / others-only / hybrid P@1 at increasing analyst counts).
