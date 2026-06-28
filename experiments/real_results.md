# Real ROGII data — held-out validation

Validated on real competition cases (train split, where true TVT is known on all
rows so toe predictions can be scored exactly). Metric = pooled RMSE of
(true_TVT − pred_TVT) over the to-predict (post-PS) rows — the competition metric.

`experiments/validate_real.py 250` (250 random cases, seed 7):

| Method | pooled toe-RMSE |
|---|---|
| trend-only (extrapolate heel structural dip) | 43.48 |
| **geosteering aligner** | **19.81** |

Aligner per-case distribution: mean 12.87 · **median 7.59** · p90 31.88 · max 91.91.

Notes:
- The aligner more than halves the trend-only error; on the median case it is ~7.6.
- The pooled score is tail-dominated (a few long, sparsely-interpreted wells with
  large structural change). A global blend toward the trend does **not** help
  (`a=1.0` is best) because the trend is even worse on exactly those hard cases —
  so the remaining gap needs a *confidence-aware* / learned correction, not a
  uniform shrink. See README "next steps".
- Submission to the live leaderboard is not possible from this environment: ROGII
  is a **Code Competition** (hidden test, notebook submission only) — direct CSV
  upload returns 403. These held-out numbers are the honest proxy.
