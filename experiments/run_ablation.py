"""Ablation: average held-out-wells (GroupKFold OOF) RMSE across seeds.

Quantifies the contribution of each stage of the pipeline.  Writes a markdown
table to experiments/ablation_results.md.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import config as C  # noqa: E402
from synth import generate  # noqa: E402
from pipeline import align_ensemble, cv_oof, rmse, smooth_per_well  # noqa: E402
from align import align_dataframe  # noqa: E402
from features import build_features  # noqa: E402

SEEDS = [42, 1, 7, 100, 5]
N_WELLS = 14
rows = {}


def add(name, seed, val):
    rows.setdefault(name, {})[seed] = val


for seed in SEEDS:
    df, tw = generate(n_wells=N_WELLS, seed=seed)
    cols = C.COLS
    y = df[cols["target"]]
    groups = df[cols["well"]].values

    add("naive: global mean", seed, rmse(y, np.full(len(y), y.mean())))

    # Single aligner (no prior, no ensemble) -- shows raw correlation power.
    a0 = align_dataframe(df, tw, cols, prior_w=0.0)
    add("align: single, no prior", seed, rmse(y, a0["tvt_align"]))

    # Single aligner with zone prior.
    a1 = align_dataframe(df, tw, cols)  # defaults incl. prior_w=8
    add("align: single + prior", seed, rmse(y, a1["tvt_align"]))

    # Ensemble aligner.
    ae = align_ensemble(df, tw, cols)
    add("align: ensemble", seed, rmse(y, ae["tvt_align"]))

    # ML direct vs residual on ensemble.
    feats = build_features(df, cols, ae)
    oof_direct = cv_oof(df, feats, y, groups)
    add("ML: LGBM direct", seed, rmse(y, oof_direct))
    oof_res = cv_oof(df, feats, y, groups, residual_base=ae["tvt_align"])
    add("ML: ensemble + LGBM residual", seed, rmse(y, oof_res))

    # + smoothing (final).
    sm = smooth_per_well(df, oof_res, cols, win=15)
    add("FINAL: + along-hole smoothing", seed, rmse(y, sm))
    print(f"seed {seed} done")

order = ["naive: global mean", "align: single, no prior", "align: single + prior",
         "align: ensemble", "ML: LGBM direct", "ML: ensemble + LGBM residual",
         "FINAL: + along-hole smoothing"]

lines = ["# Ablation — held-out-wells RMSE (GroupKFold OOF)", "",
         f"Synthetic geosteering data, {N_WELLS} wells, averaged over seeds {SEEDS}.", "",
         "| Stage | mean RMSE | per-seed |", "|---|---|---|"]
for name in order:
    d = rows[name]
    vals = [d[s] for s in SEEDS]
    mean = np.mean(vals)
    per = ", ".join(f"{v:.2f}" for v in vals)
    lines.append(f"| {name} | **{mean:.3f}** | {per} |")

txt = "\n".join(lines) + "\n"
out = os.path.join(os.path.dirname(__file__), "ablation_results.md")
with open(out, "w") as f:
    f.write(txt)
print("\n" + txt)
