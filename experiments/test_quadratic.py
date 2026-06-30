"""Compare planar (order=1) vs quadratic (order=2) offset-well dip prior,
both as a raw toe-C prior and as the final aligner output."""
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from real_io import list_cases, load_case, split_known_pred  # noqa: E402
from predict_real import predict_case  # noqa: E402
from offset import StructuralCloud, dip_trend  # noqa: E402

CLOUD = "/tmp/claude-0/-home-user-kaggle/6bd06b12-7fd0-5d4d-b9a6-49398577dea1/scratchpad/cloud.npz"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
d = np.load(CLOUD, allow_pickle=True)
cloud = StructuralCloud(d["X"], d["Y"], d["C"], d["CASE"])
cases = list(d["cases"])
random.seed(11)
samp = random.sample(range(len(cases)), N)

acc = {1: dict(psq=0.0, fsq=0.0, per=[]), 2: dict(psq=0.0, fsq=0.0, per=[])}
n = 0
nfb = {1: 0, 2: 0}
for ci in samp:
    h, tw = load_case("data_real/train", cases[ci])
    known, pred = split_known_pred(h)
    if pred.sum() < 10:
        continue
    Z = h["Z"].values; true = true_tvt = h["TVT"].values
    Ctrue = true + Z
    pr_mask = pred.values if hasattr(pred, "values") else pred
    n += pr_mask.sum()
    for order in (1, 2):
        tr = dip_trend(h, cloud, exclude_case=ci, order=order)
        if tr is None:
            nfb[order] += 1
            acc[order]["per"].append(np.nan)
            continue
        # raw prior C error
        acc[order]["psq"] += np.sum((Ctrue[pr_mask] - tr[pr_mask]) ** 2)
        pr = predict_case(h, tw, trend_override=tr)
        e = (true - pr)[pr_mask]
        acc[order]["fsq"] += np.sum(e ** 2)
        acc[order]["per"].append(np.sqrt(np.mean(e ** 2)))

for order in (1, 2):
    per = np.array([p for p in acc[order]["per"] if not np.isnan(p)])
    print(f"order={order}: prior-C RMSE {np.sqrt(acc[order]['psq']/n):.3f} | "
          f"final pooled {np.sqrt(acc[order]['fsq']/n):.3f} | "
          f"median {np.median(per):.2f} | p90 {np.percentile(per,90):.2f} | "
          f"fallbacks {nfb[order]}")
