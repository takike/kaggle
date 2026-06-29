"""Leave-one-well-out validation of the offset-well dip prior fed into the aligner.
Compares final toe-RMSE: heel-trend prior vs offset-well dip prior."""
import os
import random
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from real_io import list_cases, load_case, split_known_pred  # noqa: E402
from predict_real import predict_case  # noqa: E402
from offset import StructuralCloud, dip_trend  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
TRAIN = "data_real/train"
cases = list_cases(TRAIN)
# Build the structural cloud once; case id = index into `cases` (for leave-one-out).
Xs, Ys, Cs, ids = [], [], [], []
for ci, cid in enumerate(cases):
    h, _ = load_case(TRAIN, cid)
    s = slice(None, None, 20)
    Xs.append(h["X"].values[s]); Ys.append(h["Y"].values[s])
    Cs.append((h["TVT"].values + h["Z"].values)[s])
    ids.append(np.full(np.shape(h["X"].values[s]), ci))
cloud = StructuralCloud(np.concatenate(Xs), np.concatenate(Ys),
                        np.concatenate(Cs), np.concatenate(ids))
random.seed(11)
samp = random.sample(range(len(cases)), N)

sq = {"heel": 0.0, "dip": 0.0}
n = 0
per = {"heel": [], "dip": []}
t0 = time.time()
nfallback = 0
for ci in samp:
    cid = cases[ci]
    h, tw = load_case("data_real/train", cid)
    known, pred = split_known_pred(h)
    if pred.sum() < 10:
        continue
    true = h["TVT"].values
    pr_heel = predict_case(h, tw)                       # heel-trend prior
    tr = dip_trend(h, cloud, exclude_case=ci)           # offset-well dip prior
    if tr is None:
        nfallback += 1
        pr_dip = pr_heel
    else:
        pr_dip = predict_case(h, tw, trend_override=tr)
    for key, pr in (("heel", pr_heel), ("dip", pr_dip)):
        e = (true - pr)[pred]
        sq[key] += np.sum(e ** 2)
        per[key].append(np.sqrt(np.mean(e ** 2)))
    n += pred.sum()

print(f"\n{len(per['heel'])} wells, {nfallback} dip-fallbacks, {time.time()-t0:.0f}s")
for key in ("heel", "dip"):
    p = np.array(per[key])
    print(f"{key:5s} prior: pooled {np.sqrt(sq[key]/n):.3f} | median "
          f"{np.median(p):.2f} | p90 {np.percentile(p,90):.2f} | max {p.max():.2f}")
