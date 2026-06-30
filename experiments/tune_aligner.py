"""Re-tune the aligner (lam, e_pad) on top of the offset-well dip prior.

With a much better prior the structural deviation is small, so the search band can
be tighter (less room to wander) and the stiffness may differ.  The dip prior is
computed once per well and cached; only the aligner params vary across configs.
"""
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from real_io import list_cases, load_case, split_known_pred  # noqa: E402
from predict_real import predict_case  # noqa: E402
from offset import StructuralCloud, dip_trend  # noqa: E402

CLOUD = "/tmp/claude-0/-home-user-kaggle/6bd06b12-7fd0-5d4d-b9a6-49398577dea1/scratchpad/cloud.npz"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 120
d = np.load(CLOUD, allow_pickle=True)
cloud = StructuralCloud(d["X"], d["Y"], d["C"], d["CASE"])
cases = list(d["cases"])
random.seed(11)
samp = random.sample(range(len(cases)), N)

# Cache (h, tw, trend, pred, true) per well.
cache = []
for ci in samp:
    h, tw = load_case("data_real/train", cases[ci])
    known, pred = split_known_pred(h)
    if pred.sum() < 10:
        continue
    tr = dip_trend(h, cloud, exclude_case=ci)
    cache.append((h, tw, tr, pred.values if hasattr(pred, "values") else pred,
                  h["TVT"].values))
print(f"cached {len(cache)} wells")


def ev(**kw):
    sq = 0.0; n = 0; per = []
    for h, tw, tr, pred, true in cache:
        pr = predict_case(h, tw, trend_override=tr, **kw)
        e = (true - pr)[pred]
        sq += np.sum(e ** 2); n += pred.sum(); per.append(np.sqrt(np.mean(e ** 2)))
    per = np.array(per)
    return np.sqrt(sq / n), np.median(per), np.percentile(per, 90)


t0 = time.time()
print("baseline (lam100 e_pad200):", "%.3f %.2f %.2f" % ev(lam=100, e_pad=200))
for e_pad in (80, 120, 160):
    for lam in (40, 70, 100):
        p, md, p90 = ev(lam=lam, e_pad=e_pad)
        print(f"e_pad={e_pad:3d} lam={lam:3d} -> pooled {p:.3f} median {md:.2f} p90 {p90:.2f}")
print("time %.0fs" % (time.time() - t0))
