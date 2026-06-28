"""Held-out validation of the real-data geosteering predictor over many train
cases (we have true TVT there).  Reports pooled toe-RMSE (the LB metric) plus
per-case distribution, and tests a blend toward the heel trend for tail robustness.
"""
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from real_io import list_cases, load_case, split_known_pred  # noqa: E402
from predict_real import predict_case  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 250
random.seed(7)
cases = random.sample(list_cases("data_real/train"), N)

# Collect per-point errors for align and trend, to evaluate blends cheaply.
err_align, err_trend, wsizes, per_align = [], [], [], []
t0 = time.time()
for k, cid in enumerate(cases):
    h, tw = load_case("data_real/train", cid)
    known, pred = split_known_pred(h)
    if pred.sum() < 10:
        continue
    pr = predict_case(h, tw)                      # alignment prediction
    # trend-only prediction (e=0): reconstruct trend - Z
    MD = h["MD"].values; Z = h["Z"].values
    ps = int(np.argmax(pred))
    C_known = h["TVT_input"].values + Z
    tail = np.where(known)[0]; tail = tail[tail >= max(0, ps - 500)]
    r = np.clip(np.polyfit(MD[tail], C_known[tail], 1)[0], -0.5, 0.5) if len(tail) > 20 else 0.0
    trend_pred = (C_known[ps - 1] + r * (MD - MD[ps - 1])) - Z
    true = h["TVT"].values
    ea = (true - pr)[pred]; et = (true - trend_pred)[pred]
    err_align.append(ea); err_trend.append(et)
    per_align.append(np.sqrt(np.mean(ea ** 2)))
    if (k + 1) % 50 == 0:
        print(f"  {k+1}/{N}  {time.time()-t0:.0f}s", flush=True)

ea = np.concatenate(err_align); et = np.concatenate(err_trend)
per_align = np.array(per_align)


def pooled(e):
    return np.sqrt(np.mean(e ** 2))


print(f"\ncases={len(per_align)}  time={time.time()-t0:.0f}s")
print(f"trend-only pooled RMSE : {pooled(et):.3f}")
print(f"alignment  pooled RMSE : {pooled(ea):.3f}")
print(f"alignment  per-case    : mean {per_align.mean():.2f}  median "
      f"{np.median(per_align):.2f}  p90 {np.percentile(per_align,90):.2f}  "
      f"max {per_align.max():.2f}")
print("\nglobal blend a*align + (1-a)*trend:")
for a in (0.5, 0.7, 0.85, 1.0):
    eb = a * ea + (1 - a) * et
    print(f"  a={a:.2f}  pooled {pooled(eb):.3f}")
