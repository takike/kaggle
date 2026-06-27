"""Write synthetic data to ./data in the expected Kaggle schema.

Useful to (a) see the exact table layout the pipeline expects and (b) exercise
the full file-based path (load_real) without Kaggle credentials.

    python src/make_data.py --wells 16 --test 4

Produces data/train.csv, data/test.csv, data/typewell.csv, data/sample_submission.csv
"""
import argparse
import os

import numpy as np

from synth import generate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wells", type=int, default=16)
    ap.add_argument("--test", type=int, default=4, help="# wells held out as test")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    df, tw = generate(n_wells=args.wells, seed=args.seed)
    df = df.drop(columns=["_z0"])
    wells = sorted(df.well_id.unique())
    test_wells = set(wells[-args.test:])
    train = df[~df.well_id.isin(test_wells)].copy()
    test_full = df[df.well_id.isin(test_wells)].copy().reset_index(drop=True)
    test = test_full.drop(columns=["tvt"]).copy()
    test.insert(0, "id", np.arange(len(test)))

    os.makedirs(args.out, exist_ok=True)
    train.to_csv(os.path.join(args.out, "train.csv"), index=False)
    test.to_csv(os.path.join(args.out, "test.csv"), index=False)
    tw.to_csv(os.path.join(args.out, "typewell.csv"), index=False)
    test_full.assign(id=np.arange(len(test_full)))[["id", "tvt"]].to_csv(
        os.path.join(args.out, "solution.csv"), index=False)  # for local scoring
    test[["id"]].assign(tvt=0.0).to_csv(
        os.path.join(args.out, "sample_submission.csv"), index=False)
    print(f"wrote {args.out}/  train={len(train)} test={len(test)} "
          f"typewell={len(tw)}  (test wells: {sorted(test_wells)})")


if __name__ == "__main__":
    main()
