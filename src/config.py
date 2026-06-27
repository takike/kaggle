"""Logical-name -> real-column-name mapping.

The pipeline never hard-codes dataset column names; it goes through this map.
To run on the real Kaggle ROGII data, inspect the CSV headers and edit the
right-hand strings (and DATA dir / file names) to match.  Nothing else changes.
"""

# Lateral train/test tables
COLS = {
    "well": "well_id",   # grouping id for each lateral well
    "md": "md",          # measured depth (ft)
    "tvd": "tvd",        # true vertical depth (ft)
    "x": "x",            # horizontal displacement / vertical section (ft)
    "inc": "inc",        # inclination (deg) -- optional, derived if absent
    "gr": "gr",          # lateral gamma ray (API)
    "target": "tvt",     # TARGET (only in train)
    # typewell reference log
    "tw_tvd": "tvd",     # typewell depth column
    "tw_gr": "gr",       # typewell gamma ray column
    "id": "id",          # submission row id (test only)
}

# Where real CSVs live (when available).  Synthetic mode ignores this.
DATA_DIR = "data"
FILES = {
    "train": "train.csv",
    "test": "test.csv",
    "typewell": "typewell.csv",
    "sample_submission": "sample_submission.csv",
}
