"""Recompute the per-day numbers in summary.parquet from the stored chains, without downloading.

    python -m scripts.rebuild_summary --data-dir data

Use after changing how the summary is derived (optlab/surface.py). Spot, India VIX, lot size and
rate are kept; ATM vols, skew, the variance-swap vol and quote counts are recomputed, and a column
the summary didn't have yet is added; realized vol is recomputed on write.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from optlab.market import store
from optlab.surface import daily_summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    summary = store.read_summary(args.data_dir)
    rows = []
    month_of = pd.to_datetime(summary["trade_date"]).dt.strftime("%Y-%m")
    for (underlying, month), group in summary.groupby([summary["underlying"], month_of]):
        path = store.chain_path(args.data_dir, underlying, month)
        chains = store.as_dates(pd.read_parquet(path), "trade_date", "expiry") if path.exists() else None
        for row in group.to_dict("records"):
            chain = (
                chains[chains["trade_date"] == row["trade_date"]] if chains is not None else pd.DataFrame()
            )
            rows.append({**row, **daily_summary(chain)})

    rebuilt = pd.DataFrame(rows)
    changed = {}
    for column in [c for c in rebuilt.columns if c.startswith(("atm_iv", "skew", "vs_vol"))]:
        if column not in summary:
            changed[column] = "added"
            continue
        before = summary.set_index(["trade_date", "underlying"])[column]
        after = rebuilt.set_index(["trade_date", "underlying"])[column]
        diff = ~np.isclose(before, after.reindex(before.index), equal_nan=True)
        changed[column] = int(diff.sum())
    store.write_summary(args.data_dir, rebuilt)
    print(f"Rebuilt {len(rebuilt)} rows. Values changed per column: {changed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
