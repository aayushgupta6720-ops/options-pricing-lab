"""Apply the chain filters added since the stored chains were built, without downloading again.

    python -m scripts.reclean_chains --data-dir data

Drops the quotes that leave a static arbitrage (chain.arbitrage_free), day by day and expiry by
expiry, from every file under chains/. Run scripts.rebuild_summary and then scripts.calibrate --force
afterwards, since both read the chains.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from optlab.market import store
from optlab.market.chain import arbitrage_free


def reclean(chain: pd.DataFrame) -> pd.DataFrame:
    keep = np.ones(len(chain), dtype=bool)
    positions = np.arange(len(chain))
    for _, day in chain.assign(_pos=positions).groupby("trade_date", sort=False):
        keep[day["_pos"].to_numpy()] = arbitrage_free(day.drop(columns="_pos"))
    return chain[keep].reset_index(drop=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    files = sorted((args.data_dir / "chains").glob("*/*.parquet"))
    before = after = 0
    for path in files:
        chain = pd.read_parquet(path)
        cleaned = reclean(chain)
        before, after = before + len(chain), after + len(cleaned)
        if len(cleaned) < len(chain):
            store._write(cleaned, path)
    dropped = before - after
    print(
        f"{len(files)} chain files: dropped {dropped:,} of {before:,} quotes ({dropped / max(before, 1):.2%})."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
