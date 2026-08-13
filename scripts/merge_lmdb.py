"""Merge shard LMDBs with disjoint keys into a single LMDB.

Raw cursor-level copy — values are never unpickled, so this is I/O-bound and
safe to run on a login node (no GPU, no model load).

Usage:
    uv run python scripts/merge_lmdb.py \
        --inputs ~/data/msaflow_openfold_shard*.lmdb \
        --output ~/data/msaflow_openfold.lmdb

Duplicate keys across shards abort the merge by default (shards from
run_preprocess_multi.sh are disjoint, so a duplicate means a mixed-up input
set); pass --last_wins to overwrite instead.
"""

import argparse
import logging
from pathlib import Path

import lmdb

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

COMMIT_EVERY = 1000  # entries per write transaction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="shard LMDB files to merge")
    parser.add_argument("--output", required=True,
                        help="merged LMDB path (created/appended)")
    parser.add_argument("--map_size_gb", type=int, default=500)
    parser.add_argument("--last_wins", action="store_true",
                        help="overwrite on duplicate keys instead of aborting")
    args = parser.parse_args()

    for p in args.inputs:
        if not Path(p).is_file():
            raise SystemExit(f"input LMDB not found: {p}")

    out_env = lmdb.open(args.output, map_size=args.map_size_gb * 1024 ** 3,
                        subdir=False, meminit=False, map_async=True)
    n_copied = 0
    for src_path in args.inputs:
        src_env = lmdb.open(src_path, subdir=False, readonly=True, lock=False)
        logger.info("Merging %s (%d entries)", src_path, src_env.stat()["entries"])
        with src_env.begin() as src_txn:
            cursor = src_txn.cursor()
            has_next = cursor.first()
            while has_next:
                # Bounded write transactions keep memory flat on huge shards.
                with out_env.begin(write=True) as out_txn:
                    for _ in range(COMMIT_EVERY):
                        if not has_next:
                            break
                        key, value = cursor.item()
                        if not out_txn.put(key, value, overwrite=args.last_wins):
                            raise SystemExit(
                                f"duplicate key {key.decode()!r} in {src_path} "
                                f"(already in output) — shards should be disjoint; "
                                f"pass --last_wins to overwrite instead"
                            )
                        n_copied += 1
                        has_next = cursor.next()
        src_env.close()

    total = out_env.stat()["entries"]
    out_env.close()
    logger.info("Done: copied %d entries, output now holds %d total (%s)",
                n_copied, total, args.output)


if __name__ == "__main__":
    main()
