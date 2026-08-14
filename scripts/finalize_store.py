"""Merge per-worker ShardStore part directories into one unified store.

Multi-GPU preprocessing writes one part dir per worker (msaflow_openfold.part0,
.part1, ...). Because shards are aligned across arrays and numbered per part,
merging is just: move each part's shard bins into the output under globally
renumbered shard indices, shift that part's `shard` column, and concatenate the
Parquet indices + meta. No embedding bytes are rewritten, so this is fast and
CPU-only (safe on a login node).

Usage:
    uv run python scripts/finalize_store.py \
        --parts msaflow_openfold.part0 msaflow_openfold.part1 ... \
        --output msaflow_openfold
    # or glob:
    uv run python scripts/finalize_store.py --parts msaflow_openfold.part* --output msaflow_openfold
"""
import argparse
import json
import logging
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from msaflow.data.shard_store import ARRAYS, _INDEX_COLS, _shard_path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts", nargs="+", required=True, help="part store dirs to merge")
    ap.add_argument("--output", required=True, help="unified store dir to create")
    ap.add_argument("--keep_parts", action="store_true",
                    help="copy shard bins instead of moving (parts left intact)")
    args = ap.parse_args()

    parts = [Path(p) for p in args.parts]
    for p in parts:
        if not (p / "index.parquet").is_file():
            raise SystemExit(f"not a store dir (no index.parquet): {p}")
    out = Path(args.output)
    for array in ARRAYS:
        (out / array).mkdir(parents=True, exist_ok=True)

    xfer = shutil.copy2 if args.keep_parts else shutil.move
    rows = {c: [] for c in _INDEX_COLS}
    global_shard = 0
    n_proteins = 0
    base_meta = None

    for p in parts:
        with open(p / "meta.json") as f:
            meta = json.load(f)
        if base_meta is None:
            base_meta = meta
        elif meta["arrays"] != base_meta["arrays"]:
            raise SystemExit(f"array schema mismatch in {p}")

        n_shards = meta["n_shards"]
        for local in range(n_shards):
            for array in ARRAYS:
                src = _shard_path(p, array, local)
                dst = _shard_path(out, array, global_shard + local)
                xfer(str(src), str(dst))
        logger.info("%s: %d shards → global %d..%d",
                    p.name, n_shards, global_shard, global_shard + n_shards - 1)

        t = pq.read_table(p / "index.parquet")
        cols = {c: t.column(c).to_pylist() for c in _INDEX_COLS}
        for c in _INDEX_COLS:
            if c == "shard":
                rows[c].extend(int(s) + global_shard for s in cols[c])
            else:
                rows[c].extend(cols[c])
        global_shard += n_shards
        n_proteins += meta["n_proteins"]

    schema = pa.schema([
        ("id", pa.string()), ("shard", pa.int32()),
        ("L", pa.int32()), ("N", pa.int32()), ("seq_len", pa.int32()),
        ("emb_row", pa.int64()), ("tok_off", pa.int64()), ("wt_off", pa.int64()),
        ("has_esm", pa.bool_()), ("neff", pa.float32()),
    ])
    pq.write_table(pa.table(rows, schema=schema), out / "index.parquet")

    base_meta["n_shards"] = global_shard
    base_meta["n_proteins"] = n_proteins
    base_meta["merged_from"] = [p.name for p in parts]
    with open(out / "meta.json", "w") as f:
        json.dump(base_meta, f, indent=2)

    logger.info("Done: %s (%d proteins, %d shards from %d parts)",
                out, n_proteins, global_shard, len(parts))


if __name__ == "__main__":
    main()
