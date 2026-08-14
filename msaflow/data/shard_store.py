"""Sharded binary store for MSAFlow preprocessed data (replaces LMDB).

Motivation: LMDB is a memory-mapped writable B-tree that relies on POSIX file
locks, which misbehave on NFS/GPFS. This store is instead a directory of plain,
append-only binary shards plus a Parquet index. Reads are pure read-only mmap
slices with no locks — exactly the access pattern network filesystems handle
well.

Layout (one store = one directory):

    <root>/
      meta.json                    # dtypes, dims, shard sizing, provenance
      index.parquet                # one row per protein → offsets
      msa_emb/    shard_00000.bin   # (ΣL, 128)  float16, residue-axis concat
      esm_emb/    shard_00000.bin   # (ΣL, 1280) float16, residue-axis concat
      msa_tokens/ shard_00000.bin   # (Σ N·L,)   int8,  ragged (N,L) flattened
      weights/    shard_00000.bin   # (ΣN,)      float32

Aligned sharding: all four arrays roll to the next shard at the SAME protein
boundary (driven by the esm_emb byte budget, the dominant array). So a single
`shard` number in the index locates a protein's data in every array, and the
two residue-axis embeddings share one `emb_row` offset.

index.parquet columns (one row per protein):
    id (str), shard (int32), L, N, seq_len (int32),
    emb_row (int64)  — residue-row offset of msa_emb & esm_emb within the shard,
    tok_off (int64)  — element offset of the flat msa_tokens within its shard,
    wt_off  (int64)  — element offset of weights within its shard,
    has_esm (bool), neff (float32, = weights.sum(), for bin-free filtering)
"""

import json
import logging
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# array name -> (numpy dtype, per-residue/per-row inner dim; None = ragged/1-D)
ARRAYS = {
    "msa_emb":    (np.float16, 128),
    "esm_emb":    (np.float16, 1280),
    "msa_tokens": (np.int8, None),   # flat N*L, reshaped via (N, L) from index
    "weights":    (np.float32, None),  # flat N
}
_INDEX_COLS = ["id", "shard", "L", "N", "seq_len",
               "emb_row", "tok_off", "wt_off", "has_esm", "neff"]


def _shard_path(root: Path, array: str, shard: int) -> Path:
    return root / array / f"shard_{shard:05d}.bin"


# ──────────────────────────────────────────────────────────────────────────────
# Writer
# ──────────────────────────────────────────────────────────────────────────────

class ShardStoreWriter:
    """Append-only writer producing one store directory.

    Typical use (one instance per preprocessing worker → its own part dir):

        w = ShardStoreWriter(part_dir, shard_max_gb=4, meta_extra={...})
        skip = w.existing_ids()                      # for --resume
        for id, ... in items:
            if id in skip: continue
            w.append(id, msa_emb, esm_emb, msa_tokens, weights)
        w.close()

    On close, writes index.parquet + meta.json for this directory. Re-opening an
    existing directory continues after its current shards (append-only), so a
    killed run can resume by skipping existing_ids() and appending fresh shards.
    """

    def __init__(self, root, shard_max_gb: float = 4.0, meta_extra: dict = None):
        self.root = Path(root)
        for array in ARRAYS:
            (self.root / array).mkdir(parents=True, exist_ok=True)
        self.shard_max_bytes = int(shard_max_gb * 1024 ** 3)
        self.meta_extra = meta_extra or {}

        # Load any existing index rows so a resumed run rewrites the full index.
        self._rows = {c: [] for c in _INDEX_COLS}
        self._existing_ids = set()
        idx_path = self.root / "index.parquet"
        if idx_path.is_file():
            t = pq.read_table(idx_path)
            for c in _INDEX_COLS:
                self._rows[c] = t.column(c).to_pylist()
            self._existing_ids = set(self._rows["id"])

        # Continue shard numbering after whatever already exists on disk.
        existing = sorted((self.root / "esm_emb").glob("shard_*.bin"))
        self.shard = (int(existing[-1].stem.split("_")[1]) + 1) if existing else 0
        self._fh = None            # dict array -> open file handle
        self._emb_rows = 0         # residue rows in current shard (msa_emb & esm_emb)
        self._tok_elems = 0        # int8 elements in current msa_tokens shard
        self._wt_elems = 0         # float32 elements in current weights shard

    def existing_ids(self) -> set:
        return set(self._existing_ids)

    def _open_shard(self):
        self._fh = {a: open(_shard_path(self.root, a, self.shard), "wb") for a in ARRAYS}
        self._emb_rows = self._tok_elems = self._wt_elems = 0

    def _rotate(self):
        if self._fh is not None:
            for f in self._fh.values():
                f.close()
        self.shard += 1
        self._open_shard()

    def append(self, id: str, msa_emb, esm_emb, msa_tokens, weights):
        """Append one protein. msa_emb (L,128) and esm_emb (L,1280) must share L."""
        msa_emb = np.ascontiguousarray(msa_emb, dtype=np.float16)
        esm_emb = np.ascontiguousarray(esm_emb, dtype=np.float16)
        msa_tokens = np.ascontiguousarray(msa_tokens, dtype=np.int8)
        weights = np.ascontiguousarray(weights, dtype=np.float32)

        L = msa_emb.shape[0]
        if esm_emb.shape[0] != L:
            raise ValueError(f"{id}: esm_emb L={esm_emb.shape[0]} != msa_emb L={L}")
        N, Lt = msa_tokens.shape
        if Lt != L:
            raise ValueError(f"{id}: msa_tokens L={Lt} != msa_emb L={L}")
        if weights.shape[0] != N:
            raise ValueError(f"{id}: weights N={weights.shape[0]} != tokens N={N}")

        if self._fh is None:
            self._open_shard()
        # Roll to a fresh shard when the dominant (esm) array would overflow the
        # byte budget — but always keep at least one protein per shard.
        would = (self._emb_rows + L) * ARRAYS["esm_emb"][1] * 2
        if self._emb_rows > 0 and would > self.shard_max_bytes:
            self._rotate()

        self._rows["id"].append(id)
        self._rows["shard"].append(self.shard)
        self._rows["L"].append(L)
        self._rows["N"].append(N)
        self._rows["seq_len"].append(L)
        self._rows["emb_row"].append(self._emb_rows)
        self._rows["tok_off"].append(self._tok_elems)
        self._rows["wt_off"].append(self._wt_elems)
        self._rows["has_esm"].append(True)
        self._rows["neff"].append(float(weights.sum()))

        self._fh["msa_emb"].write(msa_emb.tobytes())
        self._fh["esm_emb"].write(esm_emb.tobytes())
        self._fh["msa_tokens"].write(msa_tokens.tobytes())
        self._fh["weights"].write(weights.tobytes())
        self._emb_rows += L
        self._tok_elems += N * L
        self._wt_elems += N

    def close(self):
        if self._fh is not None:
            for f in self._fh.values():
                f.close()
            self._fh = None
        n_shards = self.shard + 1 if self._rows["id"] else 0
        schema = pa.schema([
            ("id", pa.string()), ("shard", pa.int32()),
            ("L", pa.int32()), ("N", pa.int32()), ("seq_len", pa.int32()),
            ("emb_row", pa.int64()), ("tok_off", pa.int64()), ("wt_off", pa.int64()),
            ("has_esm", pa.bool_()), ("neff", pa.float32()),
        ])
        pq.write_table(pa.table(self._rows, schema=schema), self.root / "index.parquet")

        meta = {
            "version": 1,
            "arrays": {
                "msa_emb":    {"dtype": "float16", "dim": 128, "axis": "residue"},
                "esm_emb":    {"dtype": "float16", "dim": 1280, "axis": "residue"},
                "msa_tokens": {"dtype": "int8", "layout": "ragged_NL", "axis": "msa_row"},
                "weights":    {"dtype": "float32", "axis": "msa_row"},
            },
            "shard_max_bytes": self.shard_max_bytes,
            "n_shards": n_shards,
            "n_proteins": len(self._rows["id"]),
        }
        meta.update(self.meta_extra)
        with open(self.root / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("ShardStore written: %s (%d proteins, %d shards)",
                    self.root, meta["n_proteins"], n_shards)


# ──────────────────────────────────────────────────────────────────────────────
# Reader
# ──────────────────────────────────────────────────────────────────────────────

class ShardStore:
    """Read-only mmap reader. Safe under DataLoader multiprocessing.

    get(id) returns a dict shaped like the old pickled LMDB entry:
        {msa_emb (L,128) f16, esm_emb (L,1280) f16, msa_tokens (N,L) int8,
         weights (N,) f32, seq_len int, query_seq None}
    Consumers cast dtypes as before (e.g. .astype(np.float32)).
    """

    def __init__(self, root):
        self.root = Path(root)
        with open(self.root / "meta.json") as f:
            self.meta = json.load(f)
        t = pq.read_table(self.root / "index.parquet")
        self._col = {c: t.column(c).to_numpy(zero_copy_only=False) for c in _INDEX_COLS}
        self._ids = list(self._col["id"])
        self._pos = {k: i for i, k in enumerate(self._ids)}
        self._mm = {}
        self._pid = os.getpid()

    def __len__(self):
        return len(self._ids)

    def keys(self):
        return list(self._ids)

    def __contains__(self, id):
        return id in self._pos

    def neff(self, id) -> float:
        return float(self._col["neff"][self._pos[id]])

    def all_neff(self) -> np.ndarray:
        return np.asarray(self._col["neff"], dtype=np.float32)

    def _memmap(self, array: str, shard: int):
        # Re-open mmaps after a fork so workers don't share parent handles.
        if self._pid != os.getpid():
            self._mm.clear()
            self._pid = os.getpid()
        key = (array, shard)
        mm = self._mm.get(key)
        if mm is None:
            dtype = ARRAYS[array][0]
            mm = np.memmap(_shard_path(self.root, array, shard), dtype=dtype, mode="r")
            self._mm[key] = mm
        return mm

    def get(self, id: str) -> dict:
        i = self._pos[id]
        shard = int(self._col["shard"][i])
        L = int(self._col["L"][i])
        N = int(self._col["N"][i])
        emb_row = int(self._col["emb_row"][i])
        tok_off = int(self._col["tok_off"][i])
        wt_off = int(self._col["wt_off"][i])

        d_msa = ARRAYS["msa_emb"][1]
        d_esm = ARRAYS["esm_emb"][1]
        msa_emb = self._memmap("msa_emb", shard)[emb_row * d_msa:(emb_row + L) * d_msa].reshape(L, d_msa)
        esm_emb = self._memmap("esm_emb", shard)[emb_row * d_esm:(emb_row + L) * d_esm].reshape(L, d_esm)
        tokens = self._memmap("msa_tokens", shard)[tok_off:tok_off + N * L].reshape(N, L)
        weights = self._memmap("weights", shard)[wt_off:wt_off + N]
        return {
            "msa_emb": msa_emb,
            "esm_emb": esm_emb,
            "msa_tokens": tokens,
            "weights": weights,
            "seq_len": L,
            "query_seq": None,
        }
