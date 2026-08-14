"""
Orphan benchmark data preparation.

Reads the find_low_msa.sh output, then:
  1. Extracts query sequences from LMDB (msa_tokens[0] = query)
  2. Copies sparse a3m files to shallow_msa_dir/{protein_id}.a3m
  3. Downloads reference CIF files from RCSB PDB API
  4. Writes orphan_proteins.fasta

Expected find_low_msa output format (lines starting with a PDB key):
  protein_id                    n_msa_seqs  seq_len
  5sbj-assembly1__A                      2       30
  ...
"""

import argparse
import csv
import logging
import os
import shutil
import time
import urllib.request
from pathlib import Path

import numpy as np

from msaflow.data.shard_store import ShardStore

logger = logging.getLogger(__name__)

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY") + ["-", "X"]


def parse_orphan_list(path: str, threshold: int) -> list[str]:
    """Parse find_low_msa output and return protein IDs below threshold."""
    ids = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            # Header / separator lines
            if not line or line.startswith("=") or line.startswith("protein_id") or line.startswith("-"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    n_msa = int(parts[1])
                except ValueError:
                    continue
                if n_msa < threshold:
                    ids.append(parts[0])
    return ids


def tokens_to_seq(tokens: np.ndarray) -> str:
    """Convert integer token array to amino acid string (no gaps)."""
    return "".join(AA_LIST[i] for i in tokens if AA_LIST[i] != "-")


def download_cif(pdb_id: str, out_path: Path) -> bool:
    """Download mmCIF from RCSB PDB. Returns True on success."""
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            out_path.write_bytes(resp.read())
        logger.info("  Downloaded %s → %s", url, out_path)
        return True
    except Exception as exc:
        logger.warning("  Failed to download %s: %s", url, exc)
        return False


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  [ %(name)s ]  %(message)s",
        datefmt="%Y-%m-%d %H:%M",
    )

    parser = argparse.ArgumentParser(description="Prepare orphan benchmark data")
    parser.add_argument("--orphan_list",     required=True,
                        help="Output of find_low_msa.sh (TSV with protein_id, n_msa, seq_len)")
    parser.add_argument("--csv",             required=True,
                        help="monomer_protein.csv (pdb_id, chain_id)")
    parser.add_argument("--store_path",      required=True,
                        help="ShardStore with msa_tokens per protein")
    parser.add_argument("--a3m_dir",         required=True,
                        help="Directory with *.a3m files (one per protein)")
    parser.add_argument("--output_fasta",    required=True,
                        help="Output FASTA path")
    parser.add_argument("--shallow_msa_dir", required=True,
                        help="Output directory for sparse a3m copies")
    parser.add_argument("--ref_pdb_dir",     required=True,
                        help="Output directory for reference CIF files")
    parser.add_argument("--threshold",       type=int, default=15,
                        help="MSA sequence count threshold (< N = orphan)")
    args = parser.parse_args()

    Path(args.shallow_msa_dir).mkdir(parents=True, exist_ok=True)
    Path(args.ref_pdb_dir).mkdir(parents=True, exist_ok=True)

    # ── Parse orphan list ──────────────────────────────────────────────────────
    orphan_ids = parse_orphan_list(args.orphan_list, args.threshold)
    logger.info("Orphan proteins (n_msa < %d): %d", args.threshold, len(orphan_ids))
    if not orphan_ids:
        logger.error("No orphan proteins found in %s", args.orphan_list)
        return

    orphan_set = set(orphan_ids)

    # ── Open store ─────────────────────────────────────────────────────────────
    store = ShardStore(args.store_path)

    # ── Process each orphan ────────────────────────────────────────────────────
    fasta_entries = []
    a3m_dir = Path(args.a3m_dir)
    shallow_dir = Path(args.shallow_msa_dir)
    ref_dir = Path(args.ref_pdb_dir)

    missing_store = []
    missing_a3m  = []

    for protein_id in orphan_ids:
        logger.info("Processing %s", protein_id)

        # ── Get query sequence from the store ──────────────────────────────────
        if protein_id not in store:
            logger.warning("  NOT in store: %s", protein_id)
            missing_store.append(protein_id)
            continue

        data = store.get(protein_id)
        # The store does not keep query_seq; recover it from the first MSA row.
        query_seq = tokens_to_seq(np.array(data["msa_tokens"][0]))
        query_seq = query_seq.replace("-", "")
        fasta_entries.append((protein_id, query_seq))

        # ── Copy sparse a3m ────────────────────────────────────────────────────
        src_a3m = a3m_dir / f"{protein_id}.a3m"
        dst_a3m = shallow_dir / f"{protein_id}.a3m"

        if not dst_a3m.exists():
            if src_a3m.exists():
                shutil.copy2(src_a3m, dst_a3m)
                logger.info("  Copied a3m: %s", dst_a3m)
            else:
                # a3m not found — write depth-1 a3m from query seq
                logger.warning("  a3m not found: %s; writing depth-1 a3m", src_a3m)
                missing_a3m.append(protein_id)
                with open(dst_a3m, "w") as fh:
                    fh.write(f">query\n{query_seq}\n")

        # ── Download reference CIF ─────────────────────────────────────────────
        # protein_id format: {pdb_id}-assembly1__{chain_id}
        # extract bare pdb_id (e.g. "5sbj")
        raw_pdb = protein_id.split("-")[0].lower()
        ref_cif = ref_dir / f"{protein_id}.cif"

        if not ref_cif.exists():
            # Try exact 4-letter PDB ID first
            ok = download_cif(raw_pdb, ref_cif)
            if not ok:
                logger.warning("  Reference CIF unavailable for %s (pdb: %s)", protein_id, raw_pdb)
            time.sleep(0.1)   # be polite to RCSB

    # ── Write FASTA ────────────────────────────────────────────────────────────
    with open(args.output_fasta, "w") as fh:
        for name, seq in fasta_entries:
            fh.write(f">{name}\n{seq}\n")
    logger.info("Wrote %d sequences to %s", len(fasta_entries), args.output_fasta)

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Orphan data preparation complete")
    print(f"  Total orphans        : {len(orphan_ids)}")
    print(f"  FASTA written        : {len(fasta_entries)}")
    print(f"  Missing from store   : {len(missing_store)}")
    print(f"  Missing a3m (depth-1): {len(missing_a3m)}")
    ref_count = sum(1 for p in orphan_ids
                    if (ref_dir / f"{p}.cif").exists())
    print(f"  Reference CIFs found : {ref_count}/{len(orphan_ids)}")
    print(f"  Output FASTA         : {args.output_fasta}")
    print(f"  Shallow MSA dir      : {args.shallow_msa_dir}")
    print(f"  Ref PDB dir          : {args.ref_pdb_dir}")
    print("=" * 60)

    if missing_store:
        print("\nProteins not found in LMDB (excluded from benchmark):")
        for pid in missing_store:
            print(f"  {pid}")
    if missing_a3m:
        print("\nProteins with missing a3m (using depth-1 fallback):")
        for pid in missing_a3m:
            print(f"  {pid}")


if __name__ == "__main__":
    main()
