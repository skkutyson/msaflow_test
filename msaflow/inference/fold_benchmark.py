"""
MSAFlow Fold Benchmark — pLDDT / TM-score evaluation.

Pipeline (mirrors paper Section 4.1 / Table 2 evaluation):
  1. Read test proteins (FASTA, one sequence per entry).
  2. Generate MSA sequences via one of four modes:

       zeroshot  — 10 seeds × 32 seqs each (Section 4.2 protocol):
                   fold each seed's MSA separately with Protenix, report best pLDDT.

       fewshot   — Syn+Rec augmentation of a shallow MSA (Section 4.2):
                   requires --shallow_msa_dir and optionally --protenix_ckpt.
                   Folds the combined augmented MSA once.

       shallow   — Orphan protein baseline: fold with the original sparse MSA
                   as-is (no augmentation). Requires --shallow_msa_dir.
                   Use this to measure TM-score BEFORE MSAFlow augmentation,
                   then compare against fewshot to quantify the uplift.

       nomsa     — Baseline: skip MSA generation, fold with sequence only.

  3. Build Protenix JSON inputs pointing to the generated A3M.
  4. Run `protenix pred` for each protein (/ seed).
  5. Extract pLDDT from the output CIF/PDB.
  6. Optionally compute TM-score vs reference PDB using TMscore binary.
  7. Write summary CSV and print aggregate statistics.

Usage (zero-shot):
    python msaflow/inference/fold_benchmark.py \\
        --fasta          data/foldbench_monomer.fasta \\
        --decoder_ckpt   runs/decoder/decoder_ema_final.pt \\
        --latent_fm_ckpt runs/latent_fm/latent_fm_ema_final.pt \\
        --output_dir     runs/fold_benchmark \\
        --mode           zeroshot \\
        [--ref_pdb_dir   data/reference_pdbs/] \\
        [--device cuda]  [--n_seeds 10] [--n_seqs 32] [--n_steps 100]

Usage (no-MSA baseline):
    python msaflow/inference/fold_benchmark.py \\
        --fasta          data/foldbench_monomer.fasta \\
        --decoder_ckpt   dummy  --latent_fm_ckpt dummy \\
        --output_dir     runs/fold_benchmark_nomsa \\
        --mode           nomsa

Usage (few-shot):
    python msaflow/inference/fold_benchmark.py \\
        --fasta          data/foldbench_monomer.fasta \\
        --decoder_ckpt   runs/decoder/decoder_ema_final.pt \\
        --latent_fm_ckpt runs/latent_fm/latent_fm_ema_final.pt \\
        --output_dir     runs/fold_benchmark_fewshot \\
        --mode           fewshot \\
        --shallow_msa_dir data/shallow_msas/ \\
        [--protenix_ckpt  runs/protenix/protenix_base.pt]

Usage (shallow / orphan baseline):
    python msaflow/inference/fold_benchmark.py \\
        --fasta           data/orphan_proteins.fasta \\
        --decoder_ckpt    dummy  --latent_fm_ckpt dummy \\
        --output_dir      runs/orphan_shallow \\
        --mode            shallow \\
        --shallow_msa_dir data/shallow_msas/ \\
        --ref_pdb_dir     data/reference_pdbs/
"""

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# FASTA parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_fasta(path: str) -> list[tuple[str, str]]:
    """Return list of (name, sequence) pairs."""
    results = []
    name, buf = None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if name is not None:
                    results.append((name, "".join(buf)))
                name = line[1:].split()[0]
                buf = []
            elif line:
                buf.append(line.upper().replace(" ", ""))
    if name is not None:
        results.append((name, "".join(buf)))
    return results


def parse_a3m_seqs(path: str) -> list[str]:
    """Read sequences from an A3M file (first entry = query, rest = hits)."""
    seqs = []
    buf = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if buf:
                    seqs.append("".join(buf))
                buf = []
            elif line:
                buf.append(line.upper())
    if buf:
        seqs.append("".join(buf))
    return seqs


# ─────────────────────────────────────────────────────────────────────────────
# Protenix input JSON builder
# ─────────────────────────────────────────────────────────────────────────────

def build_protenix_json(
    name: str,
    query_seq: str,
    msa_a3m_path: Optional[str] = None,
) -> dict:
    """
    Build a single Protenix prediction task dict.

    Uses `unpairedMsaPath` for the generated A3M (new Protenix format).
    If no MSA path is provided, Protenix will fold with sequence only.
    """
    protein_chain: dict = {"sequence": query_seq, "count": 1}
    if msa_a3m_path is not None:
        protein_chain["unpairedMsaPath"] = str(msa_a3m_path)
        protein_chain["pairedMsaPath"] = str(msa_a3m_path)

    return {
        "name": name,
        "sequences": [{"proteinChain": protein_chain}],
        "modelSeeds": [42],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Run Protenix
# ─────────────────────────────────────────────────────────────────────────────

def run_protenix(
    input_json: str,
    output_dir: str,
    model_name: str = "protenix_base_default_v1.0.0",
    protenix_dir: Optional[str] = None,
    use_msa: bool = True,
    seeds: str = "101",
    n_sample: int = 5,
) -> subprocess.CompletedProcess:
    """Invoke `protenix pred` as a subprocess.

    Builds a clean PYTHONPATH that strips the MSA_FLOW repo root so that the
    local esm/ submodule does not shadow the installed fair-esm package and
    cause "AttributeError: module 'esm' has no attribute 'data'".
    """
    cmd = [
        "protenix", "pred",
        "-i", input_json,
        "-o", output_dir,
        "-n", model_name,
        # Disable all custom CUDA kernels — fall back to pure PyTorch.
        # Required when cuequivariance / fast_layer_norm are not compiled
        # for the current GPU architecture.
        "--enable_fusion",  "False",
        "--trimul_kernel",  "torch",
        "--triatt_kernel",  "torch",
        "--use_msa",        str(use_msa),
        # N seeds x N_sample samples/seed  (e.g. 5 seeds x 5 = 25 predictions)
        # NB: `protenix pred` CLI uses -e/--sample (not --sample_diffusion.N_sample,
        # which is the runner/inference.py config-override form).
        "--seeds",   seeds,
        "--sample",  str(n_sample),
    ]

    # Repo root = two levels above this file (msaflow/inference/fold_benchmark.py)
    repo_root = str(Path(__file__).parents[2].resolve())
    protenix_root = str((Path(__file__).parents[2] / "Protenix").resolve())

    # Strip $REPO_DIR from PYTHONPATH — it exposes the local esm/ submodule
    # which shadows the installed fair-esm package inside Protenix subprocess.
    old_path = os.environ.get("PYTHONPATH", "")
    clean_paths = [
        p for p in old_path.split(os.pathsep)
        if p and p.rstrip("/") != repo_root.rstrip("/")
    ]
    if protenix_root not in clean_paths:
        clean_paths.insert(0, protenix_root)
    if protenix_dir and protenix_dir not in clean_paths:
        clean_paths.insert(0, protenix_dir)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(clean_paths)

    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        logger.warning("protenix stderr:\n%s", result.stderr[-2000:])
    return result


# ─────────────────────────────────────────────────────────────────────────────
# pLDDT extraction from CIF output
# ─────────────────────────────────────────────────────────────────────────────

def extract_plddt_from_cif(cif_path: str) -> float:
    """
    Parse mean pLDDT from a Protenix output CIF file.

    Protenix writes per-residue B-factor as pLDDT (scaled 0–100).
    We average across all protein CA atoms.
    """
    plddt_values = []
    with open(cif_path) as fh:
        in_atom_site = False
        col_idx = {}
        for line in fh:
            line = line.rstrip()
            if line.startswith("_atom_site."):
                col_name = line.split(".")[1]
                col_idx[col_name] = len(col_idx)
                in_atom_site = True
                continue
            if in_atom_site and line.startswith("_"):
                in_atom_site = False
                col_idx = {}
                continue
            if in_atom_site and line and not line.startswith("#"):
                parts = line.split()
                if len(parts) <= max(col_idx.values(), default=-1):
                    continue
                group = parts[col_idx["group_PDB"]] if "group_PDB" in col_idx else "ATOM"
                atom_name = parts[col_idx["label_atom_id"]] if "label_atom_id" in col_idx else "CA"
                b_val = parts[col_idx["B_iso_or_equiv"]] if "B_iso_or_equiv" in col_idx else None
                if group == "ATOM" and atom_name == "CA" and b_val is not None:
                    try:
                        plddt_values.append(float(b_val))
                    except ValueError:
                        pass

    if not plddt_values:
        logger.warning("No CA atoms found in %s", cif_path)
        return float("nan")
    return float(np.mean(plddt_values))


def find_protenix_output_cif(output_dir: str, name: str) -> Optional[str]:
    """Find the Protenix output CIF file for a given prediction name."""
    base = Path(output_dir)
    for ext in ("*.cif", "*.pdb"):
        matches = list(base.rglob(f"{name}*{ext[1:]}"))
        if matches:
            return str(matches[0])
    matches = sorted(base.rglob("*.cif"))
    return str(matches[0]) if matches else None


# ─────────────────────────────────────────────────────────────────────────────
# TM-score computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_tmscore(
    pred_pdb: str,
    ref_pdb: str,
    tmscore_bin: str = "TMscore",
) -> tuple[float, float]:
    """Run TMscore binary and return (TM-score, RMSD)."""
    try:
        result = subprocess.run(
            [tmscore_bin, pred_pdb, ref_pdb],
            capture_output=True, text=True, timeout=60,
        )
        tm_score, rmsd = float("nan"), float("nan")
        for line in result.stdout.splitlines():
            if line.startswith("TM-score="):
                try:
                    tm_score = float(line.split("=")[1].split()[0])
                except (IndexError, ValueError):
                    pass
            if "RMSD of" in line and "aligned" in line:
                try:
                    rmsd = float(line.split("RMSD=")[1].split(",")[0].strip())
                except (IndexError, ValueError):
                    pass
        return tm_score, rmsd
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.warning("TMscore failed: %s", exc)
        return float("nan"), float("nan")


def _maybe_tmscore(args, prot_name: str, cif: Optional[str]) -> tuple[float, float]:
    """
    Compute TM-score/RMSD of a predicted CIF against the reference structure,
    if --ref_pdb_dir is set and a {prot_name}.pdb or .cif reference exists.
    Returns (nan, nan) otherwise.
    """
    if not (args.ref_pdb_dir and cif):
        return float("nan"), float("nan")
    ref_pdb = Path(args.ref_pdb_dir) / f"{prot_name}.pdb"
    if not ref_pdb.exists():
        ref_pdb = Path(args.ref_pdb_dir) / f"{prot_name}.cif"
    if not ref_pdb.exists():
        return float("nan"), float("nan")
    tm_score, rmsd = compute_tmscore(cif, str(ref_pdb), args.tmscore_bin)
    logger.info("  TM-score=%.4f  RMSD=%.2f Å", tm_score, rmsd)
    return tm_score, rmsd


# ─────────────────────────────────────────────────────────────────────────────
# Per-protein folding helpers
# ─────────────────────────────────────────────────────────────────────────────

def fold_once(
    prot_name: str,
    query_seq: str,
    a3m_path: Optional[Path],
    fold_dir: Path,
    args,
    use_msa: bool = True,
    tag: str = "",
) -> tuple[float, Optional[str]]:
    """
    Build Protenix input, run Protenix, extract pLDDT.

    Returns (plddt, cif_path_or_None).
    """
    run_name = f"{prot_name}{tag}"
    run_dir = fold_dir / run_name
    run_dir.mkdir(exist_ok=True)

    task = build_protenix_json(
        name=run_name,
        query_seq=query_seq,
        msa_a3m_path=str(a3m_path) if (a3m_path is not None and a3m_path.exists()) else None,
    )
    json_path = run_dir / f"{run_name}_input.json"
    with open(json_path, "w") as fh:
        json.dump([task], fh, indent=2)

    protenix_out_dir = run_dir / "protenix_out"
    protenix_out_dir.mkdir(exist_ok=True)

    pred_cif = find_protenix_output_cif(str(protenix_out_dir), run_name)
    if pred_cif and not args.refold:
        logger.info("  Protenix output exists, skipping: %s", pred_cif)
        return extract_plddt_from_cif(pred_cif), pred_cif

    result = run_protenix(
        input_json=str(json_path),
        output_dir=str(protenix_out_dir),
        model_name=args.protenix_model,
        use_msa=use_msa,
        seeds=args.fold_seeds,
        n_sample=args.n_sample,
    )
    if result.returncode != 0:
        logger.error("  Protenix failed (exit %d) for %s", result.returncode, run_name)
        return float("nan"), None

    pred_cif = find_protenix_output_cif(str(protenix_out_dir), run_name)
    if pred_cif is None:
        logger.error("  CIF not found for %s", run_name)
        return float("nan"), None

    return extract_plddt_from_cif(pred_cif), pred_cif


# ─────────────────────────────────────────────────────────────────────────────
# Main benchmark loop
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  [ %(name)s ]  %(message)s",
        datefmt="%Y-%m-%d %H:%M",
    )

    # Resolve mode (--no_msa is a deprecated alias for --mode nomsa)
    mode = args.mode
    if args.no_msa:
        mode = "nomsa"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    msa_dir = output_dir / "msas"
    fold_dir = output_dir / "folds"
    msa_dir.mkdir(exist_ok=True)
    fold_dir.mkdir(exist_ok=True)

    # ── Load models (skip for modes that don't generate sequences) ────────────
    _generation_modes = {"zeroshot", "fewshot"}
    if mode in _generation_modes:
        logger.info("Loading SFM decoder from %s", args.decoder_ckpt)
        sys.path.insert(0, str(Path(__file__).parents[2]))
        from msaflow.inference.generate import (
            load_sfm_decoder, load_latent_fm, load_esm2, load_protenix,
            generate_zeroshot_seeds, augment_shallow, write_a3m,
        )
        decoder = load_sfm_decoder(args.decoder_ckpt, device)
        latent_fm = load_latent_fm(args.latent_fm_ckpt, device)
        logger.info("Loading ESM2-650M ...")
        esm_model, alphabet = load_esm2(device)

        protenix_model = None
        if mode == "fewshot" and args.protenix_ckpt:
            logger.info("Loading Protenix for Rec track from %s", args.protenix_ckpt)
            protenix_model = load_protenix(args.protenix_ckpt, device)
            logger.info("Protenix loaded for Rec track.")
        elif mode == "fewshot":
            logger.warning(
                "--protenix_ckpt not provided; few-shot will run Syn track only "
                "(no Rec track from shallow MSA embedding)."
            )

        logger.info("All models loaded.")
    else:
        logger.info("Mode=%s: skipping MSA generation model loading.", mode)

    # ── Read test proteins ─────────────────────────────────────────────────────
    proteins = parse_fasta(args.fasta)
    logger.info("Loaded %d proteins from %s", len(proteins), args.fasta)

    # ── Shard the protein list across parallel workers ─────────────────────────
    if args.num_shards > 1:
        proteins = proteins[args.shard_id :: args.num_shards]
        logger.info("Shard %d/%d: processing %d proteins",
                    args.shard_id, args.num_shards, len(proteins))

    results = []

    for prot_idx, (prot_name, query_seq) in enumerate(proteins):
        logger.info("=" * 60)
        logger.info("[%d/%d] %s  L=%d  mode=%s",
                    prot_idx + 1, len(proteins), prot_name, len(query_seq), mode)

        # ── Zero-shot: fold each seed separately, report best pLDDT ───────────
        if mode == "zeroshot":
            prot_msa_dir = msa_dir / prot_name
            prot_msa_dir.mkdir(exist_ok=True)

            best_plddt = float("nan")
            best_seed = -1
            best_cif = None

            for seed, seqs in generate_zeroshot_seeds(
                query_seq=query_seq,
                decoder=decoder,
                latent_fm=latent_fm,
                esm_model=esm_model,
                alphabet=alphabet,
                n_seeds=args.n_seeds,
                n_seqs_per_seed=args.n_seqs,
                n_steps=args.n_steps,
                temperature=args.temperature,
                device=device,
            ):
                seed_a3m = prot_msa_dir / f"seed_{seed}.a3m"
                if not seed_a3m.exists() or args.regenerate:
                    write_a3m(query_seq, seqs, str(seed_a3m), prefix=prot_name)
                    logger.info("  Seed %d: wrote %d seqs to %s", seed, len(seqs), seed_a3m)

                plddt, cif = fold_once(
                    prot_name=prot_name,
                    query_seq=query_seq,
                    a3m_path=seed_a3m,
                    fold_dir=fold_dir,
                    args=args,
                    use_msa=True,
                    tag=f"_seed{seed}",
                )
                logger.info("  Seed %d pLDDT=%.2f  (best so far=%.2f)",
                            seed, plddt, best_plddt if not np.isnan(best_plddt) else -1)

                if np.isnan(best_plddt) or plddt > best_plddt:
                    best_plddt = plddt
                    best_seed = seed
                    best_cif = cif

            logger.info("  Best seed=%d  pLDDT=%.2f", best_seed, best_plddt)

            tm_score, rmsd = _maybe_tmscore(args, prot_name, best_cif)

            results.append({
                "name": prot_name, "seq_len": len(query_seq),
                "plddt": best_plddt, "best_seed": best_seed,
                "tm_score": tm_score, "rmsd": rmsd,
                "status": "ok" if not np.isnan(best_plddt) else "failed",
            })

        # ── Few-shot: augment shallow MSA, fold once ───────────────────────────
        elif mode == "fewshot":
            # Load shallow MSA for this target
            shallow_seqs = None
            if args.shallow_msa_dir:
                shallow_a3m = Path(args.shallow_msa_dir) / f"{prot_name}.a3m"
                if shallow_a3m.exists():
                    shallow_seqs = parse_a3m_seqs(str(shallow_a3m))
                    logger.info("  Loaded shallow MSA: %d seqs from %s",
                                len(shallow_seqs), shallow_a3m)
                else:
                    logger.warning("  Shallow A3M not found: %s; using depth-1 (query only)",
                                   shallow_a3m)

            if not shallow_seqs:
                # Depth-1 MSA = query sequence only (ungapped, aligned to itself)
                shallow_seqs = [query_seq]
                logger.info("  Using depth-1 shallow MSA (query only)")

            prot_msa_dir = msa_dir / prot_name
            prot_msa_dir.mkdir(exist_ok=True)
            aug_a3m = prot_msa_dir / "augmented.a3m"

            if not aug_a3m.exists() or args.regenerate:
                try:
                    aug_seqs = augment_shallow(
                        shallow_seqs=shallow_seqs,
                        decoder=decoder,
                        latent_fm=latent_fm,
                        msa_module=protenix_model,   # None → Syn track only
                        esm_model=esm_model,
                        alphabet=alphabet,
                        n_syn_seeds=args.n_seeds,
                        n_seqs_per_seed=args.n_seqs,
                        n_rec_seqs=64 if protenix_model else 0,
                        n_diverse=16,
                        n_steps=args.n_steps,
                        temperature=args.temperature,
                        device=device,
                    )
                    write_a3m(query_seq, aug_seqs, str(aug_a3m), prefix=prot_name)
                    logger.info("  Wrote %d augmented seqs to %s", len(aug_seqs), aug_a3m)
                except Exception as exc:
                    logger.error("  Augmentation failed: %s", exc, exc_info=True)
                    results.append({
                        "name": prot_name, "seq_len": len(query_seq),
                        "plddt": float("nan"), "best_seed": -1,
                        "tm_score": float("nan"), "rmsd": float("nan"),
                        "status": f"aug_error:{exc}",
                    })
                    continue

            plddt, cif = fold_once(
                prot_name=prot_name,
                query_seq=query_seq,
                a3m_path=aug_a3m,
                fold_dir=fold_dir,
                args=args,
                use_msa=True,
                tag="_fewshot",
            )
            logger.info("  pLDDT=%.2f", plddt)

            tm_score, rmsd = _maybe_tmscore(args, prot_name, cif)

            results.append({
                "name": prot_name, "seq_len": len(query_seq),
                "plddt": plddt, "best_seed": -1,
                "tm_score": tm_score, "rmsd": rmsd,
                "status": "ok" if not np.isnan(plddt) else "failed",
            })

        # ── Shallow baseline (orphan proteins): fold with real sparse MSA as-is ──
        elif mode == "shallow":
            shallow_seqs = None
            if args.shallow_msa_dir:
                shallow_a3m = Path(args.shallow_msa_dir) / f"{prot_name}.a3m"
                if shallow_a3m.exists():
                    shallow_seqs = parse_a3m_seqs(str(shallow_a3m))
                    logger.info("  Loaded shallow MSA: %d seqs from %s",
                                len(shallow_seqs), shallow_a3m)
                else:
                    logger.warning("  Shallow A3M not found: %s; falling back to depth-1",
                                   shallow_a3m)

            if not shallow_seqs:
                shallow_seqs = [query_seq]
                logger.info("  Using depth-1 shallow MSA (query only)")

            # Write the original sparse MSA into the output tree (for reproducibility)
            prot_msa_dir = msa_dir / prot_name
            prot_msa_dir.mkdir(exist_ok=True)
            sparse_a3m = prot_msa_dir / "sparse.a3m"
            if not sparse_a3m.exists() or args.regenerate:
                with open(sparse_a3m, "w") as fh:
                    fh.write(f">query\n{query_seq}\n")
                    for i, s in enumerate(shallow_seqs[1:], 1):
                        fh.write(f">hit_{i}\n{s}\n")

            plddt, cif = fold_once(
                prot_name=prot_name,
                query_seq=query_seq,
                a3m_path=sparse_a3m,
                fold_dir=fold_dir,
                args=args,
                use_msa=True,
                tag="_shallow",
            )
            logger.info("  pLDDT=%.2f  n_msa=%d", plddt, len(shallow_seqs))

            tm_score, rmsd = _maybe_tmscore(args, prot_name, cif)

            results.append({
                "name": prot_name, "seq_len": len(query_seq),
                "n_msa": len(shallow_seqs),
                "plddt": plddt, "best_seed": -1,
                "tm_score": tm_score, "rmsd": rmsd,
                "status": "ok" if not np.isnan(plddt) else "failed",
            })

        # ── No-MSA baseline ────────────────────────────────────────────────────
        elif mode == "nomsa":
            # tag 를 붙여 folds/{target}_nomsa 에 저장 → ost_score 의 {target}_* glob 이
            # 잡을 수 있게 함(다른 모드와 동일). protenix 는 --use_msa False + MSA 경로 없음.
            plddt, cif = fold_once(
                prot_name=prot_name,
                query_seq=query_seq,
                a3m_path=None,
                fold_dir=fold_dir,
                args=args,
                use_msa=False,
                tag="_nomsa",
            )
            logger.info("  pLDDT=%.2f", plddt)

            tm_score, rmsd = _maybe_tmscore(args, prot_name, cif)

            results.append({
                "name": prot_name, "seq_len": len(query_seq),
                "plddt": plddt, "best_seed": -1,
                "tm_score": tm_score, "rmsd": rmsd,
                "status": "ok" if not np.isnan(plddt) else "failed",
            })

    # ── Write CSV summary ──────────────────────────────────────────────────────
    shard_suffix = f"_shard{args.shard_id}" if args.num_shards > 1 else ""
    csv_path = output_dir / f"shard{shard_suffix}.csv"
    fieldnames = ["name", "seq_len", "n_msa", "plddt", "best_seed", "tm_score", "rmsd", "status"]
    # backfill n_msa for modes that don't set it
    for r in results:
        r.setdefault("n_msa", float("nan"))
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    logger.info("Wrote results to %s", csv_path)

    # ── Aggregate statistics ───────────────────────────────────────────────────
    ok = [r for r in results if r["status"] == "ok"]
    plddts = [r["plddt"] for r in ok if not np.isnan(r["plddt"])]
    tms = [r["tm_score"] for r in ok if not np.isnan(r["tm_score"])]

    print("\n" + "=" * 60)
    print(f"Fold Benchmark Results  mode={mode}  ({len(ok)}/{len(results)} succeeded)")
    print("=" * 60)
    if plddts:
        print(f"  pLDDT  mean={np.mean(plddts):.2f}  "
              f"median={np.median(plddts):.2f}  "
              f"min={np.min(plddts):.2f}  max={np.max(plddts):.2f}")
    if tms:
        print(f"  TM-score  mean={np.mean(tms):.4f}  "
              f"median={np.median(tms):.4f}  "
              f"min={np.min(tms):.4f}  max={np.max(tms):.4f}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Config-driven entrypoint helpers
#
# The benchmark is controlled by a single YAML file (see msaflow/configs/
# fold_benchmark.yaml). Everything below turns that YAML (plus optional CLI
# overrides) into the exact `args` Namespace that run_benchmark() consumes, so
# the folding/scoring skeleton above is untouched — only the settings move.
# ─────────────────────────────────────────────────────────────────────────────

def _read_a3m_query(path: str) -> str:
    """First a3m record = query. Strip insertions (lowercase)/gaps → canonical seq."""
    seq = []
    with open(path) as fh:
        first = fh.readline()
        if not first.startswith(">"):
            raise ValueError(f"a3m 형식 아님(첫 줄 '>' 아님): {path}")
        for line in fh:
            if line.startswith(">"):
                break
            seq.append(line.strip())
    q = "".join(seq).replace("-", "").replace(".", "").upper()
    if not q:
        raise ValueError(f"a3m query 서열 비어있음: {path}")
    return q


def resolve_target_names(targets_list, targets_csv) -> list[str]:
    """YAML targets.list 또는 targets.csv → ['pdb_id__chain', ...] (순서보존 dedup).

    CSV 는 헤더(pdb_id,chain_id) 있는 monomer_protein.csv 형식.
    """
    names: list[str] = []
    if targets_list:
        names = [str(t).strip() for t in targets_list if str(t).strip()]
    elif targets_csv:
        with open(targets_csv) as fh:
            reader = csv.reader(fh)
            next(reader, None)  # skip header (pdb_id,chain_id)
            for row in reader:
                if len(row) < 2 or not row[0].strip():
                    continue
                names.append(f"{row[0].strip()}__{row[1].strip()}")
    seen, out = set(), []
    for n in names:
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def build_subset_fasta(target_names, shallow_msa_dir, dst) -> list[str]:
    """각 타겟 a3m 첫 record(=query)로 subset FASTA 생성.

    입력 a3m 이 없는 타겟은 abort 하지 않고 skip(경고). 실제로 기록된 타겟명만 반환하므로
    이후 폴딩·OST 채점이 동일한 부분집합에 대해 일관되게 돈다.
    """
    if not shallow_msa_dir:
        raise SystemExit(
            "ERROR: targets 로부터 FASTA 를 만들려면 shallow_msa_dir 이 필요합니다 "
            "(또는 --fasta / targets.fasta 로 준비된 FASTA 를 직접 주세요)."
        )
    written: list[str] = []
    with open(dst, "w") as out:
        for t in target_names:
            a3m = Path(shallow_msa_dir) / f"{t}.a3m"
            if not a3m.exists():
                logger.warning("  [skip] 입력 a3m 없음: %s", a3m)
                continue
            try:
                q = _read_a3m_query(str(a3m))
            except ValueError as exc:
                logger.warning("  [skip] %s", exc)
                continue
            out.write(f">{t}\n{q}\n")
            written.append(t)
            logger.info("  %s: query L=%d (from a3m)", t, len(q))
    if not written:
        raise SystemExit(f"ERROR: 유효한 입력 a3m 이 하나도 없습니다 (dir={shallow_msa_dir})")
    logger.info("Subset FASTA: %d/%d targets → %s", len(written), len(target_names), dst)
    return written


def link_protenix_checkpoint(repo_dir, protenix_root, model_name) -> None:
    """Protenix 가 기대하는 $PROTENIX_ROOT_DIR/checkpoint/<model>.pt 심링크(재다운로드 방지).

    실제 가중치는 $REPO_DIR/checkpoints/checkpoints/<model>.pt.
    """
    expected = Path(protenix_root) / "checkpoint" / f"{model_name}.pt"
    actual = Path(repo_dir) / "checkpoints" / "checkpoints" / f"{model_name}.pt"
    if not expected.exists() and actual.exists():
        expected.parent.mkdir(parents=True, exist_ok=True)
        expected.symlink_to(actual)
        logger.info("Linked Protenix checkpoint: %s -> %s", expected, actual)


def run_ost_scoring(output_dir, mode, target_names, gt_dir, chain_csv, out_csv,
                    score_full_complex=False) -> None:
    """폴딩 후 예측 CIF vs GT 를 scripts/ost_score.py 로 채점."""
    repo_dir = Path(__file__).parents[2]
    if not Path(gt_dir).is_dir():
        logger.warning("GT_DIR 없음: %s — OST 채점 건너뜀", gt_dir)
        return
    cmd = [
        sys.executable, str(repo_dir / "scripts" / "ost_score.py"),
        "--pred_root", str(Path(output_dir) / "folds"),
        "--gt_dir",    str(gt_dir),
        "--mode",      mode,
        "--targets",   " ".join(target_names),
        "--out_csv",   str(out_csv),
    ]
    if chain_csv:
        cmd += ["--chain_csv", str(chain_csv)]
    if score_full_complex:
        cmd += ["--score_full_complex"]
    logger.info("=== OST 채점 (GT: %s) → %s", gt_dir, out_csv)
    subprocess.run(cmd, check=False)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MSAFlow fold benchmark (YAML config + optional CLI overrides)")
    parser.add_argument("--config", default=None,
                        help="YAML 설정 파일(OmegaConf). 개별 CLI 플래그가 항상 우선.")
    # 아래 플래그는 기본 None → 값이 있으면 config 를 덮어씀(하위호환 유지).
    parser.add_argument("--fasta",           default=None,
                        help="준비된 FASTA. 없으면 targets(list/csv)+shallow_msa_dir 로 자동 생성")
    parser.add_argument("--targets",         nargs="+", default=None,
                        help="타겟명 직접 지정: 'pdb__A pdb2__A ...' (config targets.list 대체)")
    parser.add_argument("--targets_csv",     default=None,
                        help="타겟 CSV(pdb_id,chain_id). config targets.csv 대체")
    parser.add_argument("--shallow_msa_dir", default=None,
                        help="입력 shallow A3M 디렉토리({target}.a3m). query 서열 추출/증강 입력")
    parser.add_argument("--decoder_ckpt",    default=None, help="SFM decoder EMA checkpoint")
    parser.add_argument("--latent_fm_ckpt",  default=None, help="latent FM EMA checkpoint")
    parser.add_argument("--protenix_ckpt",   default=None,
                        help="[fewshot] Rec 트랙용 Protenix .pt")
    parser.add_argument("--output_dir",      default=None,
                        help="MSA/fold/CSV 출력 루트")
    parser.add_argument("--mode",            default=None,
                        choices=["zeroshot", "fewshot", "shallow", "nomsa"],
                        help="zeroshot | fewshot | shallow | nomsa")
    parser.add_argument("--device",          default=None)
    parser.add_argument("--gpu_id",          type=int, default=None,
                        help="CUDA_VISIBLE_DEVICES 로 노출할 GPU (기본 config run.gpu_id)")
    parser.add_argument("--ref_pdb_dir",     default=None)
    parser.add_argument("--n_seqs",          type=int, default=None)
    parser.add_argument("--n_seeds",         type=int, default=None)
    parser.add_argument("--n_steps",         type=int, default=None)
    parser.add_argument("--temperature",     type=float, default=None)
    parser.add_argument("--protenix_model",  default=None)
    parser.add_argument("--fold_seeds",      default=None,
                        help="쉼표구분 폴딩 seed 직접지정. 없으면 fold_seed_base+n_fold_seeds 로 계산")
    parser.add_argument("--fold_seed_base",  type=int, default=None)
    parser.add_argument("--n_fold_seeds",    type=int, default=None)
    parser.add_argument("--n_sample",        type=int, default=None)
    parser.add_argument("--tmscore_bin",     default=None)
    parser.add_argument("--shard_id",        type=int, default=None)
    parser.add_argument("--num_shards",      type=int, default=None)
    parser.add_argument("--regenerate",      action="store_true", default=False,
                        help="A3M 이 있어도 강제 재생성(기본 fresh 라 이미 재생성됨; resume 시 강제용)")
    parser.add_argument("--refold",          action="store_true", default=False,
                        help="CIF 이 있어도 강제 재폴딩(기본 fresh 라 이미 재폴딩됨; resume 시 강제용)")
    parser.add_argument("--resume",          action="store_true", default=False,
                        help="이어하기: 이전 a3m/CIF 가 있으면 재사용하고 없는 것만 수행 "
                             "(기본은 fresh — 이전 출력이 있어도 증강·폴딩을 다시 수행)")
    # OST 채점
    parser.add_argument("--skip_ost",        action="store_true", default=False,
                        help="config 가 켜져 있어도 OST 채점을 건너뜀")
    parser.add_argument("--score_only",      action="store_true", default=False,
                        help="폴딩을 건너뛰고 이미 생성된 folds/ 의 CIF 만 OST 채점 "
                             "(모델 로딩/GPU 불필요; ost.enable 무시하고 무조건 채점)")
    parser.add_argument("--gt_dir",          default=None)
    parser.add_argument("--score_full_complex", action="store_true", default=False,
                        help="OST 채점 시 '-c model:ref' chain 매핑 없이 GT 어셈블리 "
                             "전체와 비교 (chain 매핑 자동 추정)")
    parser.add_argument("--chain_csv",       default=None)
    parser.add_argument("--ost_csv",         default=None)
    # Deprecated alias
    parser.add_argument("--no_msa",          action="store_true", default=False,
                        help="Deprecated: --mode nomsa 를 쓰세요")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  [ %(name)s ]  %(message)s",
        datefmt="%Y-%m-%d %H:%M",
    )

    cfg = OmegaConf.load(args.config) if args.config else OmegaConf.create({})

    def pick(cli_val, path, default):
        """CLI 값(≠None) 우선, 없으면 config, 없으면 default."""
        if cli_val is not None:
            return cli_val
        val = OmegaConf.select(cfg, path) if path else None
        return val if val is not None else default

    # ── 기본 실행 설정 ─────────────────────────────────────────────────────────
    mode = "nomsa" if args.no_msa else pick(args.mode, "run.mode", "zeroshot")
    base_output_dir = pick(args.output_dir, "run.output_dir", None)
    if not base_output_dir:
        parser.error("output_dir 없음 (--output_dir 또는 config run.output_dir 지정).")
    # 모드별 네임스페이스: msas/folds/csv/OST 전부 {output_dir}/{mode}/ 아래로 보내
    # 여러 모드 결과가 한 폴더에 섞이지 않게 한다. 이 아래로는 output_dir = 모드별 경로.
    output_dir = str(Path(base_output_dir) / mode)
    logger.info("출력 디렉토리(모드별): %s", output_dir)
    device = pick(args.device, "run.device", "cuda")
    gpu_id = pick(args.gpu_id, "run.gpu_id", 0)
    # SLURM(--gres)/사용자가 이미 CUDA_VISIBLE_DEVICES 를 잡았으면 그쪽을 존중한다
    # (스케줄러 할당을 config 절대인덱스로 덮어쓰면 할당 밖 GPU 를 가리킬 수 있음).
    # 다만 조용히 무시되지 않도록 명시적으로 로그를 남긴다.
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        logger.info("CUDA_VISIBLE_DEVICES=%s 가 이미 설정됨(SLURM 등) — gpu_id=%s 는 무시",
                    os.environ["CUDA_VISIBLE_DEVICES"], gpu_id)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # 기본은 fresh run: 이전 출력이 있어도 증강·폴딩을 다시 수행(덮어씀). 그래서 재실행이
    # OST 만 도는 일이 없다. 큰 배치를 이어서 돌릴 때만 resume=true 로 기존 결과를 재사용.
    resume = bool(args.resume or (OmegaConf.select(cfg, "run.resume") or False))
    force_regen = bool(args.regenerate or (OmegaConf.select(cfg, "run.regenerate") or False))
    force_refold = bool(args.refold or (OmegaConf.select(cfg, "run.refold") or False))
    regenerate = force_regen or (not resume)
    refold = force_refold or (not resume)
    logger.info("실행 모드: %s (regenerate=%s, refold=%s)",
                "resume(이어하기)" if resume else "fresh(재실행)", regenerate, refold)

    # ── 폴딩 seed 문자열 계산 (pick 과 동일한 is-not-None 규율) ────────────────
    fold_seed_base = int(pick(args.fold_seed_base, "fold.fold_seed_base", 101))
    n_fold_seeds = int(pick(args.n_fold_seeds, "fold.n_fold_seeds", 5))
    fold_seeds = pick(args.fold_seeds, "fold.fold_seeds", None) \
        or ",".join(str(fold_seed_base + i) for i in range(n_fold_seeds))

    protenix_model = pick(args.protenix_model, "model.protenix_model",
                          "protenix_base_default_v1.0.0")
    shallow_msa_dir = pick(args.shallow_msa_dir, "targets.shallow_msa_dir", None)

    # ── 타겟 → FASTA ──────────────────────────────────────────────────────────
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fasta = pick(args.fasta, "targets.fasta", None)
    if fasta:
        target_names = [n for n, _ in parse_fasta(fasta)]
        logger.info("준비된 FASTA 사용: %s (%d targets)", fasta, len(target_names))
    else:
        targets_list = args.targets if args.targets else OmegaConf.select(cfg, "targets.list")
        targets_csv = pick(args.targets_csv, "targets.csv", None)
        names = resolve_target_names(list(targets_list) if targets_list else None, targets_csv)
        if not names:
            parser.error("타겟이 없습니다 (--fasta / --targets / targets.list / targets.csv).")
        logger.info("타겟 %d개 (list/csv). shallow_msa_dir=%s", len(names), shallow_msa_dir)
        fasta = str(Path(output_dir) / "targets_subset.fasta")
        target_names = build_subset_fasta(names, shallow_msa_dir, fasta)

    # ── Protenix checkpoint 심링크 ─────────────────────────────────────────────
    repo_dir = Path(__file__).parents[2]
    protenix_root = os.environ.get("PROTENIX_ROOT_DIR", str(repo_dir))
    link_protenix_checkpoint(repo_dir, protenix_root, protenix_model)

    # ── run_benchmark 가 소비하는 Namespace 구성 ──────────────────────────────
    resolved = Namespace(
        fasta=fasta,
        decoder_ckpt=pick(args.decoder_ckpt, "model.decoder_ckpt", "dummy"),
        latent_fm_ckpt=pick(args.latent_fm_ckpt, "model.latent_fm_ckpt", "dummy"),
        output_dir=output_dir,
        mode=mode,
        ref_pdb_dir=pick(args.ref_pdb_dir, "fold.ref_pdb_dir", None),
        device=device,
        n_seqs=int(pick(args.n_seqs, "augment.n_seqs", 32)),
        n_seeds=int(pick(args.n_seeds, "augment.n_seeds", 10)),
        n_steps=int(pick(args.n_steps, "augment.n_steps", 100)),
        temperature=float(pick(args.temperature, "augment.temperature", 0.5)),
        protenix_model=protenix_model,
        fold_seeds=fold_seeds,
        n_sample=int(pick(args.n_sample, "fold.n_sample", 5)),
        tmscore_bin=pick(args.tmscore_bin, "fold.tmscore_bin", "TMscore"),
        shallow_msa_dir=shallow_msa_dir,
        protenix_ckpt=pick(args.protenix_ckpt, "model.protenix_ckpt", None),
        shard_id=int(pick(args.shard_id, "run.shard_id", 0)),
        num_shards=int(pick(args.num_shards, "run.num_shards", 1)),
        regenerate=regenerate,
        refold=refold,
        no_msa=False,  # 이미 mode 로 반영됨
    )

    if args.score_only:
        logger.info("score_only=True — 폴딩 건너뜀. 기존 folds/ CIF 만 OST 채점합니다.")
    else:
        run_benchmark(resolved)

    # ── OST 채점 ────────────────────────────────────────────────────────────────
    # score_only 면 ost.enable 과 무관하게(단, --skip_ost 는 존중) 무조건 채점.
    run_ost = (not args.skip_ost) and (args.score_only or bool(OmegaConf.select(cfg, "ost.enable")))
    if run_ost:
        gt_dir = pick(args.gt_dir, "ost.gt_dir", None)
        chain_csv = pick(args.chain_csv, "ost.chain_csv", None)
        ost_csv = pick(args.ost_csv, "ost.out_csv", None) \
            or str(Path(output_dir) / "monomer_protein_ost.csv")
        if not gt_dir:
            logger.warning("ost.enable=true 인데 gt_dir 미지정 — OST 건너뜀")
        else:
            # 샤딩 시 예측이 부분집합만 존재하므로, 이 샤드가 실제 처리한 타겟만 채점
            if resolved.num_shards > 1:
                scored = target_names[resolved.shard_id::resolved.num_shards]
            else:
                scored = target_names
            run_ost_scoring(output_dir, mode, scored, gt_dir, chain_csv, ost_csv,
                            score_full_complex=args.score_full_complex)


if __name__ == "__main__":
    main()
