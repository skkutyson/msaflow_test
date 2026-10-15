#!/usr/bin/env python3
"""Compare real vs. zero-shot MSA embeddings with PCA / t-SNE.

For each target, two MSA embeddings of the SAME query, both (L, 128):

  real      — AF3/Protenix MSA module output from the real deep MSA
              (preprocessing.extract_msa_embedding_protenix).  ONE embedding.
  latent_fm — generated purely from the query's ESM2 representation by the
              latent FM model (sample_msa_embeddings), the "latent decoder"
              zero-shot track with NO real MSA as input.  FOUR seeds.

Output layout under --out_base (default /store/yjlee4/runs/compare_emb):

    <out_base>/<target>/real/<target>.npz        real:     (L, 128)
    <out_base>/<target>/latent_fm/<target>.npz   zeroshot: (K=4, L, 128)

Commands:
  extract    GPU: load models once, dump real + zero-shot npz for N targets.
  plot       CPU: PCA / t-SNE scatter for ONE target.
  plot-all   CPU: one grid over ALL targets; the most-similar target is ★ BEST.

Usage
-----
  # 1) GPU node (inside sandbox): dump embeddings for some targets
  uv run python scripts/compare_msa_emb.py extract --targets 7tlh 8abc

  # 2) CPU: per-target figure
  uv run python scripts/compare_msa_emb.py plot --target 7tlh

  # 3) CPU: all targets in one grid, best starred
  uv run python scripts/compare_msa_emb.py plot-all
"""
import argparse
import csv
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("compare_msa_emb")

# Defaults wired to this project's layout.
DEF_A3M_DIR = "/store/deepfold3/results/msa/a3m"
DEF_CSV = "/home/yjlee4/MSA_FLOW/monomer_protein.csv"
DEF_OUT_BASE = "/store/yjlee4/runs/compare_emb"
DEF_LATENT_FM = "/home/yjlee4/MSA_FLOW/runs/latent_fm/latest.pt"
DEF_PROTENIX = "/home/yjlee4/MSA_FLOW/checkpoints/checkpoints/protenix_base_default_v1.0.0.pt"


# ──────────────────────────────────────────────────────────────────────────────
# Path helpers
# ──────────────────────────────────────────────────────────────────────────────

def _real_npz(out_base, target):
    return Path(out_base) / target / "real" / f"{target}.npz"


def _lfm_npz(out_base, target):
    return Path(out_base) / target / "latent_fm" / f"{target}.npz"


def _discover_targets(a3m_dir):
    return sorted(p.stem for p in Path(a3m_dir).glob("*.a3m"))


def _targets_from_csv(csv_path):
    """monomer_protein.csv (pdb_id,chain_id) → ['<pdb_id>__<chain_id>', ...]."""
    targets = []
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            pdb, chain = row["pdb_id"].strip(), row["chain_id"].strip()
            if pdb and chain:
                targets.append(f"{pdb}__{chain}")
    return targets


def _load_pair(out_base, target):
    """Return (real (L,128), zeroshot (K,L,128)) cropped to a common L."""
    real = np.load(_real_npz(out_base, target))["real"]
    zs = np.load(_lfm_npz(out_base, target))["zeroshot"]
    L = min(real.shape[0], zs.shape[1])
    return real[:L], zs[:, :L]


# ──────────────────────────────────────────────────────────────────────────────
# Similarity metric (also drives the ★ best pick)
# ──────────────────────────────────────────────────────────────────────────────

def per_residue_cos(real, zeroshot):
    """Mean per-residue cosine between real and the mean zero-shot seed."""
    zmean = zeroshot.mean(axis=0)                                  # (L, 128)
    num = (real * zmean).sum(axis=1)
    den = np.linalg.norm(real, axis=1) * np.linalg.norm(zmean, axis=1) + 1e-8
    return float((num / den).mean())


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: extraction (GPU) — two passes to keep peak VRAM low
# ──────────────────────────────────────────────────────────────────────────────

def extract(args):
    import torch
    from msaflow.data.preprocessing import (
        parse_a3m, filter_msa, extract_msa_embedding_protenix, extract_esm_embedding,
    )
    from msaflow.models.latent_fm import sample_msa_embeddings
    from msaflow.inference.generate import load_latent_fm, load_esm2, load_msa_module

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Target source priority: explicit --targets > --csv > all *.a3m in a3m_dir.
    if args.targets:
        targets = args.targets
    elif args.csv:
        targets = _targets_from_csv(args.csv)
        logger.info("%d target(s) from csv %s", len(targets), args.csv)
    else:
        targets = _discover_targets(args.a3m_dir)
    if not targets:
        raise SystemExit("no targets (check --targets / --csv / a3m_dir)")

    # Parse a3m up front; targets whose a3m is missing are skipped (not fatal).
    parsed, missing = {}, []
    for t in targets:
        a3m = Path(args.a3m_dir) / f"{t}.a3m"
        if not a3m.is_file():
            missing.append(t)
            continue
        _, seqs = parse_a3m(str(a3m))
        seqs = filter_msa(seqs)
        parsed[t] = seqs
        logger.info("  %s: depth=%d L=%d", t, len(seqs), len(seqs[0].replace("-", "")))
    if missing:
        logger.warning("skipped %d target(s) with no a3m in %s: %s",
                       len(missing), args.a3m_dir, ", ".join(missing))
    targets = [t for t in targets if t in parsed]
    if not targets:
        raise SystemExit(f"no target a3m found under {args.a3m_dir}")
    logger.info("extracting %d target(s)", len(targets))

    # ── Pass A — real (Protenix trunk) ───────────────────────────────────────
    protenix = load_msa_module(args.protenix_ckpt, device)
    for t in targets:
        real = extract_msa_embedding_protenix(
            parsed[t], protenix, device, n_cycle=args.n_cycle
        ).float().cpu().numpy()
        out = _real_npz(args.out_base, t)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, real=real, query=parsed[t][0].replace("-", ""),
                            n_cycle=args.n_cycle)
        logger.info("  [real] %s %s → %s", t, real.shape, out)
    del protenix
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── Pass B — zero-shot (ESM2 → latent FM), K seeds ───────────────────────
    esm_model, alphabet = load_esm2(device)
    latent_fm = load_latent_fm(args.latent_fm_ckpt, device)
    for t in targets:
        query = parsed[t][0].replace("-", "")
        esm_emb = extract_esm_embedding(query, esm_model, alphabet, device)
        esm_emb = esm_emb.unsqueeze(0).to(device)                  # (1, L, 1280)
        seeds = []
        for s in range(args.n_seeds):
            torch.manual_seed(s)
            z = sample_msa_embeddings(latent_fm, esm_emb,
                                      n_steps=args.n_steps, temperature=args.temperature)
            seeds.append(z[0].float().cpu().numpy())
        zeroshot = np.stack(seeds, axis=0)                         # (K, L, 128)
        out = _lfm_npz(args.out_base, t)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, zeroshot=zeroshot, query=query,
                            temperature=args.temperature, n_steps=args.n_steps)
        logger.info("  [latent_fm] %s %s → %s", t, zeroshot.shape, out)
    logger.info("extract done: %d target(s) under %s", len(targets), args.out_base)


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: plotting (CPU)
# ──────────────────────────────────────────────────────────────────────────────

def _project(X, method, seed=0):
    """Fit a 2-D projection JOINTLY so both clouds share axes."""
    from sklearn.decomposition import PCA
    if method == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(X)
    if method == "tsne":
        from sklearn.manifold import TSNE
        X = PCA(n_components=min(30, X.shape[1]), random_state=seed).fit_transform(X)
        perplexity = min(30, max(5, (X.shape[0] - 1) // 3))
        return TSNE(n_components=2, perplexity=perplexity,
                    init="pca", random_state=seed).fit_transform(X)
    raise ValueError(method)


def _scatter(ax, real, zeroshot, method, seed=0):
    """Joint-projected scatter of one target onto `ax`. Returns cosine sim."""
    K, L, _ = zeroshot.shape
    X = np.concatenate([real, zeroshot.reshape(K * L, -1)], axis=0)
    emb = _project(X, method, seed=seed)
    r2, z2 = emb[:L], emb[L:]
    ax.scatter(z2[:, 0], z2[:, 1], s=8, alpha=0.3, c="tab:orange",
               label=f"latent_fm (K={K})")
    ax.scatter(r2[:, 0], r2[:, 1], s=18, alpha=0.9, c="tab:blue",
               edgecolors="k", linewidths=0.2, label="real")
    return per_residue_cos(real, zeroshot)


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    real, zeroshot = _load_pair(args.out_base, args.target)
    methods = ["pca", "tsne"] if args.method == "both" else [args.method]
    fig, axes = plt.subplots(1, len(methods), figsize=(7 * len(methods), 6), squeeze=False)
    cos = None
    for ax, method in zip(axes[0], methods):
        cos = _scatter(ax, real, zeroshot, method, seed=args.seed)
        ax.set_title(f"{method.upper()} — per-residue MSA embedding")
        ax.set_xlabel(f"{method}-1"); ax.set_ylabel(f"{method}-2")
        ax.legend(loc="best", framealpha=0.9)
    fig.suptitle(f"{args.target}   L={real.shape[0]}   per-res cos={cos:.3f}", fontsize=12)
    fig.tight_layout()
    out = args.out or str(Path(args.out_base) / args.target / f"{args.target}_compare.png")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    logger.info("wrote %s   (per-res cos=%.4f)", out, cos)


def plot_all(args):
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    targets = args.targets or [
        d.name for d in sorted(Path(args.out_base).iterdir())
        if (_real_npz(args.out_base, d.name).exists()
            and _lfm_npz(args.out_base, d.name).exists())
    ]
    if not targets:
        raise SystemExit(f"no extracted targets under {args.out_base}")

    pairs = {t: _load_pair(args.out_base, t) for t in targets}
    sims = {t: per_residue_cos(*pairs[t]) for t in targets}
    best = max(sims, key=sims.get)                                 # most similar
    logger.info("similarities: %s", {t: round(v, 4) for t, v in sorted(
        sims.items(), key=lambda kv: -kv[1])})
    logger.info("★ best = %s (cos=%.4f)", best, sims[best])

    n = len(targets)
    ncol = min(args.ncol, n)
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4.6 * nrow), squeeze=False)
    for i, t in enumerate(targets):
        ax = axes[i // ncol][i % ncol]
        real, zeroshot = pairs[t]
        _scatter(ax, real, zeroshot, args.method, seed=args.seed)
        star = "★ " if t == best else ""
        ax.set_title(f"{star}{t}  cos={sims[t]:.3f}" + ("  (BEST)" if t == best else ""),
                     fontweight="bold" if t == best else "normal")
        ax.set_xticks([]); ax.set_yticks([])
    for j in range(n, nrow * ncol):            # blank unused cells
        axes[j // ncol][j % ncol].axis("off")
    axes[0][0].legend(loc="best", fontsize=8, framealpha=0.9)

    fig.suptitle(f"real vs latent_fm MSA embedding ({args.method.upper()}) — "
                 f"★ best = {best} (cos={sims[best]:.3f})", fontsize=13)
    fig.tight_layout()
    out = args.out or str(Path(args.out_base) / f"compare_all_{args.method}.png")
    fig.savefig(out, dpi=150)
    logger.info("wrote %s", out)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  [ %(name)s ]  %(message)s",
                        datefmt="%Y-%m-%d %H:%M")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_base", default=DEF_OUT_BASE)
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("extract", help="GPU: dump real + zero-shot npz")
    pe.add_argument("--targets", nargs="*", default=None,
                    help="explicit target names (a3m = <a3m_dir>/<target>.a3m); "
                         "overrides --csv")
    pe.add_argument("--csv", default=DEF_CSV,
                    help="monomer_protein.csv (pdb_id,chain_id → pdb_id__chain_id); "
                         "a3m-less targets are skipped. Used when --targets is empty.")
    pe.add_argument("--a3m_dir", default=DEF_A3M_DIR)
    pe.add_argument("--latent_fm_ckpt", default=DEF_LATENT_FM)
    pe.add_argument("--protenix_ckpt", default=DEF_PROTENIX)
    pe.add_argument("--n_seeds", type=int, default=4)
    pe.add_argument("--n_steps", type=int, default=100)
    pe.add_argument("--temperature", type=float, default=0.5)
    pe.add_argument("--n_cycle", type=int, default=3)
    pe.add_argument("--device", default="cuda")
    pe.set_defaults(func=extract)

    pp = sub.add_parser("plot", help="CPU: one target, PCA/t-SNE scatter")
    pp.add_argument("--target", required=True)
    pp.add_argument("--method", choices=["pca", "tsne", "both"], default="both")
    pp.add_argument("--out", default=None)
    pp.add_argument("--seed", type=int, default=0)
    pp.set_defaults(func=plot)

    pa = sub.add_parser("plot-all", help="CPU: all targets in one grid, best starred")
    pa.add_argument("--targets", nargs="*", default=None, help="default: all extracted")
    pa.add_argument("--method", choices=["pca", "tsne"], default="pca")
    pa.add_argument("--ncol", type=int, default=4)
    pa.add_argument("--out", default=None)
    pa.add_argument("--seed", type=int, default=0)
    pa.set_defaults(func=plot_all)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
