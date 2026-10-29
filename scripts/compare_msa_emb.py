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
    <out_base>/<target>/esm/<target>.npz         esm:      (L, 1280)
        ESM2 query embedding, cropped to the same L — fed to regression_baseline.py.

Commands:
  extract    GPU: per-target loop (real → zero-shot → inline plot → free VRAM);
             --resume skips done targets, so a scarce GPU can finish in chunks.
  plot       CPU: PCA / t-SNE scatter for ONE target.
  plot-all   CPU: one grid over ALL targets, shared centred-PCA axes; ★ = BEST
             by cos_best. Reports cos_best / cos_mean / diversity / cos_null
             (see the "Similarity metrics" section for why seed-mean cosine is
             wrong and these four replace it).
  spread     CPU: decompose low diversity into sd_real / sd_gen / sd_seed per
             channel → distinguishes MODE COLLAPSE (training) from a
             NEAR-DETERMINISTIC model (sampling). CSV + per-channel npz + figure.

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


def _esm_npz(out_base, target):
    """ESM2 e (L, 1280), saved by extract (cropped to the same L as real)."""
    return Path(out_base) / target / "esm" / f"{target}.npz"


def _extracted_targets(out_base):
    """Targets under out_base with BOTH real + latent_fm npz (the test set)."""
    return sorted(d.name for d in Path(out_base).iterdir() if d.is_dir()
                  and _real_npz(out_base, d.name).exists()
                  and _lfm_npz(out_base, d.name).exists())


class _ChannelAccum:
    """Running per-channel mean/std from streamed (L,128) arrays → O(N) total.

    Lets the extract inline plot centre on the test-set-so-far without re-reading
    every npz each target (that was O(N²)). Provisional by nature; the canonical
    stats come from _testset_stats over the full set in plot / plot-all.
    """

    def __init__(self, dim=None):
        self.n = 0
        self.s1 = None if dim is None else np.zeros(dim)
        self.s2 = None if dim is None else np.zeros(dim)

    def add(self, x):                                    # x: (L, C)
        x = np.asarray(x)
        if self.s1 is None:
            self.s1 = np.zeros(x.shape[1]); self.s2 = np.zeros(x.shape[1])
        self.n += x.shape[0]
        self.s1 += x.sum(axis=0)
        self.s2 += (x * x).sum(axis=0)

    def stats(self):
        mu = self.s1 / self.n
        sd = np.sqrt(np.maximum(self.s2 / self.n - mu * mu, 0.0)) + 1e-6
        return mu, sd


def _testset_stats(out_base, targets=None):
    """Per-channel MU/SD pooled over the real embeddings of the whole test set.

    The SINGLE centring reference shared by plot / plot-all / the inline extract
    plot, so cos_best for a given target is identical across all three. `targets`
    defaults to every extracted target (real + latent_fm present).
    """
    if targets is None:
        targets = _extracted_targets(out_base)
    reals = [np.load(_real_npz(out_base, t))["real"] for t in targets]
    return channel_stats(reals)


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
# Similarity metrics (also drive the ★ best pick)
#
# Why NOT mean-then-cos: averaging the K seeds first and taking cosine measures
# the FM's empirical conditional mean Ê[z|e] — exactly what a regression head
# optimises — which erases FM's whole point and biases toward regression. The
# paper reports best-of-K instead. So we report four numbers:
#
#   cos_best  — best-of-K per-seed cosine (paper protocol); the ★ picks on this.
#   cos_mean  — mean of per-seed cosines (NOT cos of the seed-mean). High cos_mean
#               with ~0 diversity = FM collapsed to a regressor.
#   diversity — mean pairwise distance among the K seeds (mode-collapse probe).
#   cos_null  — gen vs a DIFFERENT protein's real: the lower-bound baseline that
#               makes the absolute numbers interpretable.
#
# All cosines are computed on CENTRED (+ per-channel z-scored) vectors. The j-axis
# mean-pool leaves a large shared component (relpos, global stats) that saturates
# raw cosine to ~0.9 even for unrelated proteins; centring restores resolution.
# ──────────────────────────────────────────────────────────────────────────────

def channel_stats(reals):
    """Per-channel mean/std over all residues of pooled real embeddings → (128,), (128,).

    `reals` is an iterable of (L_i, 128) real arrays. MU is ideally the TRAINING-set
    mean; absent that artifact we fall back to the test-set mean, per the plan.
    """
    pooled = np.concatenate([np.asarray(r) for r in reals], axis=0)   # (sum L, 128)
    mu = pooled.mean(axis=0)
    sd = pooled.std(axis=0) + 1e-6
    assert mu.shape == (pooled.shape[1],), f"MU must be per-channel 1-D, got {mu.shape}"
    return mu, sd


def _center(x, mu, sd, zscore=True):
    """Centre (and optionally z-score) along the channel axis. Broadcasts over L/K."""
    return (x - mu) / sd if zscore else (x - mu)


def _cos_lastdim(a, b):
    """Cosine over the last (channel) axis, broadcasting the rest."""
    num = (a * b).sum(axis=-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-8
    return num / den


def per_seed_cos(real, zeroshot, mu, sd, zscore=True):
    """Per-seed, per-residue cosine on centred vectors → (K,) one value per seed."""
    zc = _center(real, mu, sd, zscore)                  # (L, 128)
    gc = _center(zeroshot, mu, sd, zscore)              # (K, L, 128)
    return _cos_lastdim(gc, zc[None]).mean(axis=-1)     # (K,)


def seed_diversity(zeroshot, mu, sd, zscore=True):
    """Per-dimension RMS pairwise distance among the K centred seeds.

    = mean_{i<j} ‖g_i − g_j‖₂ / √(L·128). Dividing by √D makes it L-independent
    (a raw L2 over the flattened (L·128) vector grows with length), so one
    collapse threshold is comparable across short and long proteins. ≈0 ⇒ seeds
    identical (FM degenerated to a deterministic map).
    """
    gc = _center(zeroshot, mu, sd, zscore).reshape(zeroshot.shape[0], -1)  # (K, L*128)
    K, D = gc.shape
    if K < 2:
        return 0.0
    rootD = np.sqrt(D)
    d, n = 0.0, 0
    for i in range(K):
        for j in range(i + 1, K):
            d += float(np.linalg.norm(gc[i] - gc[j]) / rootD)
            n += 1
    return d / n


def null_cos(zeroshot, real_other, mu, sd, zscore=True):
    """Lower-bound: centred gen seeds vs a DIFFERENT protein's centred real."""
    L = min(zeroshot.shape[1], real_other.shape[0])
    gc = _center(zeroshot[:, :L], mu, sd, zscore)       # (K, L, 128)
    oc = _center(real_other[:L], mu, sd, zscore)        # (L, 128)
    return float(_cos_lastdim(gc, oc[None]).mean())


def compute_metrics(real, zeroshot, mu, sd, real_other=None, zscore=True):
    """Return the four-number report dict for one target."""
    cos_k = per_seed_cos(real, zeroshot, mu, sd, zscore)
    return {
        "cos_best": float(cos_k.max()),
        "cos_mean": float(cos_k.mean()),
        "diversity": seed_diversity(zeroshot, mu, sd, zscore),
        "cos_null": None if real_other is None
        else null_cos(zeroshot, real_other, mu, sd, zscore),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: extraction (GPU)
# Per-target loop: real → zero-shot → (inline plot) → free VRAM → next.
# Models load once (weights ~4 GB); the VRAM hog is per-target activations, so
# we empty_cache() after every target. --resume skips already-done targets so a
# scarce / preemptible GPU can finish the set in several short runs.
# ──────────────────────────────────────────────────────────────────────────────

def extract(args):
    import torch
    from msaflow.data.preprocessing import (
        parse_a3m, filter_msa, extract_msa_embedding_protenix, extract_esm_embedding,
    )
    from msaflow.models.latent_fm import sample_msa_embeddings
    from msaflow.inference.generate import load_latent_fm, load_esm2, load_msa_module

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    def empty_cache():
        if device.type == "cuda":
            torch.cuda.empty_cache()

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

    # --resume: drop targets whose real + latent_fm npz already exist.
    if args.resume:
        done = [t for t in targets
                if _real_npz(args.out_base, t).exists()
                and _lfm_npz(args.out_base, t).exists()]
        if done:
            logger.info("resume: %d/%d already done, skipping", len(done), len(targets))
        targets = [t for t in targets if t not in set(done)]

    # Parse a3m; targets whose a3m is missing are skipped (not fatal).
    parsed, missing = {}, []
    for t in targets:
        a3m = Path(args.a3m_dir) / f"{t}.a3m"
        if not a3m.is_file():
            missing.append(t)
            continue
        _, seqs = parse_a3m(str(a3m))
        parsed[t] = filter_msa(seqs)
    if missing:
        logger.warning("skipped %d target(s) with no a3m in %s: %s",
                       len(missing), args.a3m_dir, ", ".join(missing))
    targets = [t for t in targets if t in parsed]
    if not targets:
        logger.info("nothing to do (all done / no a3m)")
        return
    logger.info("extracting %d target(s)", len(targets))

    # Load all models once — reloading per target would cost minutes × N.
    protenix = load_msa_module(args.protenix_ckpt, device)
    esm_model, alphabet = load_esm2(device)
    latent_fm = load_latent_fm(args.latent_fm_ckpt, device)

    # Incremental test-set channel stats for the (provisional) inline plot — seeded
    # from targets already on disk, then grown as we extract (O(N), not O(N²)).
    accum = None
    if args.plot:
        accum = _ChannelAccum()
        for prev in set(_extracted_targets(args.out_base)) - set(targets):
            accum.add(np.load(_real_npz(args.out_base, prev))["real"])

    for i, t in enumerate(targets, 1):
        seqs = parsed[t]
        query = seqs[0].replace("-", "")
        logger.info("[%d/%d] %s  depth=%d L=%d", i, len(targets), t, len(seqs), len(query))
        try:
            # real: AF3/Protenix MSA module (1 embedding)
            real = extract_msa_embedding_protenix(
                seqs, protenix, device, n_cycle=args.n_cycle
            ).float().cpu().numpy()
            empty_cache()

            # zero-shot: ESM2 → latent FM, K seeds.  Keep the (L,1280) ESM on CPU
            # so we can save it here (cropped to the same L as real) — this is what
            # the regression baseline consumes, and saving it at source guarantees
            # it is prefix-aligned with real/zeroshot (no re-crop drift later).
            esm_np = extract_esm_embedding(query, esm_model, alphabet, device).float().cpu().numpy()
            esm_emb = torch.from_numpy(esm_np).unsqueeze(0).to(device)   # (1, L, 1280)
            seeds = []
            for s in range(args.n_seeds):
                torch.manual_seed(s)
                z = sample_msa_embeddings(latent_fm, esm_emb,
                                          n_steps=args.n_steps, temperature=args.temperature)
                seeds.append(z[0].float().cpu().numpy())
            zeroshot = np.stack(seeds, axis=0)                     # (K, L, 128)
            del esm_emb
            empty_cache()
        except torch.cuda.OutOfMemoryError:
            logger.error("[%d/%d] %s OOM — skipping (rerun with more VRAM)", i, len(targets), t)
            empty_cache()
            continue

        # Crop everything to the common L, then save real / zeroshot / esm npz.
        Ls = {"real": real.shape[0], "gen": zeroshot.shape[1], "esm": esm_np.shape[0]}
        if len(set(Ls.values())) > 1:
            logger.error("[%d/%d] %s length mismatch %s — cropping to min; "
                         "check prefix alignment", i, len(targets), t, Ls)
        L = min(Ls.values())
        real, zeroshot, esm_np = real[:L], zeroshot[:, :L], esm_np[:L]
        for npz, kw in ((_real_npz(args.out_base, t), dict(real=real, n_cycle=args.n_cycle)),
                        (_lfm_npz(args.out_base, t),
                         dict(zeroshot=zeroshot, temperature=args.temperature, n_steps=args.n_steps)),
                        (_esm_npz(args.out_base, t), dict(esm=esm_np))):
            npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(npz, query=query, **kw)
        logger.info("  saved real%s + zeroshot%s + esm%s", real.shape, zeroshot.shape, esm_np.shape)

        # Inline per-target plot so results land as we go (no GPU needed here).
        # Centre by the test-set MU of targets extracted SO FAR → the cosine is
        # PROVISIONAL (the set is incomplete); the canonical value comes from
        # `plot --target <t>` / `plot-all`, which centre on the full set.
        if args.plot:
            png = Path(args.out_base) / t / f"{t}_compare.png"
            accum.add(real)
            mu, sd = accum.stats()
            cos = _render_target(t, real, zeroshot, mu, sd, str(png),
                                 args.plot_method, seed=0, provisional=True)
            logger.info("  plotted %s  (provisional cos_best=%.4f)", png, cos)

    logger.info("extract done under %s", args.out_base)


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: plotting (CPU)
# ──────────────────────────────────────────────────────────────────────────────

def _fit_projector(X, method, seed=0):
    """Fit a 2-D projection on centred X (so axes are shared across whatever is passed).

    PCA returns a reusable transformer (so a grid can share ONE common axis fit on
    the pooled real manifold). t-SNE has no out-of-sample transform, so it only
    fits/transforms the exact X given — qualitative use only (paper Fig. 6).
    """
    from sklearn.decomposition import PCA
    if method == "pca":
        return PCA(n_components=2, random_state=seed).fit(X)
    if method == "tsne":
        from sklearn.manifold import TSNE
        Xp = PCA(n_components=min(30, X.shape[1]), random_state=seed).fit_transform(X)
        perplexity = min(30, max(5, (Xp.shape[0] - 1) // 3))
        coords = TSNE(n_components=2, perplexity=perplexity,
                      init="pca", random_state=seed).fit_transform(Xp)
        return ("tsne_fixed", coords)          # precomputed, non-reusable
    raise ValueError(method)


def _transform(projector, X):
    if isinstance(projector, tuple) and projector[0] == "tsne_fixed":
        return projector[1]
    return projector.transform(X)


def _scatter(ax, real, zeroshot, mu, sd, method="pca", seed=0,
             projector=None, zscore=True):
    """Centred, joint-projected scatter of one target onto `ax`. Returns cos_best.

    `projector` (a fitted PCA) shares axes across a grid; if None we fit locally on
    this target's centred (real + gen) cloud.
    """
    K, L, _ = zeroshot.shape
    rc = _center(real, mu, sd, zscore)
    gc = _center(zeroshot.reshape(K * L, -1), mu, sd, zscore)
    X = np.concatenate([rc, gc], axis=0)
    if projector is None:
        projector = _fit_projector(X, method, seed=seed)
    emb = _transform(projector, X)
    r2, z2 = emb[:L], emb[L:]
    ax.scatter(z2[:, 0], z2[:, 1], s=8, alpha=0.3, c="tab:orange",
               label=f"latent_fm (K={K})")
    ax.scatter(r2[:, 0], r2[:, 1], s=18, alpha=0.9, c="tab:blue",
               edgecolors="k", linewidths=0.2, label="real")
    return float(per_seed_cos(real, zeroshot, mu, sd, zscore).max())


def _render_target(target, real, zeroshot, mu, sd, out, method="both", seed=0,
                   provisional=False):
    """Save a single-target scatter figure. Returns cos_best (centred, best-of-K).

    `mu`/`sd` are the SHARED test-set channel stats (same reference used by
    plot-all), so this target's cos_best matches across plot / plot-all / inline.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = compute_metrics(real, zeroshot, mu, sd)
    methods = ["pca", "tsne"] if method == "both" else [method]
    fig, axes = plt.subplots(1, len(methods), figsize=(7 * len(methods), 6), squeeze=False)
    for ax, mth in zip(axes[0], methods):
        _scatter(ax, real, zeroshot, mu, sd, method=mth, seed=seed)
        ax.set_title(f"{mth.upper()} — centred per-residue MSA embedding")
        ax.set_xlabel(f"{mth}-1"); ax.set_ylabel(f"{mth}-2")
        ax.legend(loc="best", framealpha=0.9)
    tag = "cos_best*(provisional)" if provisional else "cos_best"
    fig.suptitle(f"{target}   L={real.shape[0]}   "
                 f"{tag}={m['cos_best']:.3f}  cos_mean={m['cos_mean']:.3f}  "
                 f"div={m['diversity']:.3f}", fontsize=12)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return m["cos_best"]


def plot(args):
    real, zeroshot = _load_pair(args.out_base, args.target)
    # Same test-set MU as plot-all → cos_best is consistent between them.
    mu, sd = _testset_stats(args.out_base)
    out = args.out or str(Path(args.out_base) / args.target / f"{args.target}_compare.png")
    cos = _render_target(args.target, real, zeroshot, mu, sd, out, args.method, args.seed)
    logger.info("wrote %s   (cos_best=%.4f)", out, cos)


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

    # Test-set MU/SD from pooled real (training mean unavailable), shared by every
    # centred metric, the common PCA axes below, and plot / inline-extract.
    mu, sd = _testset_stats(args.out_base, targets)

    # Null baseline: pair each target with the NEXT target's real (cyclic), so the
    # lower bound uses a genuinely different protein.
    order = list(targets)
    others = {t: order[(i + 1) % len(order)] for i, t in enumerate(order)} if len(order) > 1 else {}

    met = {}
    for t in targets:
        real, zeroshot = pairs[t]
        real_other = pairs[others[t]][0] if t in others else None
        met[t] = compute_metrics(real, zeroshot, mu, sd, real_other=real_other)

    # ★ picks on cos_best (paper protocol), NOT the old seed-mean cosine.
    best = max(targets, key=lambda t: met[t]["cos_best"])
    logger.info("per-target metrics (sorted by cos_best):")
    for t in sorted(targets, key=lambda t: -met[t]["cos_best"]):
        m = met[t]
        logger.info("  %-14s cos_best=%.3f  cos_mean=%.3f  div=%.2f  cos_null=%s",
                    t, m["cos_best"], m["cos_mean"], m["diversity"],
                    "n/a" if m["cos_null"] is None else f"{m['cos_null']:.3f}")
    # Dataset-level means — the single-glance summary.
    import statistics as _st
    agg = {k: _st.fmean(met[t][k] for t in targets if met[t][k] is not None)
           for k in ("cos_best", "cos_mean", "diversity", "cos_null")
           if any(met[t][k] is not None for t in targets)}
    logger.info("DATASET mean: %s", {k: round(v, 4) for k, v in agg.items()})
    logger.info("★ best = %s (cos_best=%.4f)", best, met[best]["cos_best"])

    # Per-target CSV (same numbers as the log) — e.g. to join against folding
    # TM/lDDT. One row per target, sorted by cos_best.
    csv_path = args.csv or str(Path(args.out_base) / f"compare_all_{args.method}_metrics.csv")
    cols = ["target", "L", "cos_best", "cos_mean", "diversity", "cos_null"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for t in sorted(targets, key=lambda t: -met[t]["cos_best"]):
            w.writerow({"target": t, "L": pairs[t][0].shape[0], **{
                k: ("" if met[t][k] is None else met[t][k]) for k in cols[2:]}})
    logger.info("wrote %s (%d targets)", csv_path, len(targets))

    # Common axes: one PCA fit on the pooled CENTRED real manifold, reused for every
    # panel — so "is gen inside the real manifold or collapsed to the centre?" is
    # comparable across the grid. (t-SNE has no shared transform → per-panel fit.)
    projector = None
    if args.method == "pca":
        pooled = np.concatenate([_center(pairs[t][0], mu, sd) for t in targets], axis=0)
        projector = _fit_projector(pooled, "pca", seed=args.seed)

    n = len(targets)
    ncol = min(args.ncol, n)
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4.6 * nrow), squeeze=False)
    for i, t in enumerate(targets):
        ax = axes[i // ncol][i % ncol]
        real, zeroshot = pairs[t]
        _scatter(ax, real, zeroshot, mu, sd, method=args.method, seed=args.seed,
                 projector=projector)
        star = "★ " if t == best else ""
        ax.set_title(f"{star}{t}  best={met[t]['cos_best']:.2f} "
                     f"mean={met[t]['cos_mean']:.2f} div={met[t]['diversity']:.1f}"
                     + ("  (BEST)" if t == best else ""),
                     fontsize=9, fontweight="bold" if t == best else "normal")
        ax.set_xticks([]); ax.set_yticks([])
    for j in range(n, nrow * ncol):            # blank unused cells
        axes[j // ncol][j % ncol].axis("off")
    axes[0][0].legend(loc="best", fontsize=8, framealpha=0.9)

    null_txt = f"  null={agg['cos_null']:.3f}" if "cos_null" in agg else ""
    fig.suptitle(f"real vs latent_fm MSA embedding ({args.method.upper()}, centred) — "
                 f"★ best = {best} (cos_best={met[best]['cos_best']:.3f})"
                 f"  |  dataset cos_best={agg['cos_best']:.3f}{null_txt}", fontsize=13)
    fig.tight_layout()
    out = args.out or str(Path(args.out_base) / f"compare_all_{args.method}.png")
    fig.savefig(out, dpi=150)
    logger.info("wrote %s", out)


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3: spread diagnostic (CPU)
#
# Low diversity has two causes with OPPOSITE fixes, indistinguishable from the
# diversity scalar alone. Decompose the centred generated z along both axes:
#
#   sd_real  r.std(axis=residue)              per-channel real spread. Since MU/SD
#                                             are fit on real, pooled ≈ 1.0 → the
#                                             free reference ("how wide is real").
#   sd_gen   mean_seed(g).std(axis=residue)   how the generated embedding VARIES
#                                             along the sequence (per channel).
#   sd_seed  g.std(axis=seed).mean(residue)   seed-to-seed spread at a residue —
#                                             this is what `diversity` summarises.
#
# sd_gen tells the two apart (real SD = 1.0 reference):
#   sd_gen ≈ 1  generation is as wide as real but seeds cluster → model near
#               DETERMINISTIC given ESM2 → fix the SAMPLING (noise schedule /
#               seed injection / ODE steps).
#   sd_gen ≈ .2 generation collapsed toward MU → MODE COLLAPSE → fix the TRAINING
#               objective (FM loss favouring the mean).
#   sd_gen > 1  over-diffusion / scale bug (unlikely).
# Per-channel 128-vectors also show whether a FEW channels died or ALL shrank
# uniformly (→ npz, for the next clue: what the dead channels encode).
# ──────────────────────────────────────────────────────────────────────────────

def spread(args):
    targets = args.targets or _extracted_targets(args.out_base)
    if not targets:
        raise SystemExit(f"no extracted targets under {args.out_base}")
    # One shared reference: full test-set MU/SD (so real z-scored → per-channel ≈1).
    mu, sd = _testset_stats(args.out_base)

    rows, ch_real, ch_gen, ch_seed = [], [], [], []
    for t in targets:
        real, gen = _load_pair(args.out_base, t)            # (L,128), (K,L,128)
        rc = _center(real, mu, sd)                          # (L,128)
        gc = _center(gen, mu, sd)                           # (K,L,128)
        sd_real = rc.std(axis=0)                            # (128,)
        sd_gen = gc.mean(axis=0).std(axis=0)                # (128,) spread along residue
        sd_seed = (gc.std(axis=0).mean(axis=0) if gc.shape[0] > 1
                   else np.zeros(gc.shape[-1]))             # (128,) spread across seeds
        ch_real.append(sd_real); ch_gen.append(sd_gen); ch_seed.append(sd_seed)
        rows.append({"target": t, "L": real.shape[0],
                     "sd_real": float(sd_real.mean()),
                     "sd_gen": float(sd_gen.mean()),
                     "sd_seed": float(sd_seed.mean()),
                     "diversity": seed_diversity(gen, mu, sd)})

    # Dataset per-channel means (128,) — averaged over targets.
    chan_real = np.mean(ch_real, axis=0)
    chan_gen = np.mean(ch_gen, axis=0)
    chan_seed = np.mean(ch_seed, axis=0)

    # ── CSV (per target) ──
    csv_path = args.out or str(Path(args.out_base) / "spread_diag.csv")
    cols = ["target", "L", "sd_real", "sd_gen", "sd_seed", "diversity"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for r in sorted(rows, key=lambda r: r["sd_gen"]):
            w.writerow(r)
    logger.info("wrote %s (%d targets)", csv_path, len(rows))

    # ── per-channel 128-vectors (for 'which channels died') ──
    npz = str(Path(csv_path).with_name("spread_diag_channels.npz"))
    np.savez_compressed(npz, channel=np.arange(chan_gen.size),
                        sd_real=chan_real, sd_gen=chan_gen, sd_seed=chan_seed,
                        targets=np.array(targets),
                        sd_gen_per_target=np.stack(ch_gen),
                        sd_seed_per_target=np.stack(ch_seed))
    logger.info("wrote %s (per-channel vectors)", npz)

    # ── dataset summary ──
    m_real, m_gen, m_seed = chan_real.mean(), chan_gen.mean(), chan_seed.mean()
    div_mean = float(np.mean([r["diversity"] for r in rows]))
    print("\n=== spread (dataset mean, centred; real per-channel SD ≡ 1.0 ref) ===")
    print(f"  sd_real  {m_real:.3f}   (sanity: should be ≈1.0)")
    print(f"  sd_gen   {m_gen:.3f}   generation spread along residue")
    print(f"  sd_seed  {m_seed:.3f}   seed-to-seed spread  (= diversity/√2 ≈ {div_mean/np.sqrt(2):.3f})")
    print(f"  diversity {div_mean:.3f}  (cross-check of sd_seed)")

    n_dead = int((chan_gen < 0.3).sum())
    print(f"  channels with sd_gen<0.3: {n_dead}/{chan_gen.size}  "
          f"({'a FEW channels died' if 0 < n_dead < chan_gen.size // 2 else 'uniform shrink' if n_dead else 'none dead'})")

    print("\n=== cause (from sd_gen vs real=1.0) ===")
    if m_gen > 1.2:
        print(f"  sd_gen={m_gen:.2f} > 1 → OVER-DIFFUSION / scale bug (check normalisation).")
    elif m_gen >= 0.8:
        print(f"  sd_gen={m_gen:.2f} ≈ 1 → generation as wide as real but seeds cluster "
              f"(sd_seed={m_seed:.2f}). Model near-DETERMINISTIC given ESM2 → fix SAMPLING "
              f"(noise schedule / seed-injection point / ODE steps / temperature).")
    elif m_gen < 0.4:
        print(f"  sd_gen={m_gen:.2f} ≪ 1 → generation CONTRACTED toward MU → MODE COLLAPSE "
              f"→ fix the TRAINING objective (FM loss converging to the mean).")
    else:
        print(f"  sd_gen={m_gen:.2f} (0.4–0.8) → partial contraction; inspect per-channel "
              f"npz — are specific channels dead or all shrunk?")

    # ── per-channel plot ──
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            order = np.argsort(chan_gen)                    # sort channels by sd_gen
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(chan_real[order], label="sd_real (≈1)", c="tab:blue")
            ax.plot(chan_gen[order], label="sd_gen (vs residue)", c="tab:orange")
            ax.plot(chan_seed[order], label="sd_seed (vs seed)", c="tab:green")
            ax.axhline(1.0, ls=":", c="gray", lw=0.8)
            ax.set_xlabel("channel (sorted by sd_gen)"); ax.set_ylabel("per-channel SD (centred)")
            ax.set_title(f"spread per channel — sd_gen̄={m_gen:.2f} sd_seed̄={m_seed:.2f}")
            ax.legend(loc="best")
            fig.tight_layout()
            png = str(Path(csv_path).with_name("spread_diag.png"))
            fig.savefig(png, dpi=150); logger.info("wrote %s", png)
        except Exception as e:
            logger.warning("plot skipped: %s", e)


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
    pe.add_argument("--no_resume", dest="resume", action="store_false",
                    help="re-extract even if npz already exist (default: resume/skip)")
    pe.add_argument("--no_plot", dest="plot", action="store_false",
                    help="skip the inline per-target figure (default: plot inline)")
    pe.add_argument("--plot_method", choices=["pca", "tsne", "both"], default="pca",
                    help="inline plot projection (default: pca — fast)")
    pe.set_defaults(func=extract, resume=True, plot=True)

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
    pa.add_argument("--out", default=None, help="grid PNG path")
    pa.add_argument("--csv", default=None,
                    help="per-target metrics CSV path (default: "
                         "<out_base>/compare_all_<method>_metrics.csv)")
    pa.add_argument("--seed", type=int, default=0)
    pa.set_defaults(func=plot_all)

    ps = sub.add_parser("spread",
                        help="CPU: decompose low diversity → sd_real/sd_gen/sd_seed "
                             "per channel; tells mode-collapse (training) from "
                             "near-deterministic (sampling)")
    ps.add_argument("--targets", nargs="*", default=None, help="default: all extracted")
    ps.add_argument("--out", default=None, help="CSV path (npz/png written alongside)")
    ps.add_argument("--no_plot", action="store_true", help="skip the per-channel figure")
    ps.set_defaults(func=spread)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
