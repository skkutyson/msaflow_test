#!/usr/bin/env python3
"""Regression baseline vs. latent-FM generation — is cosine a valid FM metric?

Goal
----
A ridge / small-MLP head that directly regresses e → z optimises exactly the
conditional mean Ê[z|e]. If that regressor's per-residue cosine to real z BEATS
the FM's best-of-K cosine (while the FM still has non-zero sample diversity),
then cosine rewards mean-seeking and is the WRONG yardstick for a generative
model — the comparison should move to decoder-level quality instead. This script
makes that call quantitatively.

Reuses the (corrected) centred-cosine / diversity / null helpers from
`compare_msa_emb.py` so there is a single source of truth for the metric.

Inputs (same layout as compare_msa_emb extract, under --out_base)
    <base>/<target>/real/<target>.npz        real z   (L, 128)
    <base>/<target>/latent_fm/<target>.npz   gen  z   (K, L, 128), K=4
    <base>/<target>/esm/<target>.npz         esm  e   (L, 1280)   ← this script

ESM e is now written by compare's extract (cropped to the same L as real). The
GPU `cache-esm` step here is only a BACKFILL for OLDER extractions that predate
that — it never overwrites an existing esm npz. The `run` step is CPU-only.

Anti-leakage protocol (as specified)
    * Split is at the TARGET (protein) level — never shuffle residues.
    * MU (mean) and SD (per-channel std) are fit on TRAIN residues only, then
      applied to test. Cosine is always on centred (z - MU)/SD vectors; without
      centring the shared mean-pooled component saturates everything to ~0.9.
    * Ridge alpha is chosen inside the train split (RidgeCV LOO), never on test.
    * K-fold over targets so every target is held out exactly once → one row per
      target in the output, and a full diversity-vs-cos_fm_best scatter.

Metrics per test target (cosine = dim=-1 channel axis, per-residue then mean)
    cos_ridge    ridge e→z prediction vs real
    cos_fm_best  max over K seeds           (paper protocol)
    cos_fm_mean  mean over K seeds          (NOT the mean-seed cosine)
    cos_fm_avg   cosine of the K-mean seed  (the OLD metric — comparison only)
    diversity    per-dim RMS pairwise distance among K centred seeds (√D-normalised)
    cos_null     gen vs a DIFFERENT protein's real (lower bound)

Usage
-----
    # GPU node (sbatch — never on a login node): cache ESM e for all targets
    uv run python scripts/regression_baseline.py cache-esm

    # CPU: fit, evaluate, write CSV + scatter + verdict
    uv run python scripts/regression_baseline.py run --mlp
"""
import argparse
import csv
import importlib.util
import logging
import random
from pathlib import Path

import numpy as np

logger = logging.getLogger("regression_baseline")

DEF_OUT_BASE = "/store/yjlee4/runs/compare_emb"

# Reuse the corrected metric helpers from compare_msa_emb.py (same dir), so the
# centring / cosine / diversity / null definitions never drift between scripts.
_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("compare_msa_emb", _HERE / "compare_msa_emb.py")
cme = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cme)


# ESM path is owned by compare_msa_emb (extract saves it there) — reuse it so the
# two scripts never disagree on the layout.
_esm_npz = cme._esm_npz


def _has_all(base, t):
    return (cme._real_npz(base, t).exists()
            and cme._lfm_npz(base, t).exists()
            and _esm_npz(base, t).exists())


def _discover(base):
    """All target dirs under out_base that have real + latent_fm npz."""
    return sorted(d.name for d in Path(base).iterdir() if d.is_dir()
                  and cme._real_npz(base, d.name).exists()
                  and cme._lfm_npz(base, d.name).exists())


# ──────────────────────────────────────────────────────────────────────────────
# GPU: BACKFILL ESM2 e for OLD extractions that predate extract saving it.
#
# compare's extract now writes esm (cropped to real's L). This step only fills
# in targets whose esm npz is MISSING, recomputing from the stored query and
# cropping to the real/gen common L so it stays prefix-aligned. It NEVER
# overwrites an existing (already-cropped-correct) esm npz.
# ──────────────────────────────────────────────────────────────────────────────

def cache_esm(args):
    import torch
    from msaflow.data.preprocessing import extract_esm_embedding
    from msaflow.inference.generate import load_esm2

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cand = args.targets or _discover(args.out_base)
    # Protect extract's cropped esm: only backfill MISSING ones, always.
    targets = [t for t in cand if not _esm_npz(args.out_base, t).exists()]
    skipped = len(cand) - len(targets)
    if skipped:
        logger.info("%d target(s) already have esm (extract-written) — left untouched", skipped)
    if not targets:
        logger.info("nothing to backfill (all esm npz present)")
        return

    esm_model, alphabet = load_esm2(device)
    for i, t in enumerate(targets, 1):
        real_npz = cme._real_npz(args.out_base, t)
        if not real_npz.exists():
            logger.warning("[%d/%d] %s no real npz — skip", i, len(targets), t)
            continue
        rd = np.load(real_npz)
        real, query = rd["real"], str(rd["query"])
        gen = np.load(cme._lfm_npz(args.out_base, t))["zeroshot"]
        emb = extract_esm_embedding(query, esm_model, alphabet, device).float().numpy()
        # Crop to the same prefix L as real/gen (what _load_triple expects).
        L = min(real.shape[0], gen.shape[1], emb.shape[0])
        if emb.shape[0] != L:
            logger.warning("[%d/%d] %s esm L=%d → cropped to %d (real=%d gen=%d)",
                           i, len(targets), t, emb.shape[0], L, real.shape[0], gen.shape[1])
        out = _esm_npz(args.out_base, t)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, esm=emb[:L], query=query)
        logger.info("[%d/%d] %s  esm%s backfilled", i, len(targets), t, emb[:L].shape)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    logger.info("cache-esm backfill done under %s", args.out_base)


# ──────────────────────────────────────────────────────────────────────────────
# CPU: load one target cropped to a common L across real / gen / esm
# ──────────────────────────────────────────────────────────────────────────────

def _load_triple(base, t):
    real = np.load(cme._real_npz(base, t))["real"]            # (L, 128)
    gen = np.load(cme._lfm_npz(base, t))["zeroshot"]          # (K, L, 128)
    esm = np.load(_esm_npz(base, t))["esm"]                   # (L, 1280)
    L = min(real.shape[0], gen.shape[1], esm.shape[0])
    return real[:L], gen[:, :L], esm[:L]


# ──────────────────────────────────────────────────────────────────────────────
# Per-target metrics given train-fit MU/SD and a ridge model
# ──────────────────────────────────────────────────────────────────────────────

def _cos_centred(a, b, mu, sd):
    """Mean per-residue centred cosine between two (L,128) arrays."""
    return float(cme._cos_lastdim(cme._center(a, mu, sd), cme._center(b, mu, sd)).mean())


def _eval_target(real, gen, esm, scaler, ridge, mu, sd, real_other, mlp=None):
    """FM metrics + regression cosines for one test target.

    `ridge` is always evaluated → cos_ridge. If `mlp` is given it is ADDED as a
    separate cos_mlp column (never silently replacing cos_ridge).
    """
    seed_cos = cme.per_seed_cos(real, gen, mu, sd)           # (K,) centred
    xs = scaler.transform(esm)
    out = {
        "L": real.shape[0],
        "cos_ridge": _cos_centred(ridge.predict(xs), real, mu, sd),
        "cos_fm_best": float(seed_cos.max()),
        "cos_fm_mean": float(seed_cos.mean()),
        "cos_fm_avg": _cos_centred(gen.mean(axis=0), real, mu, sd),   # OLD metric
        "diversity": cme.seed_diversity(gen, mu, sd),
        "cos_null": cme.null_cos(gen, real_other, mu, sd),
    }
    if mlp is not None:
        out["cos_mlp"] = _cos_centred(mlp.predict(xs), real, mu, sd)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# CPU: K-fold-over-targets fit + eval
# ──────────────────────────────────────────────────────────────────────────────

def run(args):
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    base = args.out_base
    pool = args.targets or _discover(base)                   # scan the dir ONCE
    targets = [t for t in pool if _has_all(base, t)]
    missing = [t for t in pool if not _has_all(base, t)]
    if missing:
        logger.warning("%d target(s) missing esm/real/gen — run cache-esm first: %s",
                       len(missing), ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else ""))
    if len(targets) < 2:
        raise SystemExit(f"need >=2 fully-cached targets, have {len(targets)}")

    triples = {t: _load_triple(base, t) for t in targets}

    # Target-level folds, balanced by length: sort by L then snake-distribute so
    # each fold's train residue count is comparable (a plain order[i::k] can hand
    # one fold all the long proteins → unfair ridge fits across folds). Every
    # protein is a test protein exactly once.
    k = min(args.kfold, len(targets))
    by_len = sorted(targets, key=lambda t: triples[t][0].shape[0])
    folds = [[] for _ in range(k)]
    for idx, t in enumerate(by_len):
        f = idx % k
        if (idx // k) % 2 == 1:                              # snake: reverse every pass
            f = k - 1 - f
        folds[f].append(t)
    logger.info("%d targets, %d folds (target-level, length-balanced, no residue leakage)",
                len(targets), k)

    # cos_null partner is drawn from the SAME fold's TEST set (a protein the ridge
    # never trained on) — not the global pool. Its own RNG (seed+1) so changing
    # --kfold doesn't perturb the null via shared shuffle state.
    null_rng = random.Random(args.seed + 1)

    use_mlp = args.mlp
    if use_mlp:
        logger.warning("--mlp: an MLP head is re-fit per fold on tens of thousands of "
                       "residues (CPU) — noticeably slower than ridge-only. Added as a "
                       "separate cos_mlp column.")

    alphas = [0.1, 1.0, 10.0, 100.0]
    rows = []
    for fi, test_ts in enumerate(folds, 1):
        train_ts = [t for t in targets if t not in set(test_ts)]
        if not train_ts or not test_ts:
            continue

        # MU/SD from TRAIN residues only (single (128,) vector each).
        mu, sd = cme.channel_stats([triples[t][0] for t in train_ts])

        # Ridge e→z on TRAIN residues; alpha via LOO inside train (no test peek).
        Xtr = np.concatenate([triples[t][2] for t in train_ts], axis=0)   # (N,1280)
        Ytr = np.concatenate([triples[t][0] for t in train_ts], axis=0)   # (N,128)
        scaler = StandardScaler().fit(Xtr)
        ridge = RidgeCV(alphas=alphas).fit(scaler.transform(Xtr), Ytr)
        logger.info("fold %d/%d  train=%d targets (%d res)  test=%d  ridge alpha=%.3g",
                    fi, k, len(train_ts), Xtr.shape[0], len(test_ts),
                    float(np.atleast_1d(ridge.alpha_).mean()))

        mlp = None
        if use_mlp:
            from sklearn.neural_network import MLPRegressor
            mlp = MLPRegressor(hidden_layer_sizes=(768,), early_stopping=True,
                               max_iter=args.mlp_iter, random_state=args.seed)
            mlp.fit(scaler.transform(Xtr), Ytr)

        for t in test_ts:
            real, gen, esm = triples[t]
            cands = [u for u in test_ts if u != t] or [u for u in targets if u != t]
            other = null_rng.choice(cands)
            m = _eval_target(real, gen, esm, scaler, ridge, mu, sd, triples[other][0], mlp=mlp)
            m["target"] = t
            rows.append(m)

    _report(args, rows, use_mlp)


# ──────────────────────────────────────────────────────────────────────────────
# Output: CSV + mean table + scatter + verdict
# ──────────────────────────────────────────────────────────────────────────────

def _report(args, rows, use_mlp=False):
    rows.sort(key=lambda r: -r["cos_fm_best"])

    # cos_mlp only exists when --mlp was on → put it right after cos_ridge.
    cols = ["target", "L", "cos_ridge"]
    if use_mlp:
        cols.append("cos_mlp")
    cols += ["cos_fm_best", "cos_fm_mean", "cos_fm_avg", "diversity", "cos_null"]

    csv_path = args.csv_out or str(Path(args.out_base) / "regression_baseline.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in cols})
    logger.info("wrote %s (%d targets, model=%s)",
                csv_path, len(rows), "ridge+mlp" if use_mlp else "ridge")

    metric_cols = [c for c in cols if c not in ("target", "L")]
    means = {c: float(np.mean([r[c] for r in rows])) for c in metric_cols}
    print("\n=== dataset mean ===")
    for c in metric_cols:
        print(f"  {c:12s} {means[c]:+.4f}")

    # Sanity: cos_null should sit near 0 after centring. If it is high, the shared
    # component was NOT removed → every cosine is saturated and meaningless.
    if means["cos_null"] > 0.5:
        logger.warning("cos_null=%.3f (>0.5) — centring did NOT remove the shared "
                       "component; MU/SD are suspect. Treat all cosines as invalid "
                       "until this is fixed.", means["cos_null"])

    # Scatter: x=diversity, y=cos_fm_best, one point per target.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 6))
        xs = [r["diversity"] for r in rows]
        ys = [r["cos_fm_best"] for r in rows]
        ax.scatter(xs, ys, s=30, c="tab:blue", edgecolors="k", linewidths=0.3)
        ax.axhline(means["cos_ridge"], ls="--", c="tab:red",
                   label=f"ridge mean cos={means['cos_ridge']:.3f}")
        if "cos_mlp" in means:
            ax.axhline(means["cos_mlp"], ls="--", c="tab:green",
                       label=f"mlp mean cos={means['cos_mlp']:.3f}")
        ax.axhline(means["cos_null"], ls=":", c="gray",
                   label=f"null mean cos={means['cos_null']:.3f}")
        for r in rows:
            ax.annotate(r["target"], (r["diversity"], r["cos_fm_best"]),
                        fontsize=6, alpha=0.6)
        ax.set_xlabel("seed diversity (per-dim RMS pairwise distance, centred)")
        ax.set_ylabel("cos_fm_best (centred)")
        ax.set_title("FM best-of-K cosine vs. sample diversity")
        ax.legend(loc="best")
        fig.tight_layout()
        png = args.plot_out or str(Path(args.out_base) / "regression_baseline.png")
        fig.savefig(png, dpi=150)
        logger.info("wrote %s", png)
    except Exception as e:                                    # plotting must not kill the verdict
        logger.warning("scatter skipped: %s", e)

    _verdict(means, args.div_tol)


def _verdict(means, div_tol):
    div, fm_best, ridge = means["diversity"], means["cos_fm_best"], means["cos_ridge"]
    print("\n=== verdict ===")
    print(f"  (ref: per-dim RMS diversity ≈1.41 for independent z-scored samples, "
          f"0 = identical; collapse threshold div_tol={div_tol})")
    if div < div_tol and abs(fm_best - ridge) < 0.02:
        print(f"  FM COLLAPSED to a deterministic map (per-dim RMS diversity={div:.3g} "
              f"< {div_tol}, cos_fm_best≈cos_ridge). → inspect the sampling code "
              f"(temperature / SDE).")
    elif ridge > fm_best and div >= div_tol:
        print(f"  cos_ridge ({ridge:.3f}) > cos_fm_best ({fm_best:.3f}) with diversity "
              f"{div:.3g}. Expected: cosine rewards the conditional mean → it is NOT a "
              f"valid FM metric. → move to decoder-level quality comparison.")
    elif fm_best > ridge:
        print(f"  cos_fm_best ({fm_best:.3f}) > cos_ridge ({ridge:.3f}): FM wins even "
              f"point-wise. Unexpected — re-check ridge fit / centring before trusting.")
    else:
        print(f"  Ambiguous (fm_best={fm_best:.3f}, ridge={ridge:.3f}, div={div:.3g}). "
              f"Inspect per-target CSV.")


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

    pc = sub.add_parser("cache-esm", help="GPU: cache ESM2 e per target from stored query")
    pc.add_argument("--targets", nargs="*", default=None)
    pc.add_argument("--device", default="cuda")
    pc.add_argument("--no_resume", dest="resume", action="store_false",
                    help="(no-op) esm npz are never overwritten; only missing ones backfill")
    pc.set_defaults(func=cache_esm, resume=True)

    pr = sub.add_parser("run", help="CPU: fit ridge/MLP, evaluate, CSV + scatter + verdict")
    pr.add_argument("--targets", nargs="*", default=None)
    pr.add_argument("--kfold", type=int, default=5, help="target-level folds (default 5)")
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--mlp", action="store_true",
                    help="ALSO fit a 1280→768→128 MLP head → separate cos_mlp column "
                         "(slower, per-fold refit on CPU); ridge still runs")
    pr.add_argument("--mlp_iter", type=int, default=300)
    pr.add_argument("--div_tol", type=float, default=0.1,
                    help="per-dim RMS seed diversity below this = collapse. On the "
                         "z-scored scale two INDEPENDENT samples sit at ≈√2≈1.41 and "
                         "0 = identical, so values of 0.1–0.3 already mean severe "
                         "collapse (default 0.1); √(L·128)-normalised, so L-independent")
    pr.add_argument("--csv_out", default=None)
    pr.add_argument("--plot_out", default=None)
    pr.set_defaults(func=run)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
