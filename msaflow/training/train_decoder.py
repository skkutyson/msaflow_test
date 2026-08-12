"""
Training script for the MSAFlow SFM Decoder.

Matches the paper's setup (Section 6.8.2):
  • 129M parameter SFM decoder (DiT, 12 blocks, hidden 768, 12 heads)
  • 7 epochs on OpenFold-derived LMDB (4M MSAs)
  • 4 × H200 GPUs via Accelerate
  • LR warmup 5000 steps, peak LR 1e-5, weight decay 0.1
  • Batch size: 2560 max query-sequence tokens
  • 32 sequences sampled per MSA (Neff-weighted)

Usage:
    accelerate launch --num_processes 4 msaflow/training/train_decoder.py \
        --lmdb_path /data/msaflow.lmdb \
        --output_dir /runs/decoder \
        --config msaflow/configs/decoder.yaml
"""

import os
import math
import logging
import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from accelerate import Accelerator
from accelerate.utils import set_seed, DataLoaderConfiguration
from omegaconf import OmegaConf

import numpy as np

from msaflow.models.sfm_decoder import SFMDecoder, flow_loss
from msaflow.data.dataset import MSADecoderDataset, decoder_collate_fn
from msaflow.data.preprocessing import AA_LIST
from msaflow.utils.spherical import sample_sphere_noise, euler_step_sphere, decode_sequences
from msaflow.utils.alpha_geometry import geometry_ops as _geometry_ops

logger = logging.getLogger(__name__)


def _unwrap_model(model):
    """Strip DDP/accelerate wrappers to the underlying nn.Module.

    We avoid accelerator.unwrap_model() on purpose: it calls
    extract_model_from_parallel, which does `from deepspeed import DeepSpeedEngine`
    just for an isinstance check. On nodes without CUDA_HOME that import crashes
    (deepspeed probes the CUDA toolchain at import time). We never use deepspeed
    here (single-GPU / DDP), so peeling `.module` is equivalent and import-free.
    """
    while hasattr(model, "module"):
        model = model.module
    return model


def _tokens_to_strs(tokens: torch.Tensor) -> list:
    """Convert (B, L) integer token tensor to list of amino-acid strings."""
    rows = tokens.cpu().tolist()
    return ["".join(AA_LIST[t] if t < len(AA_LIST) else "?" for t in row) for row in rows]


# ──────────────────────────────────────────────────────────────────────────────
# Validation: generation quality metrics beyond loss
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _compute_gen_metrics(
    gen_tokens: torch.Tensor,
    ref_tokens: torch.Tensor,
    vocab_size: int,
    gap_token: int,
    weights: Optional[torch.Tensor] = None,
) -> dict:
    """Compute generation metrics for a single set of generated tokens."""
    device = gen_tokens.device
    B, L = ref_tokens.shape

    # seq_recovery: nearest-neighbor matching to avoid arbitrary i→i pairing.
    # Generated sequences are permutation-invariant samples from the same
    # conditioning; comparing gen[i] to ref[i] directly underestimates recovery
    # when the sequences are in a different order.
    # For each generated sequence, find the reference sequence with the highest
    # per-position match and measure recovery against that best match.
    match = (gen_tokens.unsqueeze(1) == ref_tokens.unsqueeze(0)).float()  # (B, B, L)
    nn_recovery = match.mean(dim=-1).max(dim=1).values                    # (B,) best-ref per gen
    seq_recovery = nn_recovery.mean().item()

    gap_frac = (gen_tokens == gap_token).float().mean().item()

    gen_np = gen_tokens.cpu().numpy()
    diff = (gen_np[:, None, :] != gen_np[None, :, :]).mean(axis=-1)
    counts = (diff < 0.2).sum(axis=1).astype(np.float32)
    neff_generated = float((1.0 / counts.clip(min=1.0)).sum())

    gen_oh = torch.zeros(B, L, vocab_size, device=device)
    gen_oh.scatter_(2, gen_tokens.unsqueeze(-1), 1.0)
    ref_oh = torch.zeros(B, L, vocab_size, device=device)
    ref_oh.scatter_(2, ref_tokens.unsqueeze(-1), 1.0)

    # Generated sequences: equal weight (no Neff for freshly sampled seqs).
    gen_dist = gen_oh.mean(dim=0).clamp(min=1e-8)
    # Reference distribution: Neff-weighted so redundant homologs don't dominate.
    if weights is not None:
        w = weights.to(device).float()
        w = w / w.sum()
        ref_dist = (ref_oh * w.view(B, 1, 1)).sum(dim=0).clamp(min=1e-8)
    else:
        ref_dist = ref_oh.mean(dim=0).clamp(min=1e-8)

    gen_dist = gen_dist / gen_dist.sum(dim=-1, keepdim=True)
    ref_dist = ref_dist / ref_dist.sum(dim=-1, keepdim=True)
    kl = (ref_dist * (ref_dist.log() - gen_dist.log())).sum(dim=-1)
    aa_kl_div = kl.mean().item()

    return {
        "seq_recovery":   seq_recovery,
        "gap_frac":       gap_frac,
        "neff_generated": neff_generated,
        "aa_kl_div":      aa_kl_div,
    }


@torch.no_grad()
def validate_generation(
    model: SFMDecoder,
    val_batch: dict,
    vocab_size: int,
    n_steps: int = 50,
    gap_token: int = 20,
    temperatures: tuple = (0.0, 0.5, 1.0),
    noise_fn=None,
    step_fn=None,
    decode_fn=None,
) -> dict:
    """
    Generate sequences via Euler integration and compute quality metrics.

    Runs generation at each temperature in `temperatures`.  The t=0.0 (argmax)
    results are always returned under the flat keys for backward compatibility:
      val/seq_recovery, val/gap_frac, val/neff_generated, val/aa_kl_div

    Additional temperature results are stored under:
      val/t{temp}/seq_recovery, val/t{temp}/gap_frac, etc.

    Args:
        model:        unwrapped SFMDecoder in eval mode.
        val_batch:    dict with keys msa_emb (B,L,128), tokens (B,L).
        vocab_size:   V = 22.
        n_steps:      Euler integration steps (50 is sufficient).
        gap_token:    index of the gap '-' token.
        temperatures: decoding temperatures to evaluate.

    Returns:
        dict of scalar metrics.
    """
    # Default to the α=0 sphere ops when none supplied (backward compatible).
    noise_fn = noise_fn or (lambda shape, device, dtype: sample_sphere_noise(shape, device=device, dtype=dtype))
    step_fn = step_fn or (lambda x, v, dt: euler_step_sphere(x, v, dt))
    decode_fn = decode_fn or (lambda x, temperature: decode_sequences(x, temperature=temperature))

    model.eval()
    device = next(model.parameters()).device

    msa_emb = val_batch["msa_emb"].to(device)   # (B, L, 128)
    ref_tokens = val_batch["tokens"].to(device)  # (B, L)
    B, L = ref_tokens.shape

    # ── ODE integration (shared across all temperatures) ─────────────────────
    x_t = noise_fn((B, L, vocab_size), device, msa_emb.dtype)
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((B,), i * dt, device=device, dtype=msa_emb.dtype)
        v = model(x_t, msa_emb, t)
        x_t = step_fn(x_t, v, dt)

    weights = val_batch.get("weights")
    if weights is not None:
        weights = weights.to(device)

    # ── Decode at each temperature and collect metrics ────────────────────────
    metrics = {}
    for temp in temperatures:
        gen_tokens = decode_fn(x_t, temp)   # (B, L)
        m = _compute_gen_metrics(gen_tokens, ref_tokens, vocab_size, gap_token, weights=weights)

        if temp == 0.0:
            # flat keys for backward compat with existing logging / dashboards
            for k, v in m.items():
                metrics[f"val/{k}"] = v
        else:
            temp_str = f"{temp:.1f}".replace(".", "")  # "0.5" → "05"
            for k, v in m.items():
                metrics[f"val/t{temp_str}/{k}"] = v

    return metrics


@torch.no_grad()
def validate_loss(
    model: SFMDecoder,
    val_items: list,
    vocab_size: int,
    alpha: float,
    t_max_eps: float = 0.01,
    n_draws: int = 8,
) -> float:
    """Teacher-forced α-Flow loss on the validation batches (same objective as train).

    flow_loss samples t / x0 / x1 stochastically, so a single call is noisy. We
    average `n_draws` draws per batch (and across batches, Neff-weighted like the
    train loss) to get a stable per-epoch scalar comparable to the train curve.
    Model must already be in eval(); caller restores train() afterwards.
    """
    device = next(model.parameters()).device
    total, count = 0.0, 0
    for vb in val_items:
        tokens = vb["tokens"].to(device)
        msa_emb = vb["msa_emb"].to(device)
        weights = vb.get("weights")
        if weights is not None:
            weights = weights.to(device)
        seq_lengths = vb.get("seq_lengths")
        if seq_lengths is not None:
            seq_lengths = seq_lengths.to(device)
            pos = torch.arange(tokens.shape[1], device=device).unsqueeze(0)
            padding_mask = pos < seq_lengths.unsqueeze(1)
        else:
            padding_mask = None
        for _ in range(n_draws):
            total += flow_loss(
                model, tokens, msa_emb,
                vocab_size=vocab_size, alpha=alpha,
                weights=weights, padding_mask=padding_mask,
                t_max_eps=t_max_eps,
            ).item()
            count += 1
    return total / max(1, count)


# ──────────────────────────────────────────────────────────────────────────────
# Validation sequence example logger
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def save_val_examples(
    model: SFMDecoder,
    val_items: list,
    vocab_size: int,
    epoch: int,
    output_dir: Path,
    n_proteins: int = 3,
    n_seqs: int = 5,
    n_steps: int = 50,
    temperatures: tuple = (0.0, 0.5, 1.0),
    noise_fn=None,
    step_fn=None,
    decode_fn=None,
) -> Path:
    """
    Generate example MSA sequences for a few validation proteins and save to a text file.

    For each of the first `n_proteins` proteins, runs ODE integration once and decodes
    at each temperature, then writes reference + generated sequences side-by-side.

    Returns the path to the saved file.
    """
    noise_fn = noise_fn or (lambda shape, device, dtype: sample_sphere_noise(shape, device=device, dtype=dtype))
    step_fn = step_fn or (lambda x, v, dt: euler_step_sphere(x, v, dt))
    decode_fn = decode_fn or (lambda x, temperature: decode_sequences(x, temperature=temperature))

    model.eval()
    device = next(model.parameters()).device

    out_dir = output_dir / "val_examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"epoch_{epoch:03d}.txt"

    SEP = "=" * 72

    with open(out_file, "w") as f:
        f.write(f"{SEP}\n")
        f.write(f"Epoch {epoch}  —  Validation Sequence Examples\n")
        f.write(f"temperatures: {temperatures}    ODE steps: {n_steps}\n")
        f.write(f"{SEP}\n\n")

        for pi, vb in enumerate(val_items[:n_proteins]):
            msa_emb   = vb["msa_emb"].to(device)    # (B, L, 128)
            ref_tokens = vb["tokens"].to(device)     # (B, L)
            B, L = ref_tokens.shape
            n = min(n_seqs, B)

            # ── ODE integration (shared across all temperatures) ──────────────
            emb = msa_emb[:n]                        # (n, L, 128)
            x_t = noise_fn((n, L, vocab_size), device, emb.dtype)
            dt = 1.0 / n_steps
            for step_i in range(n_steps):
                t = torch.full((n,), step_i * dt, device=device, dtype=emb.dtype)
                v = model(x_t, emb, t)
                x_t = step_fn(x_t, v, dt)

            f.write(f"{'─'*72}\n")
            f.write(f"Protein {pi}   (B={B} ref seqs, L={L} residues)\n")
            f.write(f"{'─'*72}\n\n")

            # Reference sequences
            f.write("Reference (from val LMDB):\n")
            for si, seq in enumerate(_tokens_to_strs(ref_tokens[:n])):
                f.write(f"  ref[{si}]: {seq}\n")

            # Generated sequences at each temperature
            for temp in temperatures:
                gen_tokens = decode_fn(x_t, temp)
                label = f"t={temp:.1f}" + ("  (argmax)" if temp == 0.0 else "")
                f.write(f"\nGenerated {label}:\n")
                for si, seq in enumerate(_tokens_to_strs(gen_tokens)):
                    f.write(f"  gen[{si}]: {seq}\n")

            f.write("\n")

    # ── stdout preview (protein 0 only, first 2 seqs, truncated to 60 chars) ──
    vb0 = val_items[0]
    ref0 = _tokens_to_strs(vb0["tokens"][:2])
    logger.info("Val examples → %s", out_file)
    logger.info("  [protein 0 | L=%d]  ref[0]: %.60s", vb0["tokens"].shape[1], ref0[0])

    # Re-use x_t from the last ODE run (protein n_proteins-1); run fresh for protein 0
    emb0 = vb0["msa_emb"].to(device)[:2]
    L0   = vb0["tokens"].shape[1]
    x0   = noise_fn((2, L0, vocab_size), device, emb0.dtype)
    dt   = 1.0 / n_steps
    for step_i in range(n_steps):
        t = torch.full((2,), step_i * dt, device=device, dtype=emb0.dtype)
        x0 = step_fn(x0, model(x0, emb0, t), dt)
    for temp in temperatures:
        gen0 = _tokens_to_strs(decode_fn(x0, temp))
        logger.info("  [protein 0 | t=%.1f]  gen[0]: %.60s", temp, gen0[0])

    return out_file


# ──────────────────────────────────────────────────────────────────────────────
# Learning rate schedule: linear warmup → cosine decay
# ──────────────────────────────────────────────────────────────────────────────

def get_lr_schedule(
    optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.01,
    last_epoch: int = -1,
):
    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        # Clamp progress to 1.0: total_steps is an estimate, and without the
        # clamp cos(pi*progress) rebounds upward for progress>1, raising the LR
        # again past the end of the schedule.
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine)
    return LambdaLR(optimizer, _lr_lambda, last_epoch=last_epoch)


# ──────────────────────────────────────────────────────────────────────────────
# EMA (Exponential Moving Average)
# ──────────────────────────────────────────────────────────────────────────────

class EMA:
    """Simple EMA wrapper for model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.named_parameters()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if name not in self.shadow:
                self.shadow[name] = param.data.clone().detach()
                continue
            if self.shadow[name].device != param.device:
                self.shadow[name] = self.shadow[name].to(param.device)
            self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def copy_to(self, model: nn.Module):
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])

    def state_dict(self):
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, state):
        # New format: {"shadow": {k: tensor}, "decay": float}
        # Old format: {k: tensor} (backward compat)
        if isinstance(state, dict) and "shadow" in state and isinstance(state["shadow"], dict):
            loaded = state["shadow"]
            self.decay = state.get("decay", self.decay)
        else:
            loaded = state
        # Partial merge: update matching keys, keep current init for new/resized keys.
        for k, v in loaded.items():
            if k in self.shadow and v.shape == self.shadow[k].shape:
                self.shadow[k] = v


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(cfg):
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.grad_accumulation,
        mixed_precision=cfg.training.get("mixed_precision", "bf16"),
        # use_seedable_sampler replaces the default generator-broadcast RNG sync with a
        # deterministic per-epoch seed.  The broadcast relies on sampler.generator being
        # seeded consistently across ranks, which breaks in Accelerate ≥1.0 when each
        # rank draws its seed from the global CPU RNG after prepare().  The result is an
        # inconsistent rng_types list: rank 0 skips the BROADCAST while other ranks
        # block on it, causing a 10-min NCCL timeout.
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
    )
    set_seed(cfg.training.seed)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  [ %(name)s ]  %(message)s",
        datefmt="%Y-%m-%d %H:%M",
    )

    # ── Geometry (α-Flow) ──────────────────────────────────────────────────────
    # Single unified path: p = 2/(1-α).  α=0 → L² sphere (SFM), α=0.5 → L⁴ sphere.
    alpha = float(cfg.model.get("alpha", 0.0))
    noise_fn, step_fn, decode_fn = _geometry_ops(alpha)

    # ── Model ────────────────────────────────────────────────────────────────
    model = SFMDecoder(
        vocab_size=cfg.model.vocab_size,
        msa_dim=cfg.model.msa_dim,
        hidden_size=cfg.model.hidden_size,
        depth=cfg.model.depth,
        num_heads=cfg.model.num_heads,
        mlp_ratio=cfg.model.mlp_ratio,
        max_seq_len=cfg.model.max_seq_len,
        alpha=alpha,
    )
    n_params = sum(p.numel() for p in model.parameters())
    if accelerator.is_main_process:
        logger.info("SFM decoder  ----- %.1fM params  depth=%d  hidden=%d  vocab=%d  alpha=%.2f (p=%.2f)",
                     n_params / 1e6, cfg.model.depth, cfg.model.hidden_size, cfg.model.vocab_size,
                     alpha, 2.0 / (1.0 - alpha))

    ema = EMA(model, decay=cfg.training.ema_decay) if cfg.training.use_ema else None

    # ── Data ─────────────────────────────────────────────────────────────────
    dataset = MSADecoderDataset(
        lmdb_path=cfg.data.lmdb_path,
        n_seqs_per_msa=cfg.data.n_seqs_per_msa,
        max_seq_len=cfg.data.max_seq_len,
    )
    if accelerator.is_main_process:
        logger.info("Dataset loaded  ----- %d entries", len(dataset))

    # Overfit / sanity-check mode: restrict to a tiny fixed subset.
    # If the model cannot overfit N=16 MSAs after 500 epochs, the training
    # pipeline has a fundamental bug (wrong loss sign, bad normalisation, etc.).
    n_debug = cfg.data.get("n_debug_samples", None)
    if n_debug is not None:
        from torch.utils.data import Subset
        n_debug = min(int(n_debug), len(dataset))
        dataset = Subset(dataset, list(range(n_debug)))
        if accelerator.is_main_process:
            logger.info("OVERFIT DEBUG: dataset clamped to %d samples", n_debug)

    loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        collate_fn=decoder_collate_fn,
        drop_last=True,
    )

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        betas=(0.9, 0.95),
    )
    # Divide by num_processes: accelerator.prepare() gives each GPU len(loader)/N batches.
    # Computing before prepare() over-counts by num_processes, making LR decay 2× too fast.
    total_steps = (len(loader) // accelerator.num_processes) * cfg.training.epochs // cfg.training.grad_accumulation
    scheduler = get_lr_schedule(
        optimizer,
        warmup_steps=cfg.training.warmup_steps,
        total_steps=total_steps,
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start_epoch, global_step = 0, 0

    # Local metrics sink (wandb-independent): one JSON object per line.
    # scripts/plot_overfit.py turns this into PNGs — works with no wandb login.
    _metrics_path = output_dir / "metrics.jsonl"

    def _log_metrics(row: dict):
        import json
        with open(_metrics_path, "a") as _f:
            _f.write(json.dumps(row) + "\n")

    ckpt_path = output_dir / "latest.pt"
    if ckpt_path.exists() and cfg.training.resume:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        # Partial load: only restore keys whose shape matches the current model.
        # Handles architecture changes (new block_cond_projs, FinalLayer gate,
        # LayerNorm→RMSNorm) without crashing. New/resized keys keep their
        # zero-initialized values, which is correct for these additions.
        ckpt_sd = ckpt["model"]
        model_sd = model.state_dict()
        compatible = {k: v for k, v in ckpt_sd.items()
                      if k in model_sd and v.shape == model_sd[k].shape}
        skipped = [k for k in ckpt_sd if k not in compatible]
        new_keys = [k for k in model_sd if k not in ckpt_sd]
        model.load_state_dict(compatible, strict=False)
        if accelerator.is_main_process:
            if skipped:
                logger.info("Resume: skipped %d checkpoint keys (shape mismatch / removed): %s",
                            len(skipped), skipped)
            if new_keys:
                logger.info("Resume: %d new keys initialised from scratch: %s",
                            len(new_keys), new_keys)
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
            # Reset optimizer LR to cfg value — load_state_dict restores the old (near-zero) LR.
            for pg in optimizer.param_groups:
                pg["lr"] = cfg.training.lr
        except ValueError as e:
            # Param count changed (e.g. new block_cond_projs added) — optimizer
            # state is incompatible. Start fresh; momentum buffers rebuild in ~1 epoch.
            if accelerator.is_main_process:
                logger.warning("Optimizer state incompatible, starting fresh: %s", e)
        # Rebuild scheduler from scratch, then directly set _step_count.
        # LambdaLR(last_epoch=N) internally calls step() N times — O(N) loop
        # that takes minutes when resuming from step 50000+.
        # Setting _step_count directly is equivalent and instant.
        scheduler = get_lr_schedule(
            optimizer,
            warmup_steps=cfg.training.warmup_steps,
            total_steps=total_steps,
        )
        scheduler._step_count = ckpt["global_step"] + 1
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        if ema is not None and "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
        if accelerator.is_main_process:
            logger.info("Resumed from epoch %d, step %d  (LR reset to %.2e)",
                        start_epoch, global_step, cfg.training.lr)

    # ── Prepare with Accelerate ───────────────────────────────────────────────
    model, optimizer, loader, scheduler = accelerator.prepare(
        model, optimizer, loader, scheduler
    )

    # ── Optional wandb ────────────────────────────────────────────────────────
    use_wandb = cfg.training.get("use_wandb", False) and bool(os.environ.get("WANDB_API_KEY"))
    if accelerator.is_main_process and use_wandb:
        import wandb
        wandb.init(
            project=cfg.training.wandb_project,
            name=cfg.training.get("wandb_run_name", None),
            config=OmegaConf.to_container(cfg),
        )
        wandb.watch(
            _unwrap_model(model),
            log="gradients",
            log_freq=cfg.training.log_every,
        )

    # ── Validation proteins (K fixed proteins, selected once at training start) ──
    val_every = cfg.training.get("val_every", 0)
    val_items = []
    if val_every > 0 and accelerator.is_main_process:
        K = cfg.training.get("val_proteins", 16)
        val_lmdb_path = cfg.data.get("val_lmdb_path", None)
        if val_lmdb_path:
            val_ds = MSADecoderDataset(
                lmdb_path=val_lmdb_path,
                n_seqs_per_msa=cfg.data.n_seqs_per_msa,
                max_seq_len=cfg.data.max_seq_len,
            )
            logger.info("Validation dataset: %s  (%d entries)", val_lmdb_path, len(val_ds))
        else:
            val_ds = dataset.dataset if hasattr(dataset, "dataset") else dataset
        n_total = len(val_ds)
        rng = np.random.default_rng(seed=cfg.training.seed)
        val_indices = rng.choice(n_total, size=min(K, n_total), replace=False).tolist()
        for idx in val_indices:
            raw = val_ds[idx]
            n_seqs = raw["tokens"].shape[0]
            val_items.append({
                "msa_emb": raw["msa_emb"].unsqueeze(0).expand(n_seqs, -1, -1),
                "tokens":  raw["tokens"],
                "weights": raw.get("weights"),
            })
        logger.info("Validation: %d proteins selected from %s (indices %s...)",
                    len(val_items),
                    val_lmdb_path or "train dataset",
                    val_indices[:4])

    # Ensure all ranks are in sync before entering the training loop.
    # Rank 0 may have spent time loading the validation dataset above;
    # without this barrier the other ranks enter the DataLoader iterator first
    # and hang in synchronize_rng_states (BROADCAST) waiting for rank 0.
    accelerator.wait_for_everyone()

    # ── Training loop ─────────────────────────────────────────────────────────
    if accelerator.is_main_process:
        logger.info("Training started  ----- total_steps=%d  epochs=%d  lr=%.2e",
                     total_steps, cfg.training.epochs, cfg.training.lr)
    for epoch in range(start_epoch, cfg.training.epochs):
        model.train()
        epoch_loss = 0.0

        grad_norm = 0.0   # initialised before first sync_gradients assignment
        for batch_idx, batch in enumerate(loader):
            with accelerator.accumulate(model):
                msa_emb     = batch["msa_emb"]      # (B_flat, L, 128)
                tokens      = batch["tokens"]       # (B_flat, L)
                weights     = batch["weights"]      # (B_flat,)
                seq_lengths = batch["seq_lengths"]  # (B_flat,)

                # Build position mask: True at real residues, False at padding.
                # With batch_size=1 all sequences have the same L so this is all-True.
                # Matters when batch_size>1 and MSAs of different lengths are mixed.
                L_max = tokens.shape[1]
                pos = torch.arange(L_max, device=tokens.device).unsqueeze(0)  # (1, L)
                padding_mask = pos < seq_lengths.unsqueeze(1)                 # (B_flat, L)

                # Pass the *wrapped* model so DDP.forward() arms gradient
                # all-reduce and the Accelerate bf16 autocast patch applies.
                # vocab_size is passed explicitly because the wrapped module
                # does not expose `.vocab_size`.  One loss for all α (α=0 → SFM).
                loss = flow_loss(
                    model,
                    tokens,
                    msa_emb,
                    vocab_size=cfg.model.vocab_size,
                    alpha=alpha,
                    weights=weights,
                    padding_mask=padding_mask,
                    t_max_eps=cfg.training.get("t_max_eps", 0.01),
                )

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)

                optimizer.step()
                optimizer.zero_grad()

                if accelerator.sync_gradients:
                    scheduler.step()
                    global_step += 1
                    if ema is not None:
                        ema.update(_unwrap_model(model))

            epoch_loss += loss.detach().float()

            # ── Logging ──────────────────────────────────────────────────────
            if global_step % cfg.training.log_every == 0 and accelerator.is_main_process:
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    "epoch %d | step %d | loss %.4f | lr %.2e",
                    epoch, global_step, loss.item(), lr,
                )
                _log_metrics({"kind": "train", "step": global_step, "epoch": epoch,
                              "loss": loss.item(), "lr": lr})
                if use_wandb:
                    import wandb
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/lr": lr,
                        "train/grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                        "train/epoch": epoch,
                        "train/batch_idx": batch_idx,
                    }, step=global_step)

        # ── Validation ───────────────────────────────────────────────────────
        if val_every > 0 and (epoch + 1) % val_every == 0 and accelerator.is_main_process:
            per_protein = [
                validate_generation(
                    _unwrap_model(model),
                    vb,
                    vocab_size=cfg.model.vocab_size,
                    n_steps=cfg.training.get("val_ode_steps", 50),
                    noise_fn=noise_fn,
                    step_fn=step_fn,
                    decode_fn=decode_fn,
                )
                for vb in val_items
            ]
            if per_protein:
                val_metrics = {
                    k: float(np.mean([m[k] for m in per_protein]))
                    for k in per_protein[0]
                }
                # Teacher-forced flow-matching loss on val — directly comparable
                # to the train loss curve (same objective, no generation).
                val_metrics["val/loss"] = validate_loss(
                    _unwrap_model(model), val_items,
                    vocab_size=cfg.model.vocab_size, alpha=alpha,
                    t_max_eps=cfg.training.get("t_max_eps", 0.01),
                )
                t05 = "val/t05"
                logger.info(
                    "Epoch %d val (%d proteins) | loss=%.4f | t=0 recovery=%.3f gap=%.3f kl=%.4f"
                    " | t=0.5 recovery=%.3f gap=%.3f kl=%.4f",
                    epoch, len(val_items),
                    val_metrics["val/loss"],
                    val_metrics["val/seq_recovery"],
                    val_metrics["val/gap_frac"],
                    val_metrics["val/aa_kl_div"],
                    val_metrics.get(f"{t05}/seq_recovery", float("nan")),
                    val_metrics.get(f"{t05}/gap_frac", float("nan")),
                    val_metrics.get(f"{t05}/aa_kl_div", float("nan")),
                )
                _log_metrics({"kind": "val", "step": global_step, "epoch": epoch,
                              **{k: float(v) for k, v in val_metrics.items()}})
                if use_wandb:
                    import wandb
                    wandb.log(val_metrics, step=global_step)

            # ── Save sequence examples ────────────────────────────────────────
            n_ex_proteins = cfg.training.get("val_example_proteins", 3)
            n_ex_seqs     = cfg.training.get("val_example_seqs", 5)
            if n_ex_proteins > 0 and val_items:
                save_val_examples(
                    _unwrap_model(model),
                    val_items,
                    vocab_size=cfg.model.vocab_size,
                    epoch=epoch,
                    output_dir=output_dir,
                    n_proteins=n_ex_proteins,
                    n_seqs=n_ex_seqs,
                    n_steps=cfg.training.get("val_ode_steps", 50),
                    noise_fn=noise_fn,
                    step_fn=step_fn,
                    decode_fn=decode_fn,
                )
            model.train()

        # ── Checkpoint ───────────────────────────────────────────────────────
        if accelerator.is_main_process:
            unwrapped = _unwrap_model(model)
            ckpt = {
                "model": unwrapped.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                # α is not a weight (it only parameterises the Lᵖ geometry), so it
                # must be stored explicitly for inference to reconstruct the model
                # with the matching tangent projection / decode geometry.
                "alpha": alpha,
            }
            if ema is not None:
                ckpt["ema"] = ema.state_dict()
            # latest.pt: 매 epoch 덮어씀 (resume용, 단일 파일)
            torch.save(ckpt, output_dir / "latest.pt")
            # epoch_NNN.pt: save_checkpoint_every마다만 저장 (기본 1 = 매 epoch)
            # 오버피팅 디버그처럼 epoch 수가 많을 때는 50 등으로 설정해 디스크 절약
            save_every = cfg.training.get("save_checkpoint_every", 1)
            is_last = (epoch == cfg.training.epochs - 1)
            if (epoch + 1) % save_every == 0 or is_last:
                torch.save(ckpt, output_dir / f"epoch_{epoch:03d}.pt")
            avg_loss = epoch_loss.item() / len(loader)
            logger.info(
                "Epoch %d complete | avg loss %.4f",
                epoch,
                avg_loss,
            )
            logger.info("-" * 60)
            if use_wandb:
                import wandb
                wandb.log({
                    "epoch/avg_loss": avg_loss,
                    "epoch/epoch_num": epoch,
                    "epoch/learning_rate": scheduler.get_last_lr()[0],
                }, step=global_step)

    if accelerator.is_main_process:
        logger.info("Training complete.")
        if ema is not None:
            # Save EMA weights as the final model
            unwrapped = _unwrap_model(model)
            ema.copy_to(unwrapped)
            torch.save({"model": unwrapped.state_dict(), "alpha": alpha},
                       output_dir / "decoder_ema_final.pt")
        if use_wandb:
            import wandb
            wandb.finish()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",        required=True,  help="Path to decoder.yaml config")
    parser.add_argument("--lmdb_path",     default=None,   help="Override data.lmdb_path")
    parser.add_argument("--val_lmdb_path", default=None,   help="Override data.val_lmdb_path")
    parser.add_argument("--output_dir",    default=None,   help="Override training.output_dir")
    parser.add_argument("--alpha",         default=None, type=float,
                        help="Override model.alpha (α-geometry; 0.0=L² sphere/SFM, 0.5=L⁴/α-Flow)")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.lmdb_path:
        cfg.data.lmdb_path = args.lmdb_path
    if args.val_lmdb_path:
        cfg.data.val_lmdb_path = args.val_lmdb_path
    if args.output_dir:
        cfg.training.output_dir = args.output_dir
    if args.alpha is not None:
        cfg.model.alpha = args.alpha

    train(cfg)


if __name__ == "__main__":
    main()
