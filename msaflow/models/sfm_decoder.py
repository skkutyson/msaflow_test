"""
MSAFlow SFM Decoder — Statistical Flow Matching decoder for MSA sequence generation.

Architecture (from paper, Section 3.1.2):
  • 12 × DiT blocks, hidden dim 768, 12 attention heads  (~130M params)
  • Input projection:      FC(22 → 768)          [22 = 20 AA + gap + X]
  • Conditioning proj:     FC(128 → 768)          [per-residue MSA embedding]
  • Time embedding:        sinusoidal → MLP → 768
  • Position-wise AdaLN:   scale/shift computed per residue from (t_emb + cond)
  • Output projection:     FC(768 → 22)

The key innovation over vanilla DiT is **position-wise AdaLN**: each residue
gets its own (shift, scale, gate) derived from the compressed MSA embedding
at that position, enabling fine-grained evolutionary conditioning.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (no mean centering, learnable scale)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x.float() / rms * self.weight).to(x.dtype)


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding → MLP, output dim = hidden_size."""

    def __init__(self, hidden_size: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def _sinusoidal(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)   # (B, half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) → (B, hidden_size)."""
        t_freq = self._sinusoidal(t, self.freq_dim)
        return self.mlp(t_freq)


# ──────────────────────────────────────────────────────────────────────────────
# Position-wise AdaLN modulation
# ──────────────────────────────────────────────────────────────────────────────

def modulate_poswise(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply position-wise adaptive layer norm modulation.

    x:     (B, L, H)
    shift: (B, L, H)
    scale: (B, L, H)
    """
    return x * (1.0 + scale) + shift


class PosWiseAdaLNBlock(nn.Module):
    """DiT block with position-wise AdaLN conditioning.

    Unlike standard DiT where conditioning is global (one scale/shift for all
    tokens), here each residue position is modulated by its own scale/shift
    derived from the per-position conditioning vector c ∈ R^(B×L×H).
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )

        # Position-wise modulation: c (B,L,H) → 6 × (B,L,H)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )
        # Zero-init so at the start of training, blocks are identity
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: (B, L, H)  — noisy sequence representation
        c: (B, L, H)  — per-position conditioning (t_emb broadcast + cond_proj)
        key_padding_mask: (B, L) bool — True at padded positions to exclude from
                          attention (matches nn.MultiheadAttention convention).
        """
        mod = self.adaLN_modulation(c)          # (B, L, 6*H)
        shift_sa, scale_sa, gate_sa, shift_ff, scale_ff, gate_ff = mod.chunk(6, dim=-1)

        # Self-attention with position-wise AdaLN pre-norm
        h = modulate_poswise(self.norm1(x), shift_sa, scale_sa)
        attn_out, _ = self.attn(
            h, h, h, key_padding_mask=key_padding_mask, need_weights=False
        )
        x = x + gate_sa * attn_out

        # Feed-forward with position-wise AdaLN pre-norm
        h = modulate_poswise(self.norm2(x), shift_ff, scale_ff)
        x = x + gate_ff * self.mlp(h)

        return x


# ──────────────────────────────────────────────────────────────────────────────
# Final projection layer
# ──────────────────────────────────────────────────────────────────────────────

class FinalLayer(nn.Module):
    """Final position-wise AdaLN + linear output layer."""

    def __init__(self, hidden_size: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        mod = self.adaLN_modulation(c)          # (B, L, 3H)
        shift, scale, gate = mod.chunk(3, dim=-1)

        x = modulate_poswise(self.norm(x), shift, scale)  # (B, L, H)
        x = gate * x                                       # gate in hidden space
        return self.linear(x)                              # (B, L, out_dim)


# ──────────────────────────────────────────────────────────────────────────────
# Main SFM Decoder
# ──────────────────────────────────────────────────────────────────────────────

class SFMDecoder(nn.Module):
    """
    Statistical Flow Matching decoder for MSA sequence generation.

    Given a compressed MSA embedding m_seq ∈ R^(B×L×msa_dim) and a noisy
    sequence x_t ∈ R^(B×L×vocab_size) on the unit sphere, predict the
    velocity field v_θ(x_t | m_seq, t) ∈ R^(B×L×vocab_size).

    Args:
        vocab_size:  V = 22 (20 AA + gap + X).
        msa_dim:     Dimension of compressed MSA embedding (128 from AF3).
        hidden_size: Transformer hidden dim (768).
        depth:       Number of transformer blocks (12).
        num_heads:   Attention heads (12).
        mlp_ratio:   FFN expansion ratio (4.0).
        max_seq_len: Maximum sequence length for sinusoidal positional encoding.
    """

    def __init__(
        self,
        vocab_size: int = 22,
        msa_dim: int = 128,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        max_seq_len: int = 2048,
        alpha: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        # α-Flow geometry: α=0 → unit L² sphere (SFM); α=0.5 → unit L⁴ sphere.
        # p = 2/(1-α) sets the Lᵖ-tangent projection applied to the output velocity.
        self.alpha = alpha
        self.p = 2.0 / (1.0 - alpha)

        # ── Input projection ──────────────────────────────────────────────────
        self.input_proj = nn.Linear(vocab_size, hidden_size)

        # ── Positional encoding (fixed sin-cos) ───────────────────────────────
        self.register_buffer(
            "pos_emb",
            self._build_sincos_pos_emb(max_seq_len, hidden_size),
            persistent=False,
        )

        # ── Time embedding ────────────────────────────────────────────────────
        self.time_emb = SinusoidalTimeEmbedding(hidden_size)

        # ── Conditioning projection (per-residue MSA emb → hidden) ────────────
        # RMSNorm normalises AF3 MSA embeddings (std >> 1) before projection.
        # Without it ∂L/∂W_cond inherits m_seq scale → cond_proj dominates
        # total grad_norm and forces an effectively near-zero LR after clipping.
        # RMSNorm preferred over LayerNorm: no mean-centering, stable on sparse inputs.
        self.msa_norm = RMSNorm(msa_dim)
        self.cond_proj = nn.Linear(msa_dim, hidden_size)

        # ── Transformer blocks ────────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            PosWiseAdaLNBlock(hidden_size, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

        # Per-block conditioning offsets: each block gets a residual on the
        # shared cond vector, allowing the network to route different conditioning
        # signals to different layers. Zero-inited → identity at init.
        self.block_cond_projs = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size, bias=False)
            for _ in range(depth)
        ])

        # ── Output ────────────────────────────────────────────────────────────
        self.final_layer = FinalLayer(hidden_size, vocab_size)

        self._init_weights()

    # ── Weight initialisation ─────────────────────────────────────────────────

    def _init_weights(self):
        def _basic(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(_basic)
        # Time MLP
        nn.init.normal_(self.time_emb.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_emb.mlp[2].weight, std=0.02)
        # block_cond_projs: zero-init so blocks start as identity conditioning
        for proj in self.block_cond_projs:
            nn.init.zeros_(proj.weight)

    @staticmethod
    def _build_sincos_pos_emb(max_len: int, dim: int) -> torch.Tensor:
        pos = torch.arange(max_len).unsqueeze(1).float()
        i = torch.arange(dim // 2).float()
        angle = pos / (10000 ** (2 * i / dim))
        emb = torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros(max_len, 1)], dim=-1)
        return emb.unsqueeze(0)  # (1, max_len, dim)

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(
        self,
        x_t: torch.Tensor,
        m_seq: torch.Tensor,
        t: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x_t:   (B, L, vocab_size) — noisy sequence on the unit sphere.
            m_seq: (B, L, msa_dim)    — compressed MSA embedding per residue.
            t:     (B,)               — timesteps in [0, 1].
            padding_mask: (B, L) bool — True at real residues, False at padding.
                          When None (single-MSA batches), no positions are masked.

        Returns:
            v: (B, L, vocab_size) — predicted velocity in tangent space.
        """
        B, L, _ = x_t.shape

        # nn.MultiheadAttention expects True = "ignore this key", so invert.
        key_padding_mask = (~padding_mask) if padding_mask is not None else None

        # Project input sequence to hidden space + positional encoding
        h = self.input_proj(x_t) + self.pos_emb[:, :L, :]       # (B, L, H)

        # Build per-position conditioning: time emb (broadcast) + MSA cond
        t_emb = self.time_emb(t).unsqueeze(1).expand(B, L, -1)  # (B, L, H)
        cond = t_emb + self.cond_proj(self.msa_norm(m_seq))      # (B, L, H)

        # Transformer blocks with position-wise AdaLN
        for block, cond_proj in zip(self.blocks, self.block_cond_projs):
            h = block(h, cond + cond_proj(cond), key_padding_mask=key_padding_mask)

        # Final projection back to vocab space
        v = self.final_layer(h, cond)                            # (B, L, vocab_size)

        # Project onto the tangent space of the Lᵖ sphere at x_t (α-Flow Eq. 11):
        #   P_x(v) = v - x · Σⱼ xⱼ^(p-1) vⱼ
        # At α=0 (p=2) this is the standard v - <v, x_t>·x_t.
        v32 = v.float()
        x_t_f32 = x_t.float()
        coeff = (x_t_f32.clamp(min=0.0).pow(self.p - 1.0) * v32).sum(dim=-1, keepdim=True)
        v_tangent = v32 - x_t_f32 * coeff
        return v_tangent.to(v.dtype)                             # (B, L, vocab_size)


# ──────────────────────────────────────────────────────────────────────────────
# Flow-matching training loss (unified α-Flow; SFM = the α=0 case)
# ──────────────────────────────────────────────────────────────────────────────

def flow_loss(
    model: SFMDecoder,
    tokens: torch.Tensor,
    m_seq: torch.Tensor,
    vocab_size: Optional[int] = None,
    alpha: float = 0.0,
    weights: Optional[torch.Tensor] = None,
    padding_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
    t_max_eps: float = 0.01,
) -> torch.Tensor:
    """
    Unified α-Flow training loss on the unit Lᵖ sphere, p = 2/(1-α)
    (Cheng et al., "α-Flow", arXiv:2504.10283, Algorithm 1).

        L(α) = E_{t~U[0,1-t_max_eps], x0, x1}[ ‖v_θ(x_t|m,t) − u_t(x_t|x0,x1)‖²_α ]

    ONE code path for every geometry: α=0 (p=2, unit L² sphere) is the special
    case that recovers SFM; α=0.5 (p=4) is the L⁴ sphere; etc.  All geometry
    lives in `msaflow.utils.alpha_geometry`.

    Note vs. the original SFM: this uses the α-Flow "simplified maps" (Eq. 9,10) —
    the Lᵖ-normalised straight-line α-geodesic parameterised by τ(t)=t.  At α=0
    this traces the same great circle as the original acos/sin SFM but with a
    non-arc-length time parameterisation: a valid, equivalent-family conditional
    path, not bit-identical to the old great-circle formulation.

    Stability: x0, x1, x_t, u_t and the squared error are all in float32
    regardless of autocast dtype; t is truncated to [0, 1-t_max_eps]; v_pred is
    upcast and re-projected to the tangent at x_t before the α-weighted norm.

    Args:
        model:      SFMDecoder, or the DDP/Accelerate-wrapped module around it.
                    Pass the *wrapped* model so DDP.forward() arms gradient
                    all-reduce and the Accelerate bf16 autocast applies; then pass
                    `vocab_size` explicitly (the wrapper hides `.vocab_size`).
        tokens:     (B, L) integer token ids in [0, vocab_size).
        m_seq:      (B, L, msa_dim) compressed MSA embedding.
        vocab_size: V. Falls back to `model.vocab_size` when None (unwrapped model).
        alpha:      α-geometry parameter in [-1, 1). 0 → L² sphere (SFM).
        weights:    (B,) optional per-sample Neff weights.
        padding_mask: (B, L) bool, True at real residues (masked from BOTH the loss
                      and self-attention).
        eps, t_max_eps: numerical margins.

    Returns:
        loss: scalar.
    """
    from msaflow.utils.alpha_geometry import (
        p_from_alpha, onehot_to_alpha, sample_alpha_noise,
        alpha_interpolate, alpha_target_velocity, tangent_project, alpha_metric_sq,
    )

    B, L = tokens.shape
    device = tokens.device
    V = vocab_size if vocab_size is not None else model.vocab_size
    p = p_from_alpha(alpha)

    # All geometry in float32 for stability (independent of autocast dtype).
    x1 = onehot_to_alpha(tokens, V, p, eps=eps)                        # (B, L, V) f32
    x0 = sample_alpha_noise((B, L, V), p, device=device, dtype=torch.float32)

    t = torch.rand(B, device=device) * (1.0 - t_max_eps)              # (B,) ∈ [0, 1-t_max_eps]
    t_bcast = t.view(B, 1, 1)

    x_t = alpha_interpolate(x0, x1, t_bcast, p, eps=eps)              # (B, L, V)
    u_t = alpha_target_velocity(x0, x1, x_t, t_bcast, p, eps=eps)     # (B, L, V)

    # padding_mask also flows into attention so padded keys don't contaminate
    # real positions (relevant when batch_size>1 packs variable-length MSAs).
    v_pred = model(x_t, m_seq, t, padding_mask=padding_mask)         # (B, L, V)
    v_pred = tangent_project(v_pred.float(), x_t, p)                  # fp32, tangent at x_t

    loss = alpha_metric_sq(v_pred - u_t, x_t, alpha, p)              # (B, L)  α-weighted
    return _reduce_perpos_loss(loss, weights, padding_mask, B, L)


def sfm_loss(
    model: SFMDecoder,
    tokens: torch.Tensor,
    m_seq: torch.Tensor,
    vocab_size: Optional[int] = None,
    weights: Optional[torch.Tensor] = None,
    padding_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
    t_max_eps: float = 0.01,
) -> torch.Tensor:
    """Backward-compat alias: the α=0 case of `flow_loss` (unit L² sphere)."""
    return flow_loss(model, tokens, m_seq, vocab_size=vocab_size, alpha=0.0,
                     weights=weights, padding_mask=padding_mask, eps=eps, t_max_eps=t_max_eps)


def _reduce_perpos_loss(loss, weights, padding_mask, B, L):
    """Mask padded positions, apply per-sample weights, and normalise by the
    effective count so the loss scale is invariant to length / batch / weighting.
    loss: (B, L)."""
    if padding_mask is not None:
        loss = loss * padding_mask.float()
    if weights is not None:
        loss = loss * weights.unsqueeze(1)

    if padding_mask is not None:
        if weights is not None:
            n_real = (weights.unsqueeze(1) * padding_mask.float()).sum().clamp(min=1.0)
        else:
            n_real = padding_mask.float().sum().clamp(min=1.0)
    else:
        if weights is not None:
            n_real = weights.sum().clamp(min=1.0) * L
        else:
            n_real = float(B * L)
    return loss.sum() / n_real


def alpha_flow_loss(
    model: SFMDecoder,
    tokens: torch.Tensor,
    m_seq: torch.Tensor,
    vocab_size: Optional[int] = None,
    alpha: float = 0.5,
    weights: Optional[torch.Tensor] = None,
    padding_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
    t_max_eps: float = 0.01,
) -> torch.Tensor:
    """Backward-compat alias for `flow_loss` with an explicit α (default 0.5)."""
    return flow_loss(model, tokens, m_seq, vocab_size=vocab_size, alpha=alpha,
                     weights=weights, padding_mask=padding_mask, eps=eps, t_max_eps=t_max_eps)
