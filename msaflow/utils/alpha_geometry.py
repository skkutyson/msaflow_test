"""
α-geometry utilities for Continuous-State Discrete Flow Matching (α-Flow).

Generalises the α=0 spherical SFM geometry (``spherical.py``) to the full family
of α-geometries from:

    Cheng et al., "α-Flow: A Unified Framework for Continuous-State Discrete
    Flow Matching Models", arXiv:2504.10283, 2025.

Key objects (paper Def 3.1 and §3.2):

    p       = 2 / (1 - α)                       (α ∈ [-1, 1), p ∈ [1, ∞))
    π(α)    : µ ↦ x = µ^(1/p)                   α-representation  (Def 3.1, Eq. 3)
    π(α)⁻¹  : x ↦ µ = x^p / Σ x^p               inverse
    ‖x‖_p   = (Σ xᵢ^p)^(1/p) = 1               x lives on the positive orthant
                                                of the unit Lᵖ sphere Sᵖ

    α = 0   → p = 2, x = √µ  → the unit L² sphere = SFM / Fisher-Flow.
    α = 0.5 → p = 4, x = µ^(1/4) → the unit L⁴ sphere.
    α = -1  → p = 1, x = µ      → raw probabilities on the simplex.

The α-geodesic (Eq. 4) is the Lᵖ-normalised straight line between α-representations.
We parameterise it directly in t (τ(t) = t), which traces the same geodesic *curve*
as the paper but with a non-arc-length time parameterisation — a valid conditional
path for flow matching, with the exact analytic velocity below.  This avoids the
constant-speed reparameterisation τ (paper Appendix B) and every acos / 1/µ term,
so all maps are closed-form and numerically stable:

    xₜ            = normalise_p((1-t)·x₀ + t·x₁)                       (interp)
    uₜ = dxₜ/dt   = P_{xₜ}(x₁ - x₀) / ‖(1-t)·x₀ + t·x₁‖_p             (velocity)
    P_x(w)        = w - x · Σⱼ xⱼ^(p-1) wⱼ                            (tangent proj, Eq. 11)
    exp̃_x(u)      = normalise_p(x + u)                                (Euler step, Eq. 9)

At α = 0 (p = 2) P_x(w) = w - x·⟨x, w⟩ and normalise_p is L² normalisation, so
these reduce to the standard sphere operations.

All geometry runs in float32 regardless of autocast dtype, matching the stability
strategy used for the α=0 path in ``spherical.py``.
"""

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# α ↔ p and simplex ↔ Lᵖ-sphere mappings
# ─────────────────────────────────────────────────────────────────────────────

def p_from_alpha(alpha: float) -> float:
    """p = 2/(1-α). Defined for α ∈ [-1, 1); α=1 (logit rep) is not supported here."""
    if alpha >= 1.0:
        raise ValueError("alpha=1 (log representation) is not supported by this module")
    return 2.0 / (1.0 - alpha)


def simplex_to_alpha(mu: torch.Tensor, p: float, eps: float = 1e-8) -> torch.Tensor:
    """π(α): probability vectors → positive orthant of the unit Lᵖ sphere, x = µ^(1/p)."""
    return mu.clamp(min=eps).pow(1.0 / p)


def alpha_to_simplex(x: torch.Tensor, p: float, eps: float = 1e-8) -> torch.Tensor:
    """π(α)⁻¹: Lᵖ-sphere points → probability vectors, µ = xᵖ / Σ xᵖ."""
    mu = x.clamp(min=0.0).pow(p)
    return mu / mu.sum(dim=-1, keepdim=True).clamp(min=eps)


def onehot_to_alpha(tokens: torch.Tensor, vocab_size: int, p: float, eps: float = 1e-8) -> torch.Tensor:
    """Integer token ids → α-representation of the one-hot categorical (a Lᵖ-sphere vertex)."""
    one_hot = F.one_hot(tokens.long(), num_classes=vocab_size).float()
    return simplex_to_alpha(one_hot, p, eps=eps)


# ─────────────────────────────────────────────────────────────────────────────
# Lᵖ-sphere operations
# ─────────────────────────────────────────────────────────────────────────────

def lp_norm(v: torch.Tensor, p: float, eps: float = 1e-8) -> torch.Tensor:
    """‖v‖_p along the last dim, keepdim, clamped away from 0."""
    return v.abs().pow(p).sum(dim=-1, keepdim=True).clamp(min=eps).pow(1.0 / p)


def lp_normalize(v: torch.Tensor, p: float, eps: float = 1e-8) -> torch.Tensor:
    """Project a positive ambient vector onto the unit Lᵖ sphere: v / ‖v‖_p."""
    return v / lp_norm(v, p, eps=eps)


def tangent_project(w: torch.Tensor, x: torch.Tensor, p: float) -> torch.Tensor:
    """Orthogonal projection of w onto the tangent space of Sᵖ at x  (α-Flow Eq. 11).

        P_x(w) = w - x · Σⱼ xⱼ^(p-1) wⱼ

    Uses Σⱼ xⱼ^p = 1 so that P_x(x) = 0 (x is the radial/normal direction).
    At p=2 this is w - x·⟨x, w⟩.
    """
    coeff = (x.clamp(min=0.0).pow(p - 1.0) * w).sum(dim=-1, keepdim=True)
    return w - x * coeff


# ─────────────────────────────────────────────────────────────────────────────
# Conditional flow: interpolation, target velocity, noise
# ─────────────────────────────────────────────────────────────────────────────

def alpha_interpolate(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor, p: float,
                      eps: float = 1e-8) -> torch.Tensor:
    """α-geodesic interpolation (τ(t)=t): xₜ = normalise_p((1-t)·x₀ + t·x₁).

    t=0 → x₀ (noise), t=1 → x₁ (data); endpoints are exact since x₀, x₁ ∈ Sᵖ.
    """
    r = (1.0 - t) * x0 + t * x1
    return lp_normalize(r, p, eps=eps)


def alpha_target_velocity(x0: torch.Tensor, x1: torch.Tensor, x_t: torch.Tensor,
                          t: torch.Tensor, p: float, eps: float = 1e-8) -> torch.Tensor:
    """Exact conditional velocity dxₜ/dt of the τ(t)=t α-geodesic:

        uₜ = P_{xₜ}(x₁ - x₀) / ‖(1-t)·x₀ + t·x₁‖_p

    Derived by differentiating xₜ = r(t)/‖r(t)‖_p, r(t) = (1-t)x₀ + t·x₁; the
    normalisation derivative collapses exactly into the tangent projection P
    (α-Flow Eq. 11). Lies in the tangent space at xₜ by construction.
    """
    r_norm = lp_norm((1.0 - t) * x0 + t * x1, p, eps=eps)
    return tangent_project(x1 - x0, x_t, p) / r_norm


def sample_alpha_noise(shape: tuple, p: float, device=None, dtype=torch.float32) -> torch.Tensor:
    """Sample µ₀ ~ Dirichlet(1) (uniform on simplex) and map to Sᵖ via π(α)."""
    e = -torch.log(torch.rand(*shape, device=device, dtype=dtype).clamp(min=1e-8))
    mu = e / e.sum(dim=-1, keepdim=True)
    return simplex_to_alpha(mu, p)


# ─────────────────────────────────────────────────────────────────────────────
# α-metric (Riemannian norm for the flow-matching loss, α-Flow Eq. 12)
# ─────────────────────────────────────────────────────────────────────────────

def alpha_metric_sq(diff: torch.Tensor, x_t: torch.Tensor, alpha: float, p: float) -> torch.Tensor:
    """Squared α-norm of a tangent vector, summed over the vocab dim.

        ‖diff‖²_α ∝ Σᵢ diffᵢ² · µ_{t,i}^α ,   µ_{t,i} = x_{t,i}^p  ⇒  weight = x_{t,i}^(pα)

    Returns (..., ) after summing the last (vocab) dim.  The paper's global p²
    prefactor is dropped so the loss magnitude stays comparable to the α=0
    Euclidean MSE (the constant only rescales the effective learning rate).
    At α=0 the weight is 1 → plain summed squared error.
    """
    weight = x_t.clamp(min=0.0).pow(p * alpha)          # (..., V)
    return (diff * diff * weight).sum(dim=-1)           # (...,)


# ─────────────────────────────────────────────────────────────────────────────
# Inference: Euler step on the Lᵖ sphere + decoding
# ─────────────────────────────────────────────────────────────────────────────

def alpha_euler_step(x_t: torch.Tensor, v: torch.Tensor, dt: float, p: float,
                     eps: float = 1e-8) -> torch.Tensor:
    """One Euler step along tangent v then re-project to Sᵖ  (exp̃, α-Flow Eq. 9):

        x_{t+dt} = normalise_p(xₜ + v·dt)
    """
    return lp_normalize(x_t + v * dt, p, eps=eps)


def decode_alpha(x: torch.Tensor, alpha: float, p: float, temperature: float = 1.0) -> torch.Tensor:
    """Decode Lᵖ-sphere points to token ids via µ = π(α)⁻¹(x)."""
    mu = alpha_to_simplex(x, p)
    if temperature <= 0.0:
        return mu.argmax(dim=-1)
    logits = torch.log(mu.clamp(min=1e-8)) / temperature
    return torch.distributions.Categorical(logits=logits).sample()


# ─────────────────────────────────────────────────────────────────────────────
# Geometry op factory (shared by training, validation, and inference)
# ─────────────────────────────────────────────────────────────────────────────

def geometry_ops(alpha: float):
    """Return (noise_fn, euler_step_fn, decode_fn) for the α-Flow Lᵖ sphere,
    p = 2/(1-α).  A single path for all α — α=0 gives p=2 (the L² sphere / SFM),
    α=0.5 gives p=4, etc.  The three fns share a geometry-agnostic signature so
    generation/validation code is identical regardless of α:

        noise_fn(shape, device, dtype) -> x0 on Sᵖ
        euler_step_fn(x, v, dt)        -> next point on Sᵖ
        decode_fn(x, temperature)      -> token ids
    """
    p = p_from_alpha(alpha)
    return (
        lambda shape, device, dtype: sample_alpha_noise(shape, p, device=device, dtype=dtype),
        lambda x, v, dt: alpha_euler_step(x, v, dt, p),
        lambda x, temperature: decode_alpha(x, alpha, p, temperature=temperature),
    )
