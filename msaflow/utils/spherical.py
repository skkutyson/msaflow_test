"""
Spherical geometry utilities for Statistical Flow Matching (SFM).

SFM operates on the positive orthant of the unit sphere via the mapping:
    π : µ → x = √µ   (µ ∈ Δ^|A|, probability simplex)
    π⁻¹: x → µ = x²  (x on unit sphere)

This preserves the Fisher-Rao metric so geodesics on the sphere correspond
to geodesics on the statistical manifold.

Reference: Cheng et al., "Categorical flow matching on statistical manifolds", 2024.
"""

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Sphere <-> simplex mappings
# ─────────────────────────────────────────────────────────────────────────────

def simplex_to_sphere(mu: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Map probability vectors (..., V) to the positive orthant of the unit sphere, x = √µ."""
    return torch.sqrt(mu.clamp(min=eps))


def sphere_to_simplex(x: torch.Tensor) -> torch.Tensor:
    """Map sphere points (..., V) back to probability vectors, µ = x² (re-normalised for fp drift)."""
    mu = x ** 2
    return mu / mu.sum(dim=-1, keepdim=True).clamp(min=1e-8)


def onehot_to_sphere(tokens: torch.Tensor, vocab_size: int, eps: float = 1e-8) -> torch.Tensor:
    """Integer token ids (..., L) → sphere points (..., L, V) via one-hot → √."""
    one_hot = F.one_hot(tokens.long(), num_classes=vocab_size).float()
    return simplex_to_sphere(one_hot, eps=eps)


# ─────────────────────────────────────────────────────────────────────────────
# Spherical exponential and logarithmic maps
# ─────────────────────────────────────────────────────────────────────────────

def exp_map(x: torch.Tensor, u: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Spherical exponential map at x along tangent u (..., V):

        exp_x(u) = x·cos(‖u‖) + (u/‖u‖)·sin(‖u‖)
    """
    norm_u = torch.norm(u, dim=-1, keepdim=True).clamp(min=eps)
    return x * torch.cos(norm_u) + (u / norm_u) * torch.sin(norm_u)


def log_map(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Spherical logarithmic map: tangent vector at x pointing toward y (..., V):

        log_x(y) = arccos(<x,y>) / √(1 − <x,y>²) · (y − <x,y>·x)
    """
    dot = (x * y).sum(dim=-1, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)
    theta = torch.acos(dot)                              # geodesic distance
    sin_theta = torch.sqrt(1.0 - dot ** 2).clamp(min=eps)
    direction = y - dot * x                              # component orthogonal to x
    return (theta / sin_theta) * direction


# ─────────────────────────────────────────────────────────────────────────────
# Geodesic interpolation and velocity field
# ─────────────────────────────────────────────────────────────────────────────

def geodesic_interpolate(
    x0: torch.Tensor,
    x1: torch.Tensor,
    t: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Spherical geodesic interpolation, x_t = exp_{x0}(t · log_{x0}(x1)).

    x0 (..., V) noise, x1 (..., V) data, t (..., 1) or scalar in [0,1].
    t=0 → x0, t=1 → x1. Returns x_t (..., V).
    """
    u = log_map(x0, x1, eps=eps)   # tangent direction from x0 toward x1
    return exp_map(x0, t * u, eps=eps)


def target_velocity(
    x_t: torch.Tensor,
    x1: torch.Tensor,
    t: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Conditional target velocity field for SFM training (tangent at x_t):

        u_t(x_t | x1) = log_{x_t}(x1) / (1 − t)

    x_t, x1 (..., V); t (..., 1) in [0, 1). Returns u (..., V).
    """
    log = log_map(x_t, x1, eps=eps)
    denom = (1.0 - t).clamp(min=eps)
    return log / denom


# ─────────────────────────────────────────────────────────────────────────────
# Noise sampling on the positive orthant of the unit sphere
# ─────────────────────────────────────────────────────────────────────────────

def sample_sphere_noise(shape: tuple, device=None, dtype=torch.float32) -> torch.Tensor:
    """Sample µ_0 ~ Dirichlet(1) (uniform on simplex) and map through π to the
    positive orthant of the unit sphere. Returns (*shape) with positive components."""
    noise = -torch.log(torch.rand(*shape, device=device, dtype=dtype).clamp(min=1e-8))
    mu = noise / noise.sum(dim=-1, keepdim=True)   # normalise to simplex
    return simplex_to_sphere(mu)


# ─────────────────────────────────────────────────────────────────────────────
# Inference ODE step (Euler on sphere)
# ─────────────────────────────────────────────────────────────────────────────

def euler_step_sphere(
    x_t: torch.Tensor,
    v: torch.Tensor,
    dt: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """One Euler step on the sphere, x_{t+dt} = exp_{x_t}(v · dt).

    x_t (..., V) current point, v (..., V) tangent velocity, dt step size.
    """
    return exp_map(x_t, v * dt, eps=eps)


def decode_sequences(x1: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Decode sphere points x1 (..., L, V) to token ids (..., L).

    temperature: 0 = argmax, 1.0 = unmodified, >1 = diverse.
    """
    mu = sphere_to_simplex(x1)      # (..., L, V) probabilities
    if temperature <= 0.0:
        return mu.argmax(dim=-1)
    logits = torch.log(mu.clamp(min=1e-8)) / temperature
    return torch.distributions.Categorical(logits=logits).sample()
