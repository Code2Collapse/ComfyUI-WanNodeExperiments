"""
freq_utils.py — frequency-domain helpers (FreeInit) + 3D spatial filters.

The FreeInit low-pass filters and ``freq_mix_3d`` reproduce the canonical
implementation from TianxingWu/FreeInit (arXiv:2312.07537), here vectorized
instead of the original triple ``for`` loops so they run on large video latents.

Also provides separable 3D Gaussian blur and a 3D Laplacian high-frequency
extractor used by the High-Frequency Latent Injector (HFLI) and the 3D
anti-alias latent upscaler.
"""

from __future__ import annotations

import math

import torch
import torch.fft as fft
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# FreeInit low-pass filters (vectorized)
# --------------------------------------------------------------------------- #
def _norm_coords(T: int, H: int, W: int, d_s: float, d_t: float, device, dtype):
    """Return d_square grid of shape (T, H, W) matching the FreeInit formula."""
    eps = 1e-8
    tt = (2.0 * torch.arange(T, device=device, dtype=dtype) / max(T, 1) - 1.0)
    hh = (2.0 * torch.arange(H, device=device, dtype=dtype) / max(H, 1) - 1.0)
    ww = (2.0 * torch.arange(W, device=device, dtype=dtype) / max(W, 1) - 1.0)
    ratio = d_s / (d_t + eps)
    d_square = (
        (ratio * tt).pow(2)[:, None, None]
        + hh.pow(2)[None, :, None]
        + ww.pow(2)[None, None, :]
    )
    return d_square


def gaussian_low_pass_filter(shape, d_s=0.25, d_t=0.25, device="cpu", dtype=torch.float32):
    T, H, W = shape[-3], shape[-2], shape[-1]
    if d_s == 0 or d_t == 0:
        return torch.zeros(shape, device=device, dtype=dtype)
    d_square = _norm_coords(T, H, W, d_s, d_t, device, dtype)
    mask = torch.exp(-1.0 / (2.0 * d_s ** 2) * d_square)
    return mask.expand(shape).contiguous()


def butterworth_low_pass_filter(shape, n=4, d_s=0.25, d_t=0.25, device="cpu", dtype=torch.float32):
    T, H, W = shape[-3], shape[-2], shape[-1]
    if d_s == 0 or d_t == 0:
        return torch.zeros(shape, device=device, dtype=dtype)
    d_square = _norm_coords(T, H, W, d_s, d_t, device, dtype)
    mask = 1.0 / (1.0 + (d_square / (d_s ** 2)).pow(n))
    return mask.expand(shape).contiguous()


def ideal_low_pass_filter(shape, d_s=0.25, d_t=0.25, device="cpu", dtype=torch.float32):
    T, H, W = shape[-3], shape[-2], shape[-1]
    if d_s == 0 or d_t == 0:
        return torch.zeros(shape, device=device, dtype=dtype)
    d_square = _norm_coords(T, H, W, d_s, d_t, device, dtype)
    mask = (d_square <= d_s * 2).to(dtype)
    return mask.expand(shape).contiguous()


def box_low_pass_filter(shape, d_s=0.25, d_t=0.25, device="cpu", dtype=torch.float32):
    T, H, W = shape[-3], shape[-2], shape[-1]
    mask = torch.zeros(shape, device=device, dtype=dtype)
    if d_s == 0 or d_t == 0:
        return mask
    threshold_s = round(int(H // 2) * d_s)
    threshold_t = round(T // 2 * d_t)
    cf, cr, cc = T // 2, H // 2, W // 2
    mask[..., cf - threshold_t:cf + threshold_t,
         cr - threshold_s:cr + threshold_s,
         cc - threshold_s:cc + threshold_s] = 1.0
    return mask


def get_freq_filter(shape, device, filter_type, n, d_s, d_t, dtype=torch.float32):
    if filter_type == "gaussian":
        return gaussian_low_pass_filter(shape, d_s, d_t, device, dtype)
    if filter_type == "butterworth":
        return butterworth_low_pass_filter(shape, n, d_s, d_t, device, dtype)
    if filter_type == "ideal":
        return ideal_low_pass_filter(shape, d_s, d_t, device, dtype)
    if filter_type == "box":
        return box_low_pass_filter(shape, d_s, d_t, device, dtype)
    raise ValueError(f"unknown filter_type: {filter_type}")


def freq_mix_3d(x, noise, LPF):
    """Mix low freq of ``x`` with high freq of ``noise`` (FreeInit reinit).

    ``x`` and ``noise`` are (..., T, H, W). ``LPF`` is the low-pass mask.
    """
    x_freq = fft.fftshift(fft.fftn(x.float(), dim=(-3, -2, -1)), dim=(-3, -2, -1))
    noise_freq = fft.fftshift(fft.fftn(noise.float(), dim=(-3, -2, -1)), dim=(-3, -2, -1))
    HPF = 1.0 - LPF
    mixed = x_freq * LPF + noise_freq * HPF
    mixed = fft.ifftshift(mixed, dim=(-3, -2, -1))
    out = fft.ifftn(mixed, dim=(-3, -2, -1)).real
    return out.to(x.dtype)


# --------------------------------------------------------------------------- #
# Separable 3D Gaussian blur + Laplacian high-frequency extraction
# --------------------------------------------------------------------------- #
def _gaussian_kernel_1d(sigma: float, device, dtype):
    sigma = max(float(sigma), 1e-6)
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k = k / k.sum()
    return k, radius


def gaussian_blur_3d(x, spatial_sigma=1.0, temporal_sigma=0.0):
    """Separable 3D Gaussian blur on a latent ``[B, C, T, H, W]``.

    ``temporal_sigma <= 0`` skips the temporal pass (purely spatial blur).
    Handles 4D ``[B, C, H, W]`` by treating it as a single frame.
    """
    squeeze_t = False
    if x.dim() == 4:
        x = x.unsqueeze(2)  # [B,C,1,H,W]
        squeeze_t = True
    B, C, T, H, W = x.shape
    device, dtype = x.device, x.dtype

    if temporal_sigma and temporal_sigma > 0 and T > 1:
        kt, rt = _gaussian_kernel_1d(temporal_sigma, device, dtype)
        kt = kt.view(1, 1, -1, 1, 1).expand(C, 1, -1, 1, 1)
        x = F.conv3d(F.pad(x, (0, 0, 0, 0, rt, rt), mode="replicate"), kt, groups=C)

    ks, rs = _gaussian_kernel_1d(spatial_sigma, device, dtype)
    kh = ks.view(1, 1, 1, -1, 1).expand(C, 1, 1, -1, 1)
    kw = ks.view(1, 1, 1, 1, -1).expand(C, 1, 1, 1, -1)
    x = F.conv3d(F.pad(x, (0, 0, rs, rs, 0, 0), mode="replicate"), kh, groups=C)
    x = F.conv3d(F.pad(x, (rs, rs, 0, 0, 0, 0), mode="replicate"), kw, groups=C)

    if squeeze_t:
        x = x.squeeze(2)
    return x


def laplacian_highfreq_3d(x, spatial_sigma=1.0, temporal_sigma=0.0):
    """High-frequency residual = x − Gaussian-blur(x). The HFLI texture signal."""
    return x - gaussian_blur_3d(x, spatial_sigma, temporal_sigma)


def edge_aware_blur_3d(x, spatial_sigma=1.0, temporal_sigma=0.0, threshold=0.05):
    """Anti-alias blur that preserves strong edges (D3).

    Blends fully-blurred and original by a per-voxel weight derived from local
    gradient magnitude: flat regions get blurred (anti-aliased), edges stay sharp.
    """
    blurred = gaussian_blur_3d(x, spatial_sigma, temporal_sigma)
    if threshold <= 0:
        return blurred
    # gradient magnitude across spatial dims, averaged over channels
    gx = torch.zeros_like(x)
    gy = torch.zeros_like(x)
    gx[..., :, :-1] = x[..., :, 1:] - x[..., :, :-1]
    gy[..., :-1, :] = x[..., 1:, :] - x[..., :-1, :]
    grad = (gx.abs() + gy.abs()).mean(dim=1, keepdim=True)
    # edge weight in [0,1]: 1 = keep original (edge), 0 = use blur (flat)
    edge_w = torch.clamp(grad / (threshold + 1e-8), 0.0, 1.0)
    return edge_w * x + (1.0 - edge_w) * blurred
