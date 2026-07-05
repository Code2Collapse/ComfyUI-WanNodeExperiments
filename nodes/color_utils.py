"""
color_utils.py — colour matching, YCbCr conversion, wavelet subband locking.

All functions operate on ComfyUI ``IMAGE`` tensors ``[B, H, W, C]`` in 0..1.

* ``color_match``  : reinhard (mean/std), mkl (Monge-Kantorovich linear, Pitié),
                     hm-mvgd (per-channel histogram match + MVGD polish).
* ``rgb_to_ycbcr`` / ``ycbcr_to_rgb`` : BT.601 full-range, used by the YUV lock.
* ``wavelet_lock`` : keep low-frequency wavelet subbands continuous with a
                     reference frame to kill tiling/chunk colour seams
                     (WF-VAE style, arXiv:2411.17459).

``PyWavelets`` (``pywt``) is a hard requirement (see requirements.txt) and is
imported at module top.
"""

from __future__ import annotations

import numpy as np
import torch
import pywt


# --------------------------------------------------------------------------- #
# YCbCr (BT.601, full range)
# --------------------------------------------------------------------------- #
_RGB2YCC = torch.tensor(
    [[0.299, 0.587, 0.114],
     [-0.168736, -0.331264, 0.5],
     [0.5, -0.418688, -0.081312]]
)
_YCC2RGB = torch.tensor(
    [[1.0, 0.0, 1.402],
     [1.0, -0.344136, -0.714136],
     [1.0, 1.772, 0.0]]
)
_YCC_OFF = torch.tensor([0.0, 0.5, 0.5])


def rgb_to_ycbcr(img: torch.Tensor) -> torch.Tensor:
    """[B,H,W,3] RGB 0..1 -> [B,H,W,3] YCbCr (Y,Cb,Cr) 0..1."""
    m = _RGB2YCC.to(img.device, img.dtype)
    off = _YCC_OFF.to(img.device, img.dtype)
    return torch.tensordot(img, m.T, dims=1) + off


def ycbcr_to_rgb(img: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`rgb_to_ycbcr`."""
    m = _YCC2RGB.to(img.device, img.dtype)
    off = _YCC_OFF.to(img.device, img.dtype)
    return torch.tensordot(img - off, m.T, dims=1).clamp(0.0, 1.0)


# --------------------------------------------------------------------------- #
# Colour matching
# --------------------------------------------------------------------------- #
def _sqrtm_psd(mat: torch.Tensor) -> torch.Tensor:
    """Symmetric PSD matrix square root via eigendecomposition."""
    mat = 0.5 * (mat + mat.transpose(-1, -2))
    vals, vecs = torch.linalg.eigh(mat)
    vals = torch.clamp(vals, min=0.0)
    return (vecs * vals.sqrt().unsqueeze(-2)) @ vecs.transpose(-1, -2)


def _flatten_pixels(img: torch.Tensor) -> torch.Tensor:
    return img.reshape(-1, img.shape[-1]).float()


def _reinhard(target: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    t = _flatten_pixels(target)
    r = _flatten_pixels(ref)
    tm, ts = t.mean(0), t.std(0) + 1e-6
    rm, rs = r.mean(0), r.std(0) + 1e-6
    out = (t - tm) / ts * rs + rm
    return out.reshape(target.shape)


def _mkl(target: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Monge-Kantorovich linear (Gaussian) colour transfer (Pitié 2007)."""
    t = _flatten_pixels(target)
    r = _flatten_pixels(ref)
    tm, rm = t.mean(0), r.mean(0)
    tc = torch.cov((t - tm).T) + 1e-5 * torch.eye(t.shape[1], device=t.device)
    rc = torch.cov((r - rm).T) + 1e-5 * torch.eye(r.shape[1], device=r.device)
    tc_half = _sqrtm_psd(tc)
    tc_half_inv = torch.linalg.inv(tc_half)
    mid = _sqrtm_psd(tc_half @ rc @ tc_half)
    transform = tc_half_inv @ mid @ tc_half_inv
    out = (t - tm) @ transform.T + rm
    return out.reshape(target.shape)


def _hist_match_channel(t: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Exact 1D histogram matching via sorted-rank quantile mapping."""
    t_sorted, t_idx = torch.sort(t)
    r_sorted, _ = torch.sort(r)
    n_t = t.numel()
    n_r = r.numel()
    # map each target rank to the corresponding reference quantile
    pos = torch.linspace(0, 1, n_t, device=t.device)
    r_pos = torch.linspace(0, 1, n_r, device=r.device)
    mapped_sorted = torch.from_numpy(
        np.interp(pos.cpu().numpy(), r_pos.cpu().numpy(), r_sorted.cpu().numpy())
    ).to(t.device, t.dtype)
    out = torch.empty_like(t)
    out[t_idx] = mapped_sorted
    return out


def _hm_mvgd(target: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Per-channel histogram match, then MVGD (=MKL) covariance polish."""
    t = _flatten_pixels(target)
    r = _flatten_pixels(ref)
    hm = torch.stack([_hist_match_channel(t[:, c], r[:, c]) for c in range(t.shape[1])], dim=1)
    polished = _mkl(hm.reshape(target.shape), ref)
    return polished


def color_match(target: torch.Tensor, ref: torch.Tensor, method: str = "reinhard") -> torch.Tensor:
    """Match ``target`` colour statistics to ``ref``. Both ``[B,H,W,C]`` 0..1."""
    if ref is None:
        return target
    if ref.dim() == 3:
        ref = ref.unsqueeze(0)
    method = (method or "reinhard").lower()
    if method == "reinhard":
        out = _reinhard(target, ref)
    elif method == "mkl":
        out = _mkl(target, ref)
    elif method in ("hm-mvgd", "hm_mvgd", "mvgd"):
        out = _hm_mvgd(target, ref)
    else:
        out = _reinhard(target, ref)
    return out.clamp(0.0, 1.0).to(target.dtype)


# --------------------------------------------------------------------------- #
# Wavelet subband locking (WF-VAE style)
# --------------------------------------------------------------------------- #
_SUBBANDS = {
    "LLL only": ("aa",),
    "LLL+LLH": ("aa", "ad"),
    "all-low": ("aa", "ad", "da"),
}


def _wavelet_lock_channel(t_ch: np.ndarray, r_ch: np.ndarray, levels: int, locked: tuple) -> np.ndarray:
    wt = pywt.wavedec2(t_ch, "haar", level=levels)
    wr = pywt.wavedec2(r_ch, "haar", level=levels)
    # approximation (LL) coefficients = low frequency → match to reference mean/std
    if "aa" in locked:
        a_t, a_r = wt[0], wr[0]
        std_t = a_t.std() + 1e-6
        wt[0] = (a_t - a_t.mean()) / std_t * (a_r.std() + 1e-6) + a_r.mean()
    # optionally also stabilise the coarsest detail bands
    for lvl_name, band_idx in (("ad", 0), ("da", 1)):
        if lvl_name in locked and len(wt) > 1:
            cH_t = list(wt[1])
            cH_r = list(wr[1])
            b_t = cH_t[band_idx]
            b_r = cH_r[band_idx]
            cH_t[band_idx] = (b_t - b_t.mean()) / (b_t.std() + 1e-6) * (b_r.std() + 1e-6) + b_r.mean()
            wt[1] = tuple(cH_t)
    rec = pywt.waverec2(wt, "haar")
    return rec[: t_ch.shape[0], : t_ch.shape[1]]


def wavelet_lock(images: torch.Tensor, reference: torch.Tensor, levels: int = 2,
                 lock_mode: str = "LLL only") -> torch.Tensor:
    """Lock low-freq wavelet subbands of every frame in ``images`` to ``reference``.

    ``images`` ``[B,H,W,C]``, ``reference`` ``[H,W,C]`` or ``[1,H,W,C]``.
    """
    if reference is None:
        return images
    if reference.dim() == 4:
        reference = reference[0]
    locked = _SUBBANDS.get(lock_mode, ("aa",))
    out = images.clone()
    ref_np = reference.detach().cpu().float().numpy()
    for b in range(images.shape[0]):
        frame = images[b].detach().cpu().float().numpy()
        for c in range(frame.shape[-1]):
            rec = _wavelet_lock_channel(frame[..., c], ref_np[..., c], levels, locked)
            out[b, ..., c] = torch.from_numpy(rec).to(out.device, out.dtype)
    return out.clamp(0.0, 1.0)
