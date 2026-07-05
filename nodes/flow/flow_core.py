"""
flow_core.py — optical-flow foundation for the Wan vid2vid consistency stack.

This is the shared, grounded core that the FlowVid / FRESCO / TokenFlow
adaptations all build on (all three reduce, on a Wan DiT, to flow-based
propagation of the SOURCE video's frame-to-frame correspondence):

  * RAFT optical flow                — torchvision `raft_large` / `raft_small`
                                       (Teed & Deng, ECCV 2020, official weights).
  * backward warp by a flow field    — `F.grid_sample` (standard).
  * forward-backward consistency     — the occlusion test used by FlowVid /
                                       FRESCO / SoftSplat: a pixel is reliable
                                       when  ||F_fwd(p) + F_bwd(p + F_fwd(p))|| is
                                       small relative to the flow magnitude.
  * flow → RGB visualization         — Middlebury colour wheel.

Nothing here is invented: RAFT is the official model, the warp is plain
grid_sample, and the FB-consistency occlusion check is the textbook formulation
(Sundaram et al. 2010; used verbatim by FlowVid & FRESCO).

Pure torch + torchvision. No GPU required to import; the RAFT weights download
lazily on first use and run on whatever device the frames live on.

Author: Code2Collapse. Apache-2.0.
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

# ── lazy RAFT model cache ────────────────────────────────────────────────────
_RAFT_CACHE: dict[str, torch.nn.Module] = {}

RaftSize = Literal["large", "small"]


def _round_to_multiple(x: int, m: int = 8) -> int:
    return max(m, int(round(x / m)) * m)


def load_raft(size: RaftSize = "large", device: torch.device | str = "cpu") -> torch.nn.Module:
    """Lazily build + cache a torchvision RAFT model on ``device`` (eval mode)."""
    key = f"{size}:{str(device)}"
    cached = _RAFT_CACHE.get(key)
    if cached is not None:
        return cached
    from torchvision.models.optical_flow import (
        raft_large, raft_small, Raft_Large_Weights, Raft_Small_Weights,
    )
    if size == "small":
        model = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False)
    else:
        model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False)
    model = model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    _RAFT_CACHE[key] = model
    return model


def _prep_for_raft(frames_bchw: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """frames [N,3,H,W] in 0..1 → RAFT input in -1..1, padded to /8. Returns (x, H, W)."""
    n, c, h, w = frames_bchw.shape
    th, tw = _round_to_multiple(h), _round_to_multiple(w)
    x = frames_bchw
    if (th, tw) != (h, w):
        x = F.interpolate(x, size=(th, tw), mode="bilinear", align_corners=False)
    x = (x * 2.0 - 1.0).clamp(-1.0, 1.0)        # RAFT expects [-1, 1]
    return x.contiguous(), h, w


@torch.no_grad()
def compute_flow(
    frames: torch.Tensor,
    *,
    size: RaftSize = "large",
    direction: Literal["forward", "backward"] = "forward",
    iters: int = 12,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Optical flow for an IMAGE batch.

    Args:
        frames: ``[T,H,W,3]`` (ComfyUI IMAGE), 0..1 float.
        direction: ``forward`` = flow t→t+1 ; ``backward`` = flow t+1→t.
        iters: RAFT refinement iterations.

    Returns:
        flow ``[T-1, 2, H, W]`` in PIXELS, at the input resolution (dx, dy).
        For ``T<=1`` returns an empty ``[0,2,H,W]`` tensor.
    """
    assert frames.ndim == 4 and frames.shape[-1] == 3, "frames must be [T,H,W,3]"
    t, h, w, _ = frames.shape
    dev = torch.device(device) if device is not None else frames.device
    if t <= 1:
        return frames.new_zeros((0, 2, h, w))

    fr = frames.permute(0, 3, 1, 2).to(dev)               # [T,3,H,W]
    if direction == "forward":
        img1, img2 = fr[:-1], fr[1:]
    else:
        img1, img2 = fr[1:], fr[:-1]

    x1, _, _ = _prep_for_raft(img1)
    x2, _, _ = _prep_for_raft(img2)
    model = load_raft(size, dev)
    flow_list = model(x1, x2, num_flow_updates=int(iters))
    flow = flow_list[-1]                                   # [T-1,2,th,tw] pixels @ raft res
    _, _, th, tw = flow.shape
    if (th, tw) != (h, w):
        flow = F.interpolate(flow, size=(h, w), mode="bilinear", align_corners=False)
        flow[:, 0] *= (w / tw)                             # rescale dx
        flow[:, 1] *= (h / th)                             # rescale dy
    return flow.contiguous()


def _base_grid(n: int, h: int, w: int, device, dtype) -> torch.Tensor:
    """Identity sampling grid in pixel coords, shape [N,H,W,2] (x,y)."""
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    grid = torch.stack((xs, ys), dim=-1)                  # [H,W,2]
    return grid.unsqueeze(0).expand(n, -1, -1, -1)


def flow_warp(
    x: torch.Tensor,
    flow: torch.Tensor,
    *,
    mode: str = "bilinear",
    padding_mode: str = "border",
) -> torch.Tensor:
    """Backward-warp ``x`` by ``flow`` (sample x at p+flow).

    With a forward flow F_{t→t+1}, ``flow_warp(x_{t+1}, F)`` ≈ x_t — i.e. it pulls
    the next frame back into the current frame's geometry. Works on any
    ``[N,C,H,W]`` tensor (images OR latents) as long as flow matches its H,W.

    Args:
        x:    ``[N,C,H,W]``.
        flow: ``[N,2,H,W]`` pixel displacements (dx, dy).
    """
    assert x.ndim == 4 and flow.ndim == 4, "x [N,C,H,W], flow [N,2,H,W]"
    n, c, h, w = x.shape
    if flow.shape[0] != n or flow.shape[-2:] != (h, w):
        flow = F.interpolate(flow, size=(h, w), mode="bilinear", align_corners=False)
        flow = flow.expand(n, -1, -1, -1) if flow.shape[0] == 1 else flow
    grid = _base_grid(n, h, w, x.device, x.dtype)         # [N,H,W,2] (x,y) pixels
    disp = flow.permute(0, 2, 3, 1).to(x.dtype)           # [N,H,W,2]
    sample = grid + disp
    # normalize to [-1,1] for grid_sample
    sx = 2.0 * sample[..., 0] / max(1, (w - 1)) - 1.0
    sy = 2.0 * sample[..., 1] / max(1, (h - 1)) - 1.0
    norm = torch.stack((sx, sy), dim=-1)
    return F.grid_sample(x, norm, mode=mode, padding_mode=padding_mode, align_corners=True)


def occlusion_mask(
    flow_fwd: torch.Tensor,
    flow_bwd: torch.Tensor,
    *,
    alpha1: float = 0.01,
    alpha2: float = 0.5,
) -> torch.Tensor:
    """Forward-backward consistency occlusion mask (Sundaram et al. 2010).

    A pixel p is *reliable* (mask=1) when the round trip is consistent:
        ||F_fwd(p) + F_bwd_warped(p)||^2  <  alpha1*(||F_fwd||^2+||F_bwd_w||^2) + alpha2
    where F_bwd_warped = backward-warp of F_bwd by F_fwd. Occluded/disoccluded
    pixels (mask→0) are exactly the ones FlowVid/FRESCO refuse to propagate.

    Args:
        flow_fwd: ``[N,2,H,W]`` flow t→t+1.
        flow_bwd: ``[N,2,H,W]`` flow t+1→t (same N).
    Returns:
        mask ``[N,1,H,W]`` in {0,1} (float).
    """
    fb = flow_warp(flow_bwd, flow_fwd, mode="bilinear", padding_mode="border")
    diff = flow_fwd + fb
    diff_sq = (diff ** 2).sum(dim=1, keepdim=True)        # [N,1,H,W]
    mag = (flow_fwd ** 2).sum(dim=1, keepdim=True) + (fb ** 2).sum(dim=1, keepdim=True)
    thresh = alpha1 * mag + alpha2
    return (diff_sq < thresh).to(flow_fwd.dtype)


def _flow_to_rgb(flow_2hw: torch.Tensor) -> torch.Tensor:
    """Single flow [2,H,W] → RGB [H,W,3] 0..1 via the Middlebury colour wheel."""
    u, v = flow_2hw[0], flow_2hw[1]
    rad = torch.sqrt(u * u + v * v)
    rad_max = torch.clamp(rad.max(), min=1e-5)
    ang = torch.atan2(-v, -u) / torch.pi                  # [-1,1]
    hue = (ang + 1.0) / 2.0                                # [0,1]
    sat = torch.clamp(rad / rad_max, 0, 1)
    val = torch.ones_like(hue)
    # HSV→RGB
    h6 = hue * 6.0
    i = torch.floor(h6).long() % 6
    f = h6 - torch.floor(h6)
    p = val * (1 - sat); q = val * (1 - sat * f); t = val * (1 - sat * (1 - f))
    r = torch.zeros_like(hue); g = torch.zeros_like(hue); b = torch.zeros_like(hue)
    for idx, (rr, gg, bb) in enumerate([(val, t, p), (q, val, p), (p, val, t),
                                        (p, q, val), (t, p, val), (val, p, q)]):
        m = i == idx
        r = torch.where(m, rr, r); g = torch.where(m, gg, g); b = torch.where(m, bb, b)
    return torch.stack((r, g, b), dim=-1).clamp(0, 1)


def flow_to_image(flow: torch.Tensor) -> torch.Tensor:
    """Flow ``[N,2,H,W]`` → IMAGE batch ``[N,H,W,3]`` 0..1 for previewing."""
    return torch.stack([_flow_to_rgb(flow[i]) for i in range(flow.shape[0])], dim=0)
