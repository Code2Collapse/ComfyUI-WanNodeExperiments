"""
flow_nodes.py — WNE nodes exposing the optical-flow consistency foundation.

  WNE_RAFTOpticalFlow      IMAGE  -> WAN_FLOW (+ flow preview IMAGE)
  WNE_FlowTemporalConsistency  IMAGE (+ optional WAN_FLOW) -> IMAGE

The temporal-consistency pass is the grounded FlowVid / FRESCO mechanism in
pixel space: each frame is blended with the occlusion-masked, flow-aligned
previous *output* frame, so content that is genuinely tracked stays locked
frame-to-frame (camera stability / de-flicker) while occluded/disoccluded
regions fall back to the per-frame image (no smearing). This is the same
forward-backward-consistency propagation FlowVid and FRESCO use; here it runs
post-decode where it is model-agnostic and Wan-VAE-temporal-compression-safe.

A WAN_FLOW object is a dict::

    {"fwd": [T-1,2,H,W], "bwd": [T-1,2,H,W],
     "occ_fwd": [T-1,1,H,W], "occ_bwd": [T-1,1,H,W], "size": (H,W), "frames": T}

so the later FRESCO / TokenFlow stages reuse the exact same flow + occlusion.

Author: Code2Collapse. Apache-2.0.
"""
from __future__ import annotations

import torch

from .flow_core import compute_flow, flow_warp, occlusion_mask, flow_to_image, RaftSize


def _build_flow(frames: torch.Tensor, size: RaftSize, iters: int, device) -> dict:
    fwd = compute_flow(frames, size=size, direction="forward", iters=iters, device=device)
    bwd = compute_flow(frames, size=size, direction="backward", iters=iters, device=device)
    occ_fwd = occlusion_mask(fwd, bwd) if fwd.shape[0] else fwd.new_zeros((0, 1, *frames.shape[1:3]))
    occ_bwd = occlusion_mask(bwd, fwd) if bwd.shape[0] else bwd.new_zeros((0, 1, *frames.shape[1:3]))
    return {"fwd": fwd, "bwd": bwd, "occ_fwd": occ_fwd, "occ_bwd": occ_bwd,
            "size": tuple(frames.shape[1:3]), "frames": int(frames.shape[0])}


class WNE_RAFTOpticalFlow:
    """Compute RAFT optical flow + forward/backward occlusion for a video batch."""

    CATEGORY = "WanNodeExperiments/Flow"
    FUNCTION = "run"
    RETURN_TYPES = ("WAN_FLOW", "IMAGE")
    RETURN_NAMES = ("flow", "flow_preview")
    DESCRIPTION = ("RAFT optical flow (torchvision, official) for an IMAGE video batch — "
                   "forward+backward flow and forward-backward-consistency occlusion masks, "
                   "the shared foundation for the FlowVid/FRESCO/TokenFlow consistency stages.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "model_size": (["large", "small"], {"default": "large",
                    "tooltip": "RAFT backbone. large = most accurate; small = faster/less VRAM."}),
                "iterations": ("INT", {"default": 12, "min": 1, "max": 32,
                    "tooltip": "RAFT refinement iterations (more = sharper flow, slower)."}),
                "device": (["auto", "gpu", "cpu"], {"default": "auto"}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, images, model_size, iterations, device):
        import hashlib
        h = hashlib.md5()
        try:
            h.update(images[:1].cpu().numpy().tobytes())
            h.update(f"{images.shape}-{model_size}-{iterations}".encode())
        except Exception:  # noqa: BLE001
            pass
        return h.hexdigest()

    def _pick_device(self, images, device):
        if device == "cpu":
            return torch.device("cpu")
        if device == "gpu":
            try:
                import comfy.model_management as mm
                return mm.get_torch_device()
            except Exception:  # noqa: BLE001
                return images.device
        return images.device

    def run(self, images, model_size, iterations, device):
        if images is None or images.shape[0] < 2:
            empty = {"fwd": images.new_zeros((0, 2, *images.shape[1:3])) if images is not None else torch.zeros(0),
                     "bwd": None, "occ_fwd": None, "occ_bwd": None,
                     "size": tuple(images.shape[1:3]) if images is not None else (0, 0),
                     "frames": int(images.shape[0]) if images is not None else 0}
            prev = images if images is not None else torch.zeros(1, 8, 8, 3)
            return (empty, prev)
        dev = self._pick_device(images, device)
        flow = _build_flow(images.float().clamp(0, 1), model_size, int(iterations), dev)
        preview = flow_to_image(flow["fwd"]).to(images.device) if flow["fwd"].shape[0] else images
        return (flow, preview)


class WNE_FlowTemporalConsistency:
    """Occlusion-aware flow temporal-consistency pass (FlowVid/FRESCO, pixel space)."""

    CATEGORY = "WanNodeExperiments/Flow"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    DESCRIPTION = (
        "Lock frame-to-frame geometry / kill jitter: blends every frame with the "
        "occlusion-masked, flow-aligned PREVIOUS output frame (FlowVid/FRESCO "
        "forward-backward-consistency propagation). Tracked content stays stable; "
        "occluded/disoccluded regions fall back to the per-frame image (no smearing). "
        "Pass a precomputed WAN_FLOW, or let it run RAFT internally."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "strength": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "How strongly each frame snaps to the flow-aligned previous frame (1 = max stability, may ghost)."}),
                "occlusion_gate": ("BOOLEAN", {"default": True,
                    "tooltip": "Only propagate where the forward-backward flow check is reliable (recommended)."}),
                "model_size": (["large", "small"], {"default": "large"}),
                "iterations": ("INT", {"default": 12, "min": 1, "max": 32}),
            },
            "optional": {
                "flow": ("WAN_FLOW", {"tooltip": "Precomputed flow from WNE_RAFTOpticalFlow (skips recompute)."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, images, strength, occlusion_gate, model_size, iterations, flow=None):
        import hashlib
        h = hashlib.md5()
        try:
            h.update(images.cpu().numpy().tobytes())
            h.update(f"{strength}-{occlusion_gate}-{model_size}-{iterations}".encode())
        except Exception:  # noqa: BLE001
            pass
        return h.hexdigest()

    def run(self, images, strength, occlusion_gate, model_size, iterations, flow=None):
        if images is None or images.shape[0] < 2 or strength <= 0.0:
            return (images,)
        x = images.float().clamp(0, 1)
        T, H, W, _ = x.shape
        dev = x.device
        if flow is None or flow.get("bwd") is None or flow.get("bwd").shape[0] != T - 1:
            flow = _build_flow(x, model_size, int(iterations), dev)
        bwd = flow["bwd"].to(dev)                 # bwd[i] = F_{i+1 -> i}
        occ = flow["occ_bwd"].to(dev) if (occlusion_gate and flow.get("occ_bwd") is not None) else None

        xb = x.permute(0, 3, 1, 2)                # [T,3,H,W]
        out = [xb[0]]
        s = float(strength)
        for t in range(1, T):
            prev = out[t - 1].unsqueeze(0)        # [1,3,H,W] previous OUTPUT
            f = bwd[t - 1:t]                       # F_{t -> t-1}
            aligned = flow_warp(prev, f)          # previous content in frame-t geometry
            w = s
            if occ is not None:
                w = s * occ[t - 1:t]              # [1,1,H,W] reliability-weighted
            cur = xb[t:t + 1]
            blended = aligned * w + cur * (1.0 - w)
            out.append(blended.clamp(0, 1).squeeze(0))
        res = torch.stack(out, dim=0).permute(0, 2, 3, 1).contiguous()
        return (res.to(images.dtype),)


NODE_CLASS_MAPPINGS = {
    "WNE_RAFTOpticalFlow": WNE_RAFTOpticalFlow,
    "WNE_FlowTemporalConsistency": WNE_FlowTemporalConsistency,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "WNE_RAFTOpticalFlow": "RAFT Optical Flow (WNE)",
    "WNE_FlowTemporalConsistency": "Flow Temporal Consistency — FlowVid/FRESCO (WNE)",
}
