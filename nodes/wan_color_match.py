"""
wan_color_match.py — WanAnchorColorMatchC2C

Kills the color/contrast DRIFT that accumulates over Wan long-video / extension /
I2V continuation (see WAN_MODEL_RESEARCH.md §2.1): match every frame to a FIXED
anchor (the first frame, or a wired reference) instead of letting each chunk drift
relative to the previous one — so bias can't compound chunk-to-chunk.

Standalone post-process (does NOT touch the sampler), so it's safe to wire after
any Wan sampler/decoder. Pure torch + optional cv2 for histogram matching.

Author: Code2Collapse. Licensed under the Apache License, Version 2.0.
"""
from __future__ import annotations

import logging

import numpy as np
import torch

log = logging.getLogger(__name__)


def _mean_std_match(frame: torch.Tensor, ref_mean, ref_std, src_mean=None, src_std=None):
    """Reinhard per-channel mean/std transfer. frame [H,W,3] 0..1."""
    if src_mean is None:
        src_mean = frame.reshape(-1, 3).mean(0)
        src_std = frame.reshape(-1, 3).std(0)
    std_ratio = ref_std / src_std.clamp(min=1e-5)
    return (frame - src_mean) * std_ratio + ref_mean


class WanAnchorColorMatchC2C:
    """Anchor every frame's colour to a fixed reference to remove temporal drift."""

    CATEGORY = "WanNodeExperiments/Postprocess"
    FUNCTION = "match"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    DESCRIPTION = (
        "Remove Wan colour/contrast DRIFT in long / extended / looped video by "
        "matching every frame to a fixed anchor (first frame or a wired reference), "
        "not to the previous frame — so bias never accumulates. Wire after the VAE "
        "decode. mean_std = fast Reinhard transfer; histogram = stronger (needs cv2)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "anchor": (["first_frame", "last_frame", "reference"], {
                    "default": "first_frame",
                    "tooltip": "Colour target every frame is matched to. 'reference' uses the wired reference image.",
                }),
                "method": (["mean_std", "histogram"], {
                    "default": "mean_std",
                    "tooltip": "mean_std = fast per-channel Reinhard transfer; histogram = per-channel CDF match (stronger, needs OpenCV).",
                }),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "0 = no change, 1 = full match to the anchor."}),
            },
            "optional": {
                "reference": ("IMAGE", {"tooltip": "Anchor frame when anchor='reference' (uses its first frame)."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, images, anchor, method, strength, reference=None):
        import hashlib
        h = hashlib.md5()
        try:
            h.update(np.ascontiguousarray(images.cpu().numpy()).tobytes())
            if reference is not None:
                h.update(np.ascontiguousarray(reference.cpu().numpy()).tobytes())
        except Exception:  # noqa: BLE001
            pass
        return f"{anchor}-{method}-{strength}-{h.hexdigest()[:12]}"

    def match(self, images, anchor, method, strength, reference=None):
        if images is None or images.shape[0] == 0 or strength <= 0.0:
            return (images,)
        imgs = images.float().clamp(0, 1)
        F = imgs.shape[0]

        # pick the anchor frame
        if anchor == "reference" and reference is not None and reference.shape[0] > 0:
            anchor_frame = reference[0].float().clamp(0, 1)
        elif anchor == "last_frame":
            anchor_frame = imgs[-1]
        else:
            anchor_frame = imgs[0]

        if method == "histogram":
            try:
                import cv2
                out = self._histogram_match(imgs, anchor_frame, strength, cv2)
                return (out,)
            except Exception as exc:  # noqa: BLE001
                log.warning("[WanAnchorColorMatch] histogram unavailable (%s); falling back to mean_std.", exc)

        ref_mean = anchor_frame.reshape(-1, 3).mean(0)
        ref_std = anchor_frame.reshape(-1, 3).std(0)
        out = torch.empty_like(imgs)
        for i in range(F):
            matched = _mean_std_match(imgs[i], ref_mean, ref_std)
            out[i] = (imgs[i] * (1.0 - strength) + matched * strength).clamp(0, 1)
        return (out,)

    @staticmethod
    def _histogram_match(imgs, anchor_frame, strength, cv2):
        ref = (anchor_frame.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        # per-channel CDF lookup from the anchor
        luts = []
        for c in range(3):
            ref_hist = cv2.calcHist([ref], [c], None, [256], [0, 256]).flatten()
            ref_cdf = np.cumsum(ref_hist) / max(1.0, ref_hist.sum())
            luts.append(ref_cdf)
        out = torch.empty_like(imgs)
        for i in range(imgs.shape[0]):
            src = (imgs[i].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            res = np.empty_like(src)
            for c in range(3):
                src_hist = cv2.calcHist([src], [c], None, [256], [0, 256]).flatten()
                src_cdf = np.cumsum(src_hist) / max(1.0, src_hist.sum())
                mapping = np.interp(src_cdf, luts[c], np.arange(256)).astype(np.uint8)
                res[..., c] = mapping[src[..., c]]
            res_t = torch.from_numpy(res.astype(np.float32) / 255.0)
            out[i] = (imgs[i] * (1.0 - strength) + res_t * strength).clamp(0, 1)
        return out


NODE_CLASS_MAPPINGS = {"WanAnchorColorMatchC2C": WanAnchorColorMatchC2C}
NODE_DISPLAY_NAME_MAPPINGS = {"WanAnchorColorMatchC2C": "Wan Anchor Color-Match (anti-drift)"}
