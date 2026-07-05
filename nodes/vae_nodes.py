"""
vae_nodes.py — decode-side colour / detail / temporal nodes.

Nodes: B1 RefDecoderVAEDecode, B2 WanLockedDecode, B3 WaveletColorLock,
       D1 TemporalCausalVAEDecode, D2 YUVColorLockDecode, D3 Latent3DAntiAlias.

Credits: Wan VAE tiling + context-window decode patterns studied from **Kijai**
(ComfyUI-WanVideoWrapper) and **wuwukaka** (ComfyUI-WanAnimatePlus). No code is
imported from them.

Algorithm sources:
  * WF-VAE wavelet continuity — Li et al. 2411.17459.
  * RefDecoder reference attention — Fan et al. 2605.15196.
"""

from __future__ import annotations

import torch

from .compat import log, humanise
from .color_utils import color_match, rgb_to_ycbcr, ycbcr_to_rgb, wavelet_lock
from .freq_utils import laplacian_highfreq_3d, edge_aware_blur_3d

CAT_VAE = "WanNodeExperiments/VAE"


# --------------------------------------------------------------------------- #
# Shared decode helper
# --------------------------------------------------------------------------- #
def _vae_decode(vae, samples, tile_size=0, overlap=64):
    """Decode a latent to IMAGE [N,H,W,3], with a tiled fallback on failure."""
    if tile_size and tile_size > 0 and hasattr(vae, "decode_tiled"):
        try:
            return vae.decode_tiled(samples, tile_x=tile_size, tile_y=tile_size, overlap=overlap)
        except Exception as exc:  # noqa: BLE001
            log.warning("[WanNodeExperiments] decode_tiled failed (%s); plain decode", exc)
    try:
        return vae.decode(samples)
    except Exception as exc:  # noqa: BLE001
        if hasattr(vae, "decode_tiled"):
            log.warning("[WanNodeExperiments] decode OOM/err (%s); falling back to tiled", exc)
            return vae.decode_tiled(samples, tile_x=256, tile_y=256, overlap=overlap)
        raise


def _as_nhwc(img):
    """Ensure decoded output is [N,H,W,3]."""
    if img.dim() == 4 and img.shape[-1] in (1, 3, 4):
        return img
    if img.dim() == 3:
        return img.unsqueeze(0)
    return img


# --------------------------------------------------------------------------- #
# B1 — RefDecoder-style reference-conditioned decode
# --------------------------------------------------------------------------- #
class RefDecoderVAEDecode:
    """B1. RefDecoder-style Reference-Conditioned Decode (Fan et al., 2605.15196).

    The paper injects reference-frame tokens at the decoder's up-sampling stages
    via reference attention to recover detail/colour the unconditional decoder
    loses. Without the trained decoder adapter we provide a faithful functional
    fallback: decode normally, then transfer the reference's high-frequency
    detail (3D Laplacian) and colour statistics onto every frame. ``dropout_rate``
    scales down the reference influence (the training-time regularizer).

    TODO: load + run the trained RefDecoder cross-attention adapter at each
    up-sampling stage for the full +PSNR gain.

    Credits: Wan 2.2 wrapper foundations by Kijai & wuwukaka.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "reference_image": ("IMAGE",),
                "dropout_rate": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "detail_weight": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 2.0, "step": 0.01}),
                "color_match_method": (["reinhard", "mkl", "hm-mvgd"], {"default": "reinhard"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "decode"
    CATEGORY = CAT_VAE
    DESCRIPTION = "Reference-conditioned decode: recover detail/colour from a reference frame."

    def decode(self, samples, vae, reference_image, dropout_rate, detail_weight=0.25,
               color_match_method="reinhard"):
        try:
            img = _as_nhwc(_vae_decode(vae, samples["samples"]))
            ref = reference_image
            if ref.dim() == 3:
                ref = ref.unsqueeze(0)
            ref0 = ref[0]
            influence = max(0.0, 1.0 - float(dropout_rate))

            # colour transfer toward reference
            matched = color_match(img, ref0, method=color_match_method)
            img = (1.0 - influence) * img + influence * matched

            # high-frequency detail transfer from reference (resized to frame size)
            import torch.nn.functional as F
            ref_chw = ref0.permute(2, 0, 1).unsqueeze(0)
            ref_rs = F.interpolate(ref_chw, size=img.shape[1:3], mode="bilinear", align_corners=False)
            ref_hf = laplacian_highfreq_3d(ref_rs, spatial_sigma=1.5)  # [1,3,H,W]
            ref_hf = ref_hf[0].permute(1, 2, 0).unsqueeze(0)  # [1,H,W,3]
            img = img + (detail_weight * influence) * ref_hf
            return (img.clamp(0.0, 1.0),)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(humanise(exc)) from exc

    @classmethod
    def IS_CHANGED(cls, samples, vae, reference_image, dropout_rate, **kw):
        import hashlib
        h = hashlib.md5(reference_image.detach().cpu().numpy().tobytes()).hexdigest()
        return f"refdec-{dropout_rate}-{h}"


# --------------------------------------------------------------------------- #
# B2 — Forced non-tiled decode + auto colour-lock
# --------------------------------------------------------------------------- #
class WanLockedDecode:
    """B2. Forced Non-Tiled Decode + Auto Colour-Lock.

    Decodes with the largest tile the user allows (``max_tile_size``; 0 = fully
    non-tiled) and colour-matches each context-window chunk back to a reference
    frame to stop the saturation drift that tiled decode introduces.

    Credits: Wan VAE tiling studied from Kijai & wuwukaka.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "reference_frame": ("IMAGE",),
                "max_tile_size": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 64,
                                          "tooltip": "0 = fully non-tiled (largest)."}),
                "color_match_method": (["reinhard", "mkl", "hm-mvgd"], {"default": "reinhard"}),
                "apply_every_window": ("BOOLEAN", {"default": True}),
            },
            "optional": {"window_size": ("INT", {"default": 16, "min": 1, "max": 1024})},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "decode"
    CATEGORY = CAT_VAE
    DESCRIPTION = "Largest-tile decode with per-window colour-lock to a reference."

    def decode(self, samples, vae, reference_frame, max_tile_size, color_match_method,
               apply_every_window, window_size=16):
        try:
            img = _as_nhwc(_vae_decode(vae, samples["samples"], tile_size=max_tile_size))
            ref = reference_frame[0] if reference_frame.dim() == 4 else reference_frame
            if not apply_every_window:
                out = color_match(img, ref, method=color_match_method)
                return (out,)
            out = img.clone()
            n = img.shape[0]
            for start in range(0, n, window_size):
                end = min(start + window_size, n)
                out[start:end] = color_match(img[start:end], ref, method=color_match_method)
            return (out,)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(humanise(exc)) from exc

    @classmethod
    def IS_CHANGED(cls, samples, vae, reference_frame, max_tile_size, color_match_method,
                   apply_every_window, window_size=16):
        import hashlib
        h = hashlib.md5(reference_frame.detach().cpu().numpy().tobytes()).hexdigest()
        return f"locked-{max_tile_size}-{color_match_method}-{apply_every_window}-{window_size}-{h}"


# --------------------------------------------------------------------------- #
# B3 — WF-VAE wavelet colour-lock
# --------------------------------------------------------------------------- #
class WaveletColorLock:
    """B3. WF-VAE Wavelet Colour-Lock (Li et al., arXiv:2411.17459).

    Keeps low-frequency wavelet subbands continuous with a reference frame to fix
    tiling/chunk colour seams at the mechanism level. Accepts already-decoded
    frames (``images``) or decodes ``samples`` with ``vae``.

    Credits: Wan VAE behaviour studied from Kijai & wuwukaka.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_frame": ("IMAGE",),
                "wavelet_levels": ("INT", {"default": 2, "min": 1, "max": 5}),
                "lock_subbands": (["LLL only", "LLL+LLH", "all-low"], {"default": "LLL only"}),
                "cache_mode": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "images": ("IMAGE",),
                "samples": ("LATENT",),
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "lock"
    CATEGORY = CAT_VAE
    DESCRIPTION = "Lock low-frequency wavelet subbands to a reference (kill tiling seams)."

    def lock(self, reference_frame, wavelet_levels, lock_subbands, cache_mode,
             images=None, samples=None, vae=None):
        try:
            if images is None:
                if samples is None or vae is None:
                    raise RuntimeError("Connect either 'images', or both 'samples' and 'vae'.")
                images = _as_nhwc(_vae_decode(vae, samples["samples"]))
            ref = reference_frame[0] if reference_frame.dim() == 4 else reference_frame
            out = wavelet_lock(images, ref, levels=wavelet_levels, lock_mode=lock_subbands)
            return (out,)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(humanise(exc)) from exc

    @classmethod
    def IS_CHANGED(cls, reference_frame, wavelet_levels, lock_subbands, cache_mode, **kw):
        import hashlib
        h = hashlib.md5(reference_frame.detach().cpu().numpy().tobytes()).hexdigest()
        return f"wavelet-{wavelet_levels}-{lock_subbands}-{cache_mode}-{h}"


# --------------------------------------------------------------------------- #
# D1 — Temporal causal VAE decode (overlap + cross-fade)
# --------------------------------------------------------------------------- #
class TemporalCausalVAEDecode:
    """D1. Temporal Causal VAE Tiler.

    Decodes the latent in overlapping temporal windows and cross-fades the
    overlapping pixel frames, eliminating chunk-boundary strobing in long video.
    Spatial tiling (``spatial_tile_size``/``spatial_overlap``) is forwarded to the
    VAE's tiled decode.

    Credits: Wan VAE temporal tiling studied from Kijai & wuwukaka.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "temporal_overlap": ("INT", {"default": 9, "min": 0, "max": 64,
                                             "tooltip": "Overlapping pixel frames cross-faded between windows."}),
                "spatial_tile_size": ("INT", {"default": 512, "min": 0, "max": 4096, "step": 64}),
                "spatial_overlap": ("INT", {"default": 64, "min": 0, "max": 512, "step": 16}),
            },
            "optional": {
                "latent_window": ("INT", {"default": 16, "min": 2, "max": 256,
                                          "tooltip": "Latent frames decoded per window."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "decode"
    CATEGORY = CAT_VAE
    DESCRIPTION = "Temporal-overlap decode with pixel-space cross-fade (no strobing)."

    def decode(self, samples, vae, temporal_overlap, spatial_tile_size, spatial_overlap, latent_window=16):
        try:
            s = samples["samples"]
            # Non-temporal latent → straight (optionally tiled) decode.
            if s.dim() != 5 or s.shape[2] <= latent_window:
                return (_as_nhwc(_vae_decode(vae, s, tile_size=spatial_tile_size, overlap=spatial_overlap)),)

            T = s.shape[2]
            lat_overlap = max(0, temporal_overlap // 4)  # ~4x temporal compression
            step = max(1, latent_window - lat_overlap)
            pieces = []
            starts = list(range(0, T, step))
            for wi, start in enumerate(starts):
                end = min(start + latent_window, T)
                chunk = s[:, :, start:end]
                dec = _as_nhwc(_vae_decode(vae, chunk, tile_size=spatial_tile_size, overlap=spatial_overlap))
                pieces.append(dec)
                if end >= T:
                    break

            if len(pieces) == 1:
                return (pieces[0],)

            px_overlap = temporal_overlap
            result = pieces[0]
            for nxt in pieces[1:]:
                ov = min(px_overlap, result.shape[0], nxt.shape[0])
                if ov <= 0:
                    result = torch.cat([result, nxt], dim=0)
                    continue
                w = torch.linspace(0, 1, ov, device=result.device).view(ov, 1, 1, 1).to(result.dtype)
                tail = result[-ov:]
                head = nxt[:ov]
                blended = (1.0 - w) * tail + w * head
                result = torch.cat([result[:-ov], blended, nxt[ov:]], dim=0)
            return (result.clamp(0.0, 1.0),)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(humanise(exc)) from exc

    @classmethod
    def IS_CHANGED(cls, samples, vae, temporal_overlap, spatial_tile_size, spatial_overlap, latent_window=16):
        return f"tcvae-{temporal_overlap}-{spatial_tile_size}-{spatial_overlap}-{latent_window}"


# --------------------------------------------------------------------------- #
# D2 — YUV colour-lock decode
# --------------------------------------------------------------------------- #
class YUVColorLockDecode:
    """D2. YUV Colour-Lock Decode.

    Converts decoded frames + reference to YCbCr and selectively matches channels:
    ``luma_only`` keeps generated Y and matches chroma to the reference (fixes
    washed-out/neon colour without touching generated lighting), ``chroma_only``
    matches luma, ``both`` matches all. ``chroma_blend`` controls strength.

    Credits: Wan VAE behaviour studied from Kijai & wuwukaka.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "reference_frame": ("IMAGE",),
                "yuv_lock_mode": (["luma_only", "chroma_only", "both"], {"default": "luma_only"}),
                "chroma_blend": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "decode"
    CATEGORY = CAT_VAE
    DESCRIPTION = "Lock luma, colour-match chroma (or vice-versa) in YCbCr space."

    def decode(self, samples, vae, reference_frame, yuv_lock_mode, chroma_blend):
        try:
            img = _as_nhwc(_vae_decode(vae, samples["samples"]))
            ref = reference_frame[0] if reference_frame.dim() == 4 else reference_frame
            img_ycc = rgb_to_ycbcr(img)
            ref_ycc = rgb_to_ycbcr(ref.unsqueeze(0))

            out = img_ycc.clone()
            blend = float(chroma_blend)
            if yuv_lock_mode in ("chroma_only", "both"):
                # match luma (Y)
                matched_y = color_match(img_ycc[..., 0:1], ref_ycc[..., 0:1], method="reinhard")
                out[..., 0:1] = (1 - blend) * img_ycc[..., 0:1] + blend * matched_y
            if yuv_lock_mode in ("luma_only", "both"):
                # match chroma (Cb, Cr)
                matched_c = color_match(img_ycc[..., 1:3], ref_ycc[..., 1:3], method="reinhard")
                out[..., 1:3] = (1 - blend) * img_ycc[..., 1:3] + blend * matched_c

            return (ycbcr_to_rgb(out),)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(humanise(exc)) from exc

    @classmethod
    def IS_CHANGED(cls, samples, vae, reference_frame, yuv_lock_mode, chroma_blend):
        import hashlib
        h = hashlib.md5(reference_frame.detach().cpu().numpy().tobytes()).hexdigest()
        return f"yuv-{yuv_lock_mode}-{chroma_blend}-{h}"


# --------------------------------------------------------------------------- #
# D3 — 3D anti-alias latent upscaler
# --------------------------------------------------------------------------- #
class Latent3DAntiAlias:
    """D3. 3D Anti-Alias Latent Upscaler.

    Applies an edge-aware 3D Gaussian filter (spatial + temporal) in latent space
    before decode, removing temporal jitter/shimmer on upscaled edges while
    preserving real edges (``threshold``).

    Credits: Wan latent behaviour studied from Kijai & wuwukaka.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "spatial_sigma": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 5.0, "step": 0.05}),
                "temporal_sigma": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 5.0, "step": 0.05}),
                "threshold": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01,
                                        "tooltip": "Edge preservation: higher keeps more edges sharp."}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "filter"
    CATEGORY = CAT_VAE
    DESCRIPTION = "Edge-aware 3D anti-alias filter in latent space."

    def filter(self, samples, spatial_sigma, temporal_sigma, threshold):
        try:
            s = samples["samples"]
            if spatial_sigma <= 0 and temporal_sigma <= 0:
                return (samples,)
            if s.dim() == 5:
                B, C, T, H, W = s.shape
                x = s.reshape(B, C, T, H, W)
            elif s.dim() == 4:
                x = s  # [B,C,H,W] treated as single frame
            else:
                return (samples,)
            filtered = edge_aware_blur_3d(x, spatial_sigma, temporal_sigma, threshold)
            out = dict(samples)
            out["samples"] = filtered.to(s.dtype)
            return (out,)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(humanise(exc)) from exc

    @classmethod
    def IS_CHANGED(cls, samples, spatial_sigma, temporal_sigma, threshold):
        return f"aa3d-{spatial_sigma}-{temporal_sigma}-{threshold}"


# --------------------------------------------------------------------------- #
# Phase 4 — refined tiling/context decode + encode (forked from Kijai's Wan VAE)
# Wan VAE is a causal 3D VAE (8x spatial / 4x temporal, z_dim=16) processed in
# temporal chunks with a feature cache. These nodes drive the native ComfyUI VAE
# with explicit spatial+temporal tiling and graceful fallbacks.
# --------------------------------------------------------------------------- #
def _vae_encode(vae, pixels, tile_size=0, overlap=64):
    """Encode IMAGE [N,H,W,3] -> latent samples, tiled fallback on failure."""
    import torch as _t

    with _t.inference_mode():
        if tile_size and tile_size > 0 and hasattr(vae, "encode_tiled"):
            try:
                return vae.encode_tiled(pixels, tile_x=tile_size, tile_y=tile_size, overlap=overlap)
            except Exception as exc:  # noqa: BLE001
                log.warning("[WanNodeExperiments] encode_tiled failed (%s); plain encode", exc)
        try:
            return vae.encode(pixels)
        except Exception as exc:  # noqa: BLE001
            if hasattr(vae, "encode_tiled"):
                log.warning("[WanNodeExperiments] encode err (%s); tiled fallback", exc)
                return vae.encode_tiled(pixels, tile_x=256, tile_y=256, overlap=overlap)
            raise


def _clean_vram():
    try:
        import comfy.model_management as mm

        mm.soft_empty_cache()
    except Exception:  # noqa: BLE001
        pass


class WanVAEDecodeTiled:
    """Refined spatial+temporal tiled VAE decode (native VAE).

    Exposes spatial tile size/overlap and an optional temporal tile (Wan VAE is
    causal-3D); forwards temporal params to the VAE's tiled decode when supported,
    else falls back to spatial tiling, else plain decode. Empties VRAM after.

    Credits: Wan VAE tiling/context studied from Kijai (wan_video_vae.py).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "tile_size": ("INT", {"default": 512, "min": 0, "max": 4096, "step": 64,
                                      "tooltip": "0 = no spatial tiling."}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 512, "step": 16}),
            },
            "optional": {
                "temporal_tile": ("INT", {"default": 0, "min": 0, "max": 256,
                                          "tooltip": "Latent frames per temporal tile (0 = off)."}),
                "temporal_overlap": ("INT", {"default": 8, "min": 0, "max": 64}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "decode"
    CATEGORY = CAT_VAE
    DESCRIPTION = "Spatial+temporal tiled Wan VAE decode with graceful fallback."

    def decode(self, samples, vae, tile_size, overlap, temporal_tile=0, temporal_overlap=8):
        try:
            s = samples["samples"]
            if temporal_tile and temporal_tile > 0 and hasattr(vae, "decode_tiled"):
                try:
                    img = vae.decode_tiled(
                        s, tile_x=tile_size or 512, tile_y=tile_size or 512,
                        overlap=overlap, tile_t=temporal_tile, overlap_t=temporal_overlap)
                    _clean_vram()
                    return (_as_nhwc(img),)
                except TypeError:
                    log.info("[WanNodeExperiments] VAE has no temporal-tile params; spatial only")
                except Exception as exc:  # noqa: BLE001
                    log.warning("[WanNodeExperiments] temporal tiled decode failed (%s)", exc)
            img = _as_nhwc(_vae_decode(vae, s, tile_size=tile_size, overlap=overlap))
            _clean_vram()
            return (img,)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(humanise(exc)) from exc

    @classmethod
    def IS_CHANGED(cls, samples, vae, tile_size, overlap, temporal_tile=0, temporal_overlap=8):
        return f"wvaedec-{tile_size}-{overlap}-{temporal_tile}-{temporal_overlap}"


class WanVAEEncode:
    """Tiled Wan VAE encode: IMAGE (video frames) -> LATENT.

    The encode-side counterpart for V2V / I2V pipelines (Wan VAE handles the
    temporal dimension internally). Tiled with graceful fallback; empties VRAM.

    Credits: Wan VAE behaviour studied from Kijai (wan_video_vae.py).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pixels": ("IMAGE",),
                "vae": ("VAE",),
                "tile_size": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 64,
                                      "tooltip": "0 = no spatial tiling."}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 512, "step": 16}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "encode"
    CATEGORY = CAT_VAE
    DESCRIPTION = "Encode video frames to a Wan latent (tiled, V2V/I2V)."

    def encode(self, pixels, vae, tile_size, overlap):
        try:
            latent = _vae_encode(vae, pixels, tile_size=tile_size, overlap=overlap)
            _clean_vram()
            return ({"samples": latent},)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(humanise(exc)) from exc

    @classmethod
    def IS_CHANGED(cls, pixels, vae, tile_size, overlap):
        import hashlib
        h = hashlib.md5(pixels.detach().cpu().numpy().tobytes()).hexdigest()
        return f"wvaeenc-{tile_size}-{overlap}-{h}"


NODE_CLASS_MAPPINGS = {
    "WNE_RefDecoderVAEDecode": RefDecoderVAEDecode,
    "WNE_WanLockedDecode": WanLockedDecode,
    "WNE_WaveletColorLock": WaveletColorLock,
    "WNE_TemporalCausalVAEDecode": TemporalCausalVAEDecode,
    "WNE_YUVColorLockDecode": YUVColorLockDecode,
    "WNE_Latent3DAntiAlias": Latent3DAntiAlias,
    "WNE_WanVAEDecodeTiled": WanVAEDecodeTiled,
    "WNE_WanVAEEncode": WanVAEEncode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WNE_RefDecoderVAEDecode": "RefDecoder VAE Decode (WNE)",
    "WNE_WanLockedDecode": "Wan Locked Decode + Colour-Lock (WNE)",
    "WNE_WaveletColorLock": "Wavelet Colour-Lock · WF-VAE (WNE)",
    "WNE_TemporalCausalVAEDecode": "Temporal Causal VAE Decode (WNE)",
    "WNE_YUVColorLockDecode": "YUV Colour-Lock Decode (WNE)",
    "WNE_Latent3DAntiAlias": "3D Anti-Alias Latent (WNE)",
    "WNE_WanVAEDecodeTiled": "Wan VAE Decode · tiled (WNE)",
    "WNE_WanVAEEncode": "Wan VAE Encode · tiled (WNE)",
}
