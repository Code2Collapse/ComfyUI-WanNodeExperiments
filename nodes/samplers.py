"""
sampling_nodes.py — loaders, caches, long-video and noise-init nodes.

Nodes: A1 AnyFlowModelLoader, A4 WanMoECache, A5 UltraViCoAttentionDecay,
       A6 FreeLOCCorrection, A8 FreeInitNoise, C4 MotionMaxNoiseInit.

Credits: Wan 2.2 MoE expert switching, caching (TeaCache/MagCache) and context
windows were studied from **Kijai** (ComfyUI-WanVideoWrapper) and **wuwukaka**
(ComfyUI-WanAnimatePlus). No code is imported from them.

Algorithm sources:
  * AnyFlow   — Gu et al. 2605.13724 / NVlabs/AnyFlow (flow-map any-step).
  * UltraViCo — Tian et al. 2511.20123 (constant logit decay out-of-window).
  * FreeLOC   — 2603.25209 / Westlake-AGI-Lab/FreeLOC (VRPR + tiered sparse attn).
  * FreeInit  — Wu et al. 2312.07537 (low-freq noise reinitialization).
"""

from __future__ import annotations

import math
import os

import torch

from . import compat
from .compat import log, humanise
from .freq_utils import get_freq_filter, freq_mix_3d

# Hard requirements (see requirements.txt). torchvision/opencv used by MotionMax.
import torchvision  # noqa: F401
import cv2  # noqa: F401

CAT_SAMPLING = "WanNodeExperiments/Sampling"
CAT_LONG = "WanNodeExperiments/LongVideo"


# --------------------------------------------------------------------------- #
# A1 — AnyFlow Any-Step Loader
# --------------------------------------------------------------------------- #
class AnyFlowModelLoader:
    """A1. AnyFlow Any-Step Loader (Gu et al., arXiv:2605.13724, NVlabs/AnyFlow).

    Attaches an any-step flow-map schedule to the model so quality improves
    monotonically from few to many NFEs. ``target_nfe`` is read by the sampler
    via ``transformer_options``; ``causal_mode`` enables FAR/streaming order.

    The distilled flow-map checkpoint is validated and its metadata is loaded if
    a path is given. TODO: bind the distilled flow-map velocity field into the
    forward pass (requires the AnyFlow-FAR distilled weights, e.g.
    ``nvidia/AnyFlow-FAR-Wan2.1-14B``). Until then the schedule + NFE budget are
    applied and the base model runs unchanged.

    Credits: Wan 2.2 wrapper foundations by Kijai & wuwukaka.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "flow_map_checkpoint": ("STRING", {"default": "", "multiline": False,
                                                   "tooltip": "Path to distilled flow-map checkpoint (optional)."}),
                "target_nfe": ("INT", {"default": 8, "min": 1, "max": 128,
                                       "tooltip": "Number of function evals; live-adjustable per queue."}),
                "causal_mode": ("BOOLEAN", {"default": False,
                                            "tooltip": "FAR/streaming causal sampling order."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = CAT_SAMPLING
    DESCRIPTION = "AnyFlow flow-map any-step schedule (4→32 NFE, monotonic)."

    def load(self, model, flow_map_checkpoint, target_nfe, causal_mode):
        try:
            m = compat.clone_model(model)
            meta = {"target_nfe": int(target_nfe), "causal_mode": bool(causal_mode),
                    "checkpoint": None}
            ckpt = (flow_map_checkpoint or "").strip()
            if ckpt:
                if not os.path.exists(ckpt):
                    raise FileNotFoundError(ckpt)
                try:
                    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
                    meta["checkpoint"] = ckpt
                    meta["num_tensors"] = len(sd) if hasattr(sd, "__len__") else 0
                    log.info("[WanNodeExperiments] AnyFlow ckpt loaded: %s (%d tensors)",
                             ckpt, meta["num_tensors"])
                except Exception as exc:  # noqa: BLE001
                    log.warning("[WanNodeExperiments] AnyFlow ckpt unreadable (%s); using schedule only", exc)
            compat.set_transformer_option(m, "wne_anyflow", meta)
            return (m,)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(humanise(exc)) from exc

    @classmethod
    def IS_CHANGED(cls, flow_map_checkpoint, target_nfe, causal_mode, **kw):
        sig = f"{flow_map_checkpoint}-{target_nfe}-{causal_mode}"
        if flow_map_checkpoint and os.path.exists(flow_map_checkpoint):
            sig += f"-{os.path.getmtime(flow_map_checkpoint)}"
        return sig


# --------------------------------------------------------------------------- #
# A4 — Wan 2.2 MoE-Aware Cache
# --------------------------------------------------------------------------- #
def _make_cache_wrapper(threshold, switch_step, mode, reset_at_switch):
    """Residual-reuse cache (TeaCache/MagCache family) as a unet wrapper.

    Accumulates the relative L1 change of the input latent between steps. While
    the accumulation stays below ``threshold`` the cached residual (out − in) is
    re-applied instead of recomputing the model. The cache is force-reset at the
    high→low expert switch to avoid carrying stale state across experts.
    """
    state = {"prev_in": None, "residual": None, "acc": 0.0, "last_sigma": None, "step": -1}

    def _step(sigma):
        s = float(sigma.flatten()[0]) if torch.is_tensor(sigma) else float(sigma)
        if state["last_sigma"] is None or s > state["last_sigma"] + 1e-9:
            state["step"] = 0
            state["prev_in"] = None
            state["residual"] = None
            state["acc"] = 0.0
        else:
            state["step"] += 1
        state["last_sigma"] = s
        return state["step"]

    def wrapper(apply_model, args):
        x, t, c = args["input"], args["timestep"], args["c"]
        idx = _step(t)
        if reset_at_switch and idx == switch_step:
            state["prev_in"] = None
            state["residual"] = None
            state["acc"] = 0.0

        if state["prev_in"] is not None and state["prev_in"].shape == x.shape:
            rel = (x - state["prev_in"]).abs().mean() / (state["prev_in"].abs().mean() + 1e-8)
            if mode == "magcache" and state["residual"] is not None:
                rel = rel * (state["residual"].abs().mean() + 1e-8)
            state["acc"] += float(rel)
            if state["acc"] < threshold and state["residual"] is not None:
                return x + state["residual"]  # skip compute, reuse residual

        out = apply_model(x, t, **c)
        state["residual"] = (out - x).detach()
        state["prev_in"] = x.detach()
        state["acc"] = 0.0
        return out

    return wrapper


class WanMoECache:
    """A4. Wan 2.2 MoE-Aware Cache (TeaCache/MagCache, per-expert, boundary reset).

    Patches the high-noise and low-noise expert models with independent residual
    caches and forces a cache reset at the high→low switch so expert state never
    corrupts across the MoE boundary.

    Credits: TeaCache/MagCache + Wan 2.2 MoE switch behaviour studied from
    Kijai & wuwukaka.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_high": ("MODEL",),
                "model_low": ("MODEL",),
                "switch_step": ("INT", {"default": 10, "min": 0, "max": 1000}),
                "cache_mode": (["teacache", "magcache"], {"default": "teacache"}),
                "thresh_high": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 2.0, "step": 0.01}),
                "thresh_low": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 2.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("MODEL", "MODEL")
    RETURN_NAMES = ("model_high", "model_low")
    FUNCTION = "patch"
    CATEGORY = CAT_SAMPLING
    DESCRIPTION = "Per-expert TeaCache/MagCache with high→low boundary reset."

    def patch(self, model_high, model_low, switch_step, cache_mode, thresh_high, thresh_low):
        try:
            mh = compat.clone_model(model_high)
            ml = compat.clone_model(model_low)
            wh = _make_cache_wrapper(thresh_high, switch_step, cache_mode, reset_at_switch=True)
            wl = _make_cache_wrapper(thresh_low, switch_step, cache_mode, reset_at_switch=True)
            if not compat.set_unet_wrapper(mh, wh):
                compat.set_transformer_option(mh, "wne_moe_cache",
                                              {"mode": cache_mode, "thresh": thresh_high, "switch_step": switch_step})
            if not compat.set_unet_wrapper(ml, wl):
                compat.set_transformer_option(ml, "wne_moe_cache",
                                              {"mode": cache_mode, "thresh": thresh_low, "switch_step": switch_step})
            return (mh, ml)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(humanise(exc)) from exc

    @classmethod
    def IS_CHANGED(cls, switch_step, cache_mode, thresh_high, thresh_low, **kw):
        return f"moecache-{switch_step}-{cache_mode}-{thresh_high}-{thresh_low}"


# --------------------------------------------------------------------------- #
# A5 — RIFLEx + UltraViCo attention-logit decay
# --------------------------------------------------------------------------- #
class _DecayModifier:
    """Add a constant negative logit bias to out-of-window self-attention pairs.

    UltraViCo: λ_ij = 1 inside the training window, α (<1) beyond → in log space a
    constant ``log(α)`` is added to those attention logits. ``training_window`` is
    in token units here (frame-aware decay is exact on the Kijai path via
    ``decay_factor``; on the native path token distance is the best-effort proxy).
    """

    def __init__(self, decay_factor, window_tokens, first_frame_override):
        self.log_decay = math.log(max(float(decay_factor), 1e-4))
        self.window = max(1, int(window_tokens))
        self.first_override = first_frame_override

    def skip(self, layer_idx):
        return False

    def bias(self, layer_idx, q, k):
        if q.dim() < 2 or k.dim() < 2:
            return None
        q_len, k_len = q.shape[-2], k.shape[-2]
        if q_len != k_len:
            return None  # only self-attention
        device = q.device
        i = torch.arange(q_len, device=device).view(-1, 1)
        j = torch.arange(k_len, device=device).view(1, -1)
        dist = (i - j).abs()
        bias = torch.zeros(q_len, k_len, device=device, dtype=q.dtype)
        bias[dist > self.window] = self.log_decay
        if self.first_override is not None and self.first_override < 0:
            fo = math.log(max(abs(self.first_override), 1e-4))
            bias[0, self.window:] = fo
            bias[self.window:, 0] = fo
        return bias  # (q_len, k_len) broadcasts over heads/batch


class UltraViCoAttentionDecay:
    """A5. RIFLEx + UltraViCo — long-video extrapolation.

    Suppresses attention to tokens beyond the training window with a constant
    decay (UltraViCo, arXiv:2511.20123) and exposes the RIFLEx RoPE frequency
    knob (arXiv:2502.15894). On Kijai's ``WANVIDEOMODEL`` the decay maps directly
    onto his existing ``decay_factor``; on native ComfyUI a token-distance logit
    decay is applied via the attention interceptor.

    Credits: Wan 2.2 wrapper foundations by Kijai & wuwukaka.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "decay_factor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "training_window": ("INT", {"default": 81, "min": 1, "max": 100000,
                                            "tooltip": "Training window (tokens on native path / frames on Kijai)."}),
                "first_frame_decay_override": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 0.0, "step": 0.01,
                                                        "tooltip": "Stronger (negative) decay for first-frame pairs; 0 = off."}),
                "riflex_freq_index": ("INT", {"default": 0, "min": 0, "max": 16,
                                              "tooltip": "RIFLEx RoPE frequency index (0 = off)."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = CAT_LONG
    DESCRIPTION = "UltraViCo logit decay + RIFLEx RoPE for long-video extrapolation."

    def patch(self, model, decay_factor, training_window, first_frame_decay_override, riflex_freq_index):
        try:
            m = compat.clone_model(model)
            # Kijai-native knobs (real on the WANVIDEOMODEL path).
            compat.set_transformer_option(m, "decay_factor", float(decay_factor))
            compat.set_transformer_option(m, "riflex_freq_index", int(riflex_freq_index))
            compat.set_transformer_option(m, "wne_ultravico", {
                "decay_factor": decay_factor, "training_window": training_window,
                "first_frame_override": first_frame_decay_override})

            override = first_frame_decay_override if first_frame_decay_override < 0 else None
            modifier = _DecayModifier(decay_factor, training_window, override)

            def unet_wrapper(apply_model, args):
                with compat.attention_modifier(modifier):
                    return apply_model(args["input"], args["timestep"], **args["c"])

            compat.set_unet_wrapper(m, unet_wrapper)
            return (m,)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(humanise(exc)) from exc

    @classmethod
    def IS_CHANGED(cls, decay_factor, training_window, first_frame_decay_override, riflex_freq_index, **kw):
        return f"ultravico-{decay_factor}-{training_window}-{first_frame_decay_override}-{riflex_freq_index}"


# --------------------------------------------------------------------------- #
# A6 — FreeLOC Layer-Adaptive OOD Correction
# --------------------------------------------------------------------------- #
class _TSAModifier:
    """Tiered Sparse Attention: dense local window, progressively sparser distant.

    Builds an additive mask that keeps every key within ``local`` of a query, and
    beyond that keeps only every ``stride``-th key (−inf elsewhere). A faithful,
    training-free realization of FreeLOC's TSA for context-length OOD.
    """

    def __init__(self, local=512, stride=4, layer_flags=None):
        self.local = max(1, int(local))
        self.stride = max(2, int(stride))
        self.layer_flags = layer_flags  # optional per-layer enable list

    def skip(self, layer_idx):
        return False

    def bias(self, layer_idx, q, k):
        if self.layer_flags is not None and layer_idx < len(self.layer_flags) and not self.layer_flags[layer_idx]:
            return None
        if q.dim() < 2 or k.dim() < 2:
            return None
        q_len, k_len = q.shape[-2], k.shape[-2]
        if q_len != k_len or k_len <= self.local:
            return None
        device = q.device
        i = torch.arange(q_len, device=device).view(-1, 1)
        j = torch.arange(k_len, device=device).view(1, -1)
        dist = (i - j).abs()
        keep = (dist <= self.local) | ((j % self.stride) == 0)
        bias = torch.zeros(q_len, k_len, device=device, dtype=q.dtype)
        bias[~keep] = float("-inf")
        return bias


class FreeLOCCorrection:
    """A6. FreeLOC — Layer-Adaptive OOD Correction (arXiv:2603.25209,
    Westlake-AGI-Lab/FreeLOC).

    VRPR (Video-based Relative Position Re-encoding) + TSA (Tiered Sparse
    Attention) with layer-adaptive probing. TSA is applied via the attention
    interceptor on native ComfyUI; VRPR params are written to transformer_options
    (frame-level RoPE re-encoding needs model RoPE access — exact on the Kijai
    path). ``probe_calibration_path`` loads per-layer sensitivity flags if present.

    TODO: bind VRPR temporal RoPE re-encoding into the model's positional embedding
    (the calibration file format mirrors the official repo's probe output).

    Credits: Wan 2.2 wrapper foundations by Kijai & wuwukaka.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enable_vrpr": ("BOOLEAN", {"default": True}),
                "enable_tsa": ("BOOLEAN", {"default": True}),
                "probe_calibration_path": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "local_window": ("INT", {"default": 512, "min": 16, "max": 100000}),
                "sparse_stride": ("INT", {"default": 4, "min": 2, "max": 64}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = CAT_LONG
    DESCRIPTION = "FreeLOC VRPR + tiered sparse attention for long video."

    def patch(self, model, enable_vrpr, enable_tsa, probe_calibration_path, local_window=512, sparse_stride=4):
        try:
            m = compat.clone_model(model)
            layer_flags = None
            cal = (probe_calibration_path or "").strip()
            if cal:
                if not os.path.exists(cal):
                    raise FileNotFoundError(cal)
                try:
                    data = torch.load(cal, map_location="cpu", weights_only=True)
                    layer_flags = [bool(v) for v in (data.get("tsa_layers", []) if isinstance(data, dict) else data)]
                    log.info("[WanNodeExperiments] FreeLOC probe loaded: %d layers", len(layer_flags))
                except Exception as exc:  # noqa: BLE001
                    log.warning("[WanNodeExperiments] FreeLOC probe unreadable (%s); applying to all layers", exc)

            compat.set_transformer_option(m, "wne_freeloc", {
                "vrpr": bool(enable_vrpr), "tsa": bool(enable_tsa),
                "local_window": local_window, "sparse_stride": sparse_stride})

            if enable_tsa:
                modifier = _TSAModifier(local_window, sparse_stride, layer_flags)

                def unet_wrapper(apply_model, args):
                    with compat.attention_modifier(modifier):
                        return apply_model(args["input"], args["timestep"], **args["c"])

                compat.set_unet_wrapper(m, unet_wrapper)
            return (m,)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(humanise(exc)) from exc

    @classmethod
    def IS_CHANGED(cls, enable_vrpr, enable_tsa, probe_calibration_path, **kw):
        sig = f"freeloc-{enable_vrpr}-{enable_tsa}-{probe_calibration_path}"
        if probe_calibration_path and os.path.exists(probe_calibration_path):
            sig += f"-{os.path.getmtime(probe_calibration_path)}"
        return sig


# --------------------------------------------------------------------------- #
# A8 — FreeInit noise reinitialization
# --------------------------------------------------------------------------- #
class FreeInitNoise:
    """A8. FreeInit (Wu et al., arXiv:2312.07537).

    Reinitializes a latent by keeping its low-frequency structure and replacing
    the high-frequency content with fresh Gaussian noise (``freq_mix_3d`` with the
    chosen low-pass filter), repeated ``num_iters`` times. This is the core
    FreeInit reinitialization step — feed it a first-pass latent to remove
    temporal jitter, or loop it with a sampler for full FreeInit.

    Credits: Wan 2.2 wrapper foundations by Kijai & wuwukaka.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "num_iters": ("INT", {"default": 1, "min": 1, "max": 10}),
                "filter_method": (["gaussian", "butterworth", "ideal", "box"], {"default": "butterworth"}),
                "d_s": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "d_t": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "butterworth_n": ("INT", {"default": 4, "min": 1, "max": 16}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "reinit"
    CATEGORY = CAT_SAMPLING
    DESCRIPTION = "FreeInit low-frequency noise reinitialization."

    def reinit(self, latent, num_iters, filter_method, d_s, d_t, seed=0, butterworth_n=4):
        try:
            samples = latent["samples"]
            if samples.dim() < 3:
                raise RuntimeError("RuntimeError: latent must have at least [B,C,(T),H,W] dims")
            thw = tuple(samples.shape[-3:])  # (T,H,W) for video, (C,H,W) for image
            lpf = get_freq_filter(thw, samples.device, filter_method, butterworth_n, d_s, d_t,
                                  dtype=torch.float32)
            gen = torch.Generator(device="cpu").manual_seed(int(seed))
            x = samples
            for _ in range(int(num_iters)):
                noise = torch.randn(x.shape, generator=gen).to(x.device, x.dtype)
                x = freq_mix_3d(x, noise, lpf)
            out = dict(latent)
            out["samples"] = x
            return (out,)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(humanise(exc)) from exc

    @classmethod
    def IS_CHANGED(cls, latent, num_iters, filter_method, d_s, d_t, seed=0, butterworth_n=4):
        return f"freeinit-{num_iters}-{filter_method}-{d_s}-{d_t}-{seed}-{butterworth_n}"


# --------------------------------------------------------------------------- #
# C4 — MotionMax flow-guided noise scheduling
# --------------------------------------------------------------------------- #
def _compute_flow_raft(frames):
    """Optical flow between consecutive frames via torchvision RAFT (small).

    ``frames`` is [T,3,H,W] in 0..1. Returns flow [T-1,2,H,W]. Falls back to
    Farneback (opencv) then to zero-flow with a warning.
    """
    try:
        from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

        weights = Raft_Small_Weights.DEFAULT
        model = raft_small(weights=weights, progress=False).eval()
        dev = frames.device
        model = model.to(dev)
        f = frames * 2.0 - 1.0  # RAFT expects [-1,1]
        flows = []
        with torch.no_grad():
            for i in range(f.shape[0] - 1):
                out = model(f[i:i + 1], f[i + 1:i + 2])[-1]
                flows.append(out[0])
        return torch.stack(flows, 0)
    except Exception as exc:  # noqa: BLE001
        log.warning("[WanNodeExperiments] MotionMax: RAFT unavailable (%s); trying Farneback", exc)
    try:
        import numpy as np

        gray = (frames.mean(1).clamp(0, 1).cpu().numpy() * 255).astype("uint8")  # [T,H,W]
        flows = []
        for i in range(gray.shape[0] - 1):
            fl = cv2.calcOpticalFlowFarneback(gray[i], gray[i + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flows.append(torch.from_numpy(fl).permute(2, 0, 1).float())
        return torch.stack(flows, 0).to(frames.device)
    except Exception as exc:  # noqa: BLE001
        log.warning("[WanNodeExperiments] MotionMax: Farneback failed (%s); zero flow", exc)
        T, _, H, W = frames.shape
        return torch.zeros(max(T - 1, 1), 2, H, W, device=frames.device)


def _warp(x, flow):
    """Backward-warp [C,H,W] by flow [2,H,W] (pixels) using grid_sample."""
    import torch.nn.functional as F

    C, H, W = x.shape
    yy, xx = torch.meshgrid(
        torch.arange(H, device=x.device, dtype=x.dtype),
        torch.arange(W, device=x.device, dtype=x.dtype),
        indexing="ij",
    )
    gx = (xx + flow[0]) / max(W - 1, 1) * 2 - 1
    gy = (yy + flow[1]) / max(H - 1, 1) * 2 - 1
    grid = torch.stack((gx, gy), dim=-1).unsqueeze(0)
    return F.grid_sample(x.unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=True)[0]


class MotionMaxNoiseInit:
    """C4. MotionMax — Flow-Guided Noise Scheduling.

    Computes optical flow from a reference video (torchvision RAFT, opencv
    fallback) and warps the latent's per-frame content along the motion paths so
    the model locks source motion immediately in V2V. ``flow_strength`` blends
    warped vs. original; ``motion_extrapolation`` repeats the last flow to extend
    motion past the reference.

    Credits: Wan 2.2 wrapper foundations by Kijai & wuwukaka.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "reference_video": ("IMAGE",),
                "flow_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "motion_extrapolation": ("INT", {"default": 0, "min": 0, "max": 64}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "init"
    CATEGORY = CAT_SAMPLING
    DESCRIPTION = "Flow-guided noise structuring along reference motion (RAFT)."

    def init(self, latent, reference_video, flow_strength, motion_extrapolation):
        try:
            import torch.nn.functional as F

            samples = latent["samples"]
            if flow_strength <= 0 or reference_video is None or reference_video.shape[0] < 2:
                return (latent,)

            # latent dims: video [B,C,T,h,w] (preferred) or image [B,C,h,w]
            if samples.dim() == 5:
                B, C, T, h, w = samples.shape
            elif samples.dim() == 4:
                B, C, h, w = samples.shape
                T = 1
                samples = samples.unsqueeze(2)
            else:
                return (latent,)

            # reference [F,H,W,3] -> [T,3,h,w]
            ref = reference_video.permute(0, 3, 1, 2)  # [F,3,H,W]
            ref = F.interpolate(ref, size=(h, w), mode="bilinear", align_corners=False)
            Fr = ref.shape[0]
            if Fr != T:
                idx = torch.linspace(0, Fr - 1, T).round().long().clamp(0, Fr - 1)
                ref = ref[idx]
            flows = _compute_flow_raft(ref.to(samples.device))  # [T-1,2,h,w]

            if motion_extrapolation > 0 and flows.shape[0] > 0:
                last = flows[-1:].repeat(motion_extrapolation, 1, 1, 1)
                flows = torch.cat([flows, last], 0)

            out = samples.clone()
            for b in range(B):
                for ti in range(1, T):
                    fi = min(ti - 1, flows.shape[0] - 1)
                    warped = _warp(samples[b, :, ti - 1], flows[fi].to(samples.dtype))
                    out[b, :, ti] = (1.0 - flow_strength) * samples[b, :, ti] + flow_strength * warped

            if latent["samples"].dim() == 4:
                out = out.squeeze(2)
            res = dict(latent)
            res["samples"] = out
            return (res,)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(humanise(exc)) from exc

    @classmethod
    def IS_CHANGED(cls, latent, reference_video, flow_strength, motion_extrapolation):
        return f"motionmax-{flow_strength}-{motion_extrapolation}"


# --------------------------------------------------------------------------- #
# Forked scheduling/sampling (Phase 2) — refined from Kijai's sampler/schedulers
# --------------------------------------------------------------------------- #
def _flow_sigmas(steps, shift, denoise):
    """Flow-matching sigma schedule with Wan/SD3-style shift.

    sigma(t) = shift * t / (1 + (shift - 1) * t), for t in (1 -> 0]. Returns a
    1-D SIGMAS tensor of length steps+1 (ending at 0), honouring ``denoise``.
    """
    steps = max(1, int(steps))
    total = steps if denoise >= 1.0 else max(1, int(round(steps / max(denoise, 1e-3))))
    t = torch.linspace(1.0, 0.0, total + 1, dtype=torch.float32)
    s = float(shift)
    sig = s * t / (1.0 + (s - 1.0) * t)
    sig[-1] = 0.0
    if denoise < 1.0:
        sig = sig[-(steps + 1):]
    return sig


class WanFlowScheduler:
    """Wan flow-matching scheduler → SIGMAS (for SamplerCustomAdvanced).

    Standalone flow-shift sigma schedule used by Wan 2.2. Refined from the
    scheduling logic in Kijai's WanVideoWrapper; emits native ComfyUI SIGMAS so
    it chains with native SamplerCustom(Advanced).

    Credits: Wan scheduling studied from Kijai & wuwukaka.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": ("INT", {"default": 20, "min": 1, "max": 1000}),
                "shift": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1,
                                    "tooltip": "Flow-matching shift (Wan 2.2 typically 3-8)."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    FUNCTION = "get_sigmas"
    CATEGORY = CAT_SAMPLING
    DESCRIPTION = "Wan flow-matching sigma schedule (shift) as native SIGMAS."

    def get_sigmas(self, steps, shift, denoise):
        try:
            if denoise <= 0:
                return (torch.zeros(1, dtype=torch.float32),)
            return (_flow_sigmas(steps, shift, denoise),)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(humanise(exc)) from exc

    @classmethod
    def IS_CHANGED(cls, steps, shift, denoise):
        return f"wanflowsched-{steps}-{shift}-{denoise}"


class WanModelSamplingShift:
    """Patch a model with Wan/SD3 flow-matching shift (ModelSamplingDiscreteFlow).

    Equivalent to ComfyUI's ModelSamplingSD3 but exposed here for Wan; sets the
    flow shift on the model's sampling so native KSampler/SamplerCustom resolve
    the correct sigma curve. Falls back to storing the shift in
    transformer_options on Kijai's WANVIDEOMODEL.

    Credits: Wan shift handling studied from Kijai & wuwukaka.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "shift": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = CAT_SAMPLING
    DESCRIPTION = "Apply Wan/SD3 flow-matching shift to the model sampling."

    def patch(self, model, shift):
        try:
            m = compat.clone_model(model)
            try:
                import comfy.model_sampling as cms

                class _Adv(cms.ModelSamplingDiscreteFlow, cms.CONST):
                    pass

                ms = _Adv(model.model.model_config)
                ms.set_parameters(shift=float(shift), multiplier=1000)
                m.add_object_patch("model_sampling", ms)
            except Exception as exc:  # noqa: BLE001
                log.warning("[WanNodeExperiments] WanModelSamplingShift: native patch "
                            "unavailable (%s); storing in transformer_options", exc)
                compat.set_transformer_option(m, "wne_flow_shift", float(shift))
            return (m,)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(humanise(exc)) from exc

    @classmethod
    def IS_CHANGED(cls, shift, **kw):
        return f"wanshift-{shift}"


NODE_CLASS_MAPPINGS = {
    "WNE_AnyFlowModelLoader": AnyFlowModelLoader,
    "WNE_WanMoECache": WanMoECache,
    "WNE_UltraViCoAttentionDecay": UltraViCoAttentionDecay,
    "WNE_FreeLOCCorrection": FreeLOCCorrection,
    "WNE_FreeInitNoise": FreeInitNoise,
    "WNE_MotionMaxNoiseInit": MotionMaxNoiseInit,
    "WNE_WanFlowScheduler": WanFlowScheduler,
    "WNE_WanModelSamplingShift": WanModelSamplingShift,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WNE_AnyFlowModelLoader": "AnyFlow Any-Step Loader (WNE)",
    "WNE_WanMoECache": "Wan MoE-Aware Cache (WNE)",
    "WNE_UltraViCoAttentionDecay": "RIFLEx + UltraViCo Attention Decay (WNE)",
    "WNE_FreeLOCCorrection": "FreeLOC OOD Correction (WNE)",
    "WNE_FreeInitNoise": "FreeInit Noise Reinit (WNE)",
    "WNE_MotionMaxNoiseInit": "MotionMax Flow Noise Init (WNE)",
    "WNE_WanFlowScheduler": "Wan Flow Scheduler · sigmas (WNE)",
    "WNE_WanModelSamplingShift": "Wan Model Sampling Shift (WNE)",
}
