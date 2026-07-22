"""
guidance_nodes.py — guidance / sampling-time model patches.

Nodes: A2 APGGuidance, A3 ZeResFDGuidance, A7 STGGuidance,
       C1 MoECrossFadeRouter, C2 T5AttentionAmplifier, C3 HighFreqLatentInjector.

Credits: the model-patching approach was informed by studying **Kijai**
(ComfyUI-WanVideoWrapper) and **wuwukaka** (ComfyUI-WanAnimatePlus). No code is
imported from them — every patch below is a standalone reimplementation against
ComfyUI's native ``ModelPatcher`` hooks (``set_model_sampler_post_cfg_function``,
``set_model_unet_function_wrapper``) plus our portable attention interceptor, and
falls back gracefully on Kijai's ``WANVIDEOMODEL``.

Algorithm sources (cross-checked against public code):
  * APG      — Sadat et al. 2410.02416, mirrors comfy_extras/nodes_apg.py math.
  * ZeResFDG — Rychkovskiy 2510.12954 (FDG + energy rescale + zero-projection).
  * STG      — Hyung et al. 2411.18664, junhahyung/STGuidance (skip ST layers).
"""

from __future__ import annotations

import torch

from . import compat
from .compat import log, humanise
from .freq_utils import gaussian_blur_3d, laplacian_highfreq_3d

CATEGORY = "WanNodeExperiments/Guidance"


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _project(v0: torch.Tensor, v1: torch.Tensor):
    """Decompose v0 into components parallel/orthogonal to v1 (APG)."""
    dims = list(range(1, v1.ndim))
    v1n = torch.nn.functional.normalize(v1, dim=dims)
    par = (v0 * v1n).sum(dim=dims, keepdim=True) * v1n
    return par, v0 - par


class _StepState:
    """Infers a 0-based step index from the sigma sequence seen at sample time."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.last = None
        self.step = -1

    def update(self, sigma) -> int:
        try:
            s = float(sigma.flatten()[0]) if torch.is_tensor(sigma) else float(sigma)
        except Exception:  # noqa: BLE001
            s = None
        if s is None:
            self.step += 1
        elif self.last is None or s > self.last + 1e-9:
            self.step = 0  # sigma rose → new sampling run
        else:
            self.step += 1
        self.last = s
        return self.step


def _warn_kijai(node: str):
    log.warning(
        "[WanNodeExperiments] %s: native CFG/attention hooks are bypassed by "
        "Kijai's WanVideoSampler. Parameters were written to transformer_options "
        "and mapped to the closest equivalent where one exists; full effect "
        "requires the native ComfyUI sampler.", node,
    )


# --------------------------------------------------------------------------- #
# A2 — APG (Adaptive Projected Guidance)
# --------------------------------------------------------------------------- #
class APGGuidance:
    """A2. Adaptive Projected Guidance (Sadat et al., arXiv:2410.02416).

    Down-weights the component of the CFG update that is parallel to the
    conditional prediction (the part that causes oversaturation/tone drift),
    keeps the orthogonal detail component, and adds reverse momentum + a norm
    threshold. Lets you push high CFG without burn.

    Credits: Wan 2.2 wrapper foundations by Kijai & wuwukaka.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "eta": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01,
                                  "tooltip": "Reverse momentum on the guidance running average."}),
                "norm_threshold": ("FLOAT", {"default": 15.0, "min": 0.0, "max": 100.0, "step": 0.1,
                                             "tooltip": "Clamp guidance L2 norm (0 = off)."}),
                "parallel_weight": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                              "tooltip": "Keep this fraction of the parallel component (0 = full APG)."}),
            },
            "optional": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = CATEGORY
    DESCRIPTION = "Adaptive Projected Guidance: eliminate oversaturation at high CFG."

    def patch(self, model, eta, norm_threshold, parallel_weight, positive=None, negative=None):
        try:
            m = compat.clone_model(model)
            state = {"avg": None, "last_sigma": None}

            def post_cfg(args):
                cond = args["cond_denoised"]
                uncond = args["uncond_denoised"]
                scale = args["cond_scale"]
                sigma = args["sigma"]
                guidance = cond - uncond

                s = float(sigma.flatten()[0]) if torch.is_tensor(sigma) else float(sigma)
                if state["last_sigma"] is None or s > state["last_sigma"] + 1e-9:
                    state["avg"] = None
                state["last_sigma"] = s

                if eta != 0.0:
                    if state["avg"] is None:
                        state["avg"] = guidance.clone()
                    else:
                        state["avg"] = eta * state["avg"] + guidance
                    guidance = state["avg"]

                if norm_threshold > 0:
                    dims = list(range(1, guidance.ndim))
                    norm = guidance.norm(p=2, dim=dims, keepdim=True)
                    guidance = guidance * (norm_threshold / norm).clamp(max=1.0)

                par, orth = _project(guidance, cond)
                modified = orth + parallel_weight * par
                # Anchor at the conditional prediction with (scale-1), matching
                # APG (Sadat 2410.02416) and comfy_extras/nodes_apg.py. Anchoring
                # at uncond with `scale` double-counts once the guidance has been
                # projected/norm-clamped (at scale=1 it must reduce to `cond`).
                return cond + (scale - 1.0) * modified

            if not compat.set_post_cfg(m, post_cfg):
                _warn_kijai("APGGuidance")
                compat.set_transformer_option(m, "wne_apg", {
                    "eta": eta, "norm_threshold": norm_threshold, "parallel_weight": parallel_weight})
            return (m,)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(humanise(exc)) from exc

    @classmethod
    def IS_CHANGED(cls, eta, norm_threshold, parallel_weight, **kw):
        return f"apg-{eta}-{norm_threshold}-{parallel_weight}"


# --------------------------------------------------------------------------- #
# A3 — ZeResFDG (CADE 2.5)
# --------------------------------------------------------------------------- #
class ZeResFDGuidance:
    """A3. ZeResFDG — Frequency-Decoupled, Rescaled, Zero-Projected Guidance
    (Rychkovskiy, arXiv:2510.12954).

    1. Zero-projection: remove the part of the conditional prediction parallel
       to the unconditional one (early steps).
    2. Frequency-decoupled guidance: split the guidance delta into low/high
       spatial frequency and reweight (protect tone, boost micro-detail).
    3. Energy rescale: match the per-sample magnitude back to the cond branch.

    Stacks cleanly after APG (both are post-CFG functions).

    Credits: Wan 2.2 wrapper foundations by Kijai & wuwukaka.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "freq_split": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01,
                                         "tooltip": "Low-frequency gain (high gain = 2 - this)."}),
                "energy_rescale": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01,
                                             "tooltip": "Blend toward magnitude-matched prediction."}),
                "zero_proj_steps": ("INT", {"default": 3, "min": 0, "max": 1000,
                                            "tooltip": "Apply zero-projection for this many early steps."}),
            },
            "optional": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = CATEGORY
    DESCRIPTION = "Frequency-decoupled + rescaled + zero-projected guidance (CADE 2.5)."

    def patch(self, model, freq_split, energy_rescale, zero_proj_steps, positive=None, negative=None):
        try:
            m = compat.clone_model(model)
            step = _StepState()
            lam_low = float(freq_split)
            lam_high = 2.0 - float(freq_split)

            def post_cfg(args):
                cond = args["cond_denoised"]
                uncond = args["uncond_denoised"]
                scale = args["cond_scale"]
                idx = step.update(args["sigma"])
                dims = list(range(1, cond.ndim))

                cond_eff = cond
                if idx < zero_proj_steps:
                    num = (cond * uncond).sum(dim=dims, keepdim=True)
                    den = (uncond * uncond).sum(dim=dims, keepdim=True) + 1e-8
                    cond_eff = cond - (num / den) * uncond

                delta = cond_eff - uncond
                delta_low = gaussian_blur_3d(delta, spatial_sigma=1.5, temporal_sigma=0.0)
                delta_high = delta - delta_low
                delta_t = lam_low * delta_low + lam_high * delta_high

                y_cfg = uncond + scale * delta_t

                # energy rescale toward cond std
                std_c = cond.std(dim=dims, keepdim=True)
                std_y = y_cfg.std(dim=dims, keepdim=True) + 1e-8
                y_res = y_cfg * (std_c / std_y)
                return energy_rescale * y_res + (1.0 - energy_rescale) * y_cfg

            if not compat.set_post_cfg(m, post_cfg):
                _warn_kijai("ZeResFDGuidance")
                compat.set_transformer_option(m, "wne_zeresfdg", {
                    "freq_split": freq_split, "energy_rescale": energy_rescale,
                    "zero_proj_steps": zero_proj_steps})
            return (m,)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(humanise(exc)) from exc

    @classmethod
    def IS_CHANGED(cls, freq_split, energy_rescale, zero_proj_steps, **kw):
        return f"zeresfdg-{freq_split}-{energy_rescale}-{zero_proj_steps}"


# --------------------------------------------------------------------------- #
# A7 — STG (Spatiotemporal Skip Guidance)
# --------------------------------------------------------------------------- #
class _SkipModifier:
    """Attention modifier: skip (identity-passthrough) selected layer indices."""

    def __init__(self, skip_set):
        self.skip_set = skip_set

    def skip(self, layer_idx):
        return layer_idx in self.skip_set

    def bias(self, layer_idx, q, k):
        return None


class STGGuidance:
    """A7. Spatiotemporal Skip Guidance (Hyung et al., arXiv:2411.18664).

    Builds an implicit *weak model* by skipping selected spatiotemporal attention
    layers, then steers away from it: out + scale * (out - weak). No unconditional
    model required; preserves motion dynamics.

    The weak forward uses our portable attention interceptor (native ComfyUI). On
    Kijai's path it maps to his ``slg_args`` (skip-layer guidance) equivalent.

    Credits: Wan 2.2 wrapper foundations by Kijai & wuwukaka.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "skip_mode": (["attention", "residual"], {"default": "attention"}),
                "stg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
                "layer_indices": ("STRING", {"default": "8,9", "multiline": False,
                                             "tooltip": "Comma-separated attention layer indices to skip."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = CATEGORY
    DESCRIPTION = "Spatiotemporal Skip Guidance: self-perturbation weak model."

    @staticmethod
    def _parse_indices(s):
        out = set()
        for tok in str(s).replace(" ", "").split(","):
            if tok.lstrip("-").isdigit():
                out.add(int(tok))
        return out

    def patch(self, model, skip_mode, stg_scale, layer_indices):
        try:
            m = compat.clone_model(model)
            skip_set = self._parse_indices(layer_indices)

            def unet_wrapper(apply_model, args):
                x = args["input"]
                t = args["timestep"]
                c = args["c"]
                out = apply_model(x, t, **c)
                if stg_scale == 0 or not skip_set:
                    return out
                with compat.attention_modifier(_SkipModifier(skip_set)) as active:
                    if not active:
                        return out
                    weak = apply_model(x, t, **c)
                return out + stg_scale * (out - weak)

            if not compat.set_unet_wrapper(m, unet_wrapper):
                _warn_kijai("STGGuidance")
                compat.set_transformer_option(m, "wne_stg", {
                    "skip_mode": skip_mode, "stg_scale": stg_scale,
                    "skip_layers": sorted(skip_set)})
                # map onto Kijai's skip-layer-guidance convention if present
                compat.set_transformer_option(m, "slg_layers", sorted(skip_set))
            return (m,)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(humanise(exc)) from exc

    @classmethod
    def IS_CHANGED(cls, skip_mode, stg_scale, layer_indices, **kw):
        return f"stg-{skip_mode}-{stg_scale}-{layer_indices}"


# --------------------------------------------------------------------------- #
# C1 — MoE Cross-Fade Router
# --------------------------------------------------------------------------- #
class MoECrossFadeRouter:
    """C1. MoE Cross-Fade Router — smooth the Wan 2.2 high→low expert switch.

    Around ``switch_step``, a linear/cubic cross-fade blends predictions across a
    short window to kill the contrast pop / texture boil at the boundary.

    * With ``model_low`` connected → a *true* two-expert cross-fade: both experts
      are evaluated in the window and blended by the curve.
    * Without it → single-model boundary smoothing: the current prediction is
      blended with the previous step's prediction across the window (EMA), which
      still suppresses the boundary pop.

    Credits: Wan 2.2 MoE switch behaviour studied from Kijai & wuwukaka.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "switch_step": ("INT", {"default": 10, "min": 0, "max": 1000}),
                "fade_window": ("INT", {"default": 3, "min": 1, "max": 50}),
                "fade_curve": (["linear", "cubic"], {"default": "linear"}),
            },
            "optional": {"model_low": ("MODEL",)},
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "WanNodeExperiments/Sampling"
    DESCRIPTION = "Cross-fade across the MoE expert switch to remove boundary pops."

    @staticmethod
    def _weight(idx, switch_step, fade_window, curve):
        # 0 before window start, 1 after window end, ramped within
        start = switch_step - fade_window
        if idx <= start:
            return 0.0
        if idx >= switch_step + fade_window:
            return 1.0
        w = (idx - start) / max(1, (2 * fade_window))
        w = min(max(w, 0.0), 1.0)
        return w * w * (3 - 2 * w) if curve == "cubic" else w

    def patch(self, model, switch_step, fade_window, fade_curve, model_low=None):
        try:
            m = compat.clone_model(model)
            step = _StepState()
            prev = {"out": None}
            low_apply = None
            if model_low is not None:
                low_apply = getattr(getattr(model_low, "model", None), "apply_model", None)

            def unet_wrapper(apply_model, args):
                x, t, c = args["input"], args["timestep"], args["c"]
                idx = step.update(t)
                out = apply_model(x, t, **c)
                w = self._weight(idx, switch_step, fade_window, fade_curve)
                if w <= 0.0 or w >= 1.0:
                    if low_apply is None:
                        prev["out"] = out.detach()
                    return out
                if low_apply is not None:
                    try:
                        low_out = low_apply(x, t, **c)
                        return (1.0 - w) * out + w * low_out
                    except Exception:  # noqa: BLE001
                        pass  # fall through to EMA smoothing
                if prev["out"] is not None and prev["out"].shape == out.shape:
                    blended = (1.0 - 0.5 * w) * out + (0.5 * w) * prev["out"]
                else:
                    blended = out
                prev["out"] = out.detach()
                return blended

            if not compat.set_unet_wrapper(m, unet_wrapper):
                _warn_kijai("MoECrossFadeRouter")
                compat.set_transformer_option(m, "wne_moe_crossfade", {
                    "switch_step": switch_step, "fade_window": fade_window, "fade_curve": fade_curve})
            return (m,)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(humanise(exc)) from exc

    @classmethod
    def IS_CHANGED(cls, switch_step, fade_window, fade_curve, **kw):
        return f"moe-{switch_step}-{fade_window}-{fade_curve}"


# --------------------------------------------------------------------------- #
# C2 — T5 Token Attention Amplifier
# --------------------------------------------------------------------------- #
class _AmplifyModifier:
    """Boost salient cross-attention key tokens, optionally suppress stop-words.

    Cross-attention is detected by key length != query length. Token salience is
    proxied by per-key vector norm (content tokens carry more energy than padding
    / stop-words). The boost is applied as an additive log-bias on the key
    columns, which is mathematically equivalent to scaling the attention weights.

    TODO: exact noun/adjective vs stop-word tagging via the CLIP/T5 tokenizer for
    token-accurate amplification (the salience proxy is a strong heuristic).
    """

    def __init__(self, amplify, suppress_stopwords):
        self.amplify = float(amplify)
        self.suppress = bool(suppress_stopwords)

    def skip(self, layer_idx):
        return False

    def bias(self, layer_idx, q, k):
        # q,k expected (..., seq, dim); cross-attn when seq lengths differ
        if q.dim() < 2 or k.dim() < 2:
            return None
        q_len, k_len = q.shape[-2], k.shape[-2]
        if q_len == k_len:
            return None  # self-attention → leave untouched
        sal = k.float().norm(dim=-1)  # (..., k_len) per-key salience
        # normalise to [0,1]
        lo = sal.amin(dim=-1, keepdim=True)
        hi = sal.amax(dim=-1, keepdim=True)
        norm = (sal - lo) / (hi - lo + 1e-6)
        import math
        log_amp = math.log(max(self.amplify, 1e-3))
        bias = norm * log_amp  # high-salience keys boosted
        if self.suppress:
            bias = bias + (norm - 1.0) * abs(log_amp)  # low-salience keys suppressed
        # shape -> (..., 1, k_len) so it broadcasts over query rows
        return bias.unsqueeze(-2).to(q.dtype)


class T5AttentionAmplifier:
    """C2. T5 Token Attention Amplifier.

    Hooks DiT cross-attention to amplify structurally-important text tokens and
    suppress stop-words, countering T5 token dilution on long prompts.

    Credits: Wan 2.2 wrapper foundations by Kijai & wuwukaka.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "token_amplify_factor": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 5.0, "step": 0.05}),
                "suppress_stopwords": ("BOOLEAN", {"default": True}),
            },
            "optional": {"clip": ("CLIP",)},
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = CATEGORY
    DESCRIPTION = "Amplify important T5 tokens / suppress stop-words in cross-attention."

    def patch(self, model, token_amplify_factor, suppress_stopwords, clip=None):
        try:
            m = compat.clone_model(model)
            modifier = _AmplifyModifier(token_amplify_factor, suppress_stopwords)

            def unet_wrapper(apply_model, args):
                with compat.attention_modifier(modifier) as active:
                    if not active:
                        return apply_model(args["input"], args["timestep"], **args["c"])
                    return apply_model(args["input"], args["timestep"], **args["c"])

            if not compat.set_unet_wrapper(m, unet_wrapper):
                _warn_kijai("T5AttentionAmplifier")
                compat.set_transformer_option(m, "wne_t5_amplify", {
                    "factor": token_amplify_factor, "suppress_stopwords": suppress_stopwords})
            return (m,)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(humanise(exc)) from exc

    @classmethod
    def IS_CHANGED(cls, token_amplify_factor, suppress_stopwords, **kw):
        return f"t5amp-{token_amplify_factor}-{suppress_stopwords}"


# --------------------------------------------------------------------------- #
# C3 — High-Frequency Latent Injector (HFLI)
# --------------------------------------------------------------------------- #
class HighFreqLatentInjector:
    """C3. High-Frequency Latent Injector.

    During the final steps, extract the high-frequency residual (3D Laplacian =
    x − Gaussian blur) of the conditional prediction and inject it into the
    combined denoised result to restore texture lost at high CFG.

    Credits: Wan 2.2 wrapper foundations by Kijai & wuwukaka.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "start_step": ("INT", {"default": 15, "min": 0, "max": 1000}),
                "end_step": ("INT", {"default": 1000, "min": 0, "max": 1000}),
                "frequency_weight": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 2.0, "step": 0.01}),
                "blur_kernel_size": ("INT", {"default": 7, "min": 3, "max": 31, "step": 2}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = CATEGORY
    DESCRIPTION = "Inject conditional high-frequency texture during final steps."

    def patch(self, model, start_step, end_step, frequency_weight, blur_kernel_size):
        try:
            m = compat.clone_model(model)
            step = _StepState()
            sigma_eq = max(0.5, blur_kernel_size / 6.0)

            def post_cfg(args):
                idx = step.update(args["sigma"])
                denoised = args["denoised"]
                if frequency_weight == 0 or not (start_step <= idx <= end_step):
                    return denoised
                hf = laplacian_highfreq_3d(args["cond_denoised"], spatial_sigma=sigma_eq, temporal_sigma=0.0)
                return denoised + frequency_weight * hf

            if not compat.set_post_cfg(m, post_cfg):
                _warn_kijai("HighFreqLatentInjector")
                compat.set_transformer_option(m, "wne_hfli", {
                    "start_step": start_step, "end_step": end_step,
                    "frequency_weight": frequency_weight, "blur_kernel_size": blur_kernel_size})
            return (m,)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(humanise(exc)) from exc

    @classmethod
    def IS_CHANGED(cls, start_step, end_step, frequency_weight, blur_kernel_size, **kw):
        return f"hfli-{start_step}-{end_step}-{frequency_weight}-{blur_kernel_size}"


# --------------------------------------------------------------------------- #
# Phase 3 — refined CFG techniques (forked from Kijai's guidance set), standalone
# native post-CFG patches that stack with APG/ZeResFDG.
# --------------------------------------------------------------------------- #
class RescaleCFG:
    """RescaleCFG (Lin et al., "Common Diffusion Noise Schedules").

    Rescales the CFG result so its per-sample std matches the conditional branch,
    blended by ``multiplier`` — prevents the over-exposure/burn that high CFG
    causes on Wan. Native post-CFG; stacks with APG/ZeResFDG.

    Credits: refined from the rescale logic in Kijai's WanVideoWrapper.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "multiplier": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = CATEGORY
    DESCRIPTION = "Rescale CFG to the conditional std (anti-burn)."

    def patch(self, model, multiplier):
        try:
            m = compat.clone_model(model)

            def post_cfg(args):
                cond = args["cond_denoised"]
                uncond = args["uncond_denoised"]
                denoised = args["denoised"]  # standard cfg result
                dims = list(range(1, cond.ndim))
                ro_pos = cond.std(dim=dims, keepdim=True)
                ro_cfg = denoised.std(dim=dims, keepdim=True) + 1e-8
                rescaled = denoised * (ro_pos / ro_cfg)
                return multiplier * rescaled + (1.0 - multiplier) * denoised

            if not compat.set_post_cfg(m, post_cfg):
                _warn_kijai("RescaleCFG")
                compat.set_transformer_option(m, "wne_rescale_cfg", {"multiplier": multiplier})
            return (m,)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(humanise(exc)) from exc

    @classmethod
    def IS_CHANGED(cls, multiplier, **kw):
        return f"rescalecfg-{multiplier}"


class CFGZeroStar:
    """CFG-Zero★ (arXiv:2503.18886).

    Two parts: (1) a per-sample optimized scale ``s*`` for the unconditional
    branch (s* = <cond,uncond>/||uncond||²) so the combine is
    ``s*·uncond + scale·(cond − s*·uncond)``; (2) zero-init — output ~0 for the
    first ``zero_init_steps`` so early noise doesn't get over-steered.

    Credits: refined from Kijai's ``cfg_zero_star`` experimental option.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "zero_init_steps": ("INT", {"default": 1, "min": 0, "max": 100,
                                            "tooltip": "Number of first steps to zero-init."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = CATEGORY
    DESCRIPTION = "CFG-Zero*: optimized uncond scale + early-step zero-init."

    def patch(self, model, zero_init_steps):
        try:
            m = compat.clone_model(model)
            step = _StepState()

            def post_cfg(args):
                cond = args["cond_denoised"]
                uncond = args["uncond_denoised"]
                scale = args["cond_scale"]
                idx = step.update(args["sigma"])
                if idx < zero_init_steps:
                    return cond * 0.0
                dims = list(range(1, cond.ndim))
                num = (cond * uncond).sum(dim=dims, keepdim=True)
                den = (uncond * uncond).sum(dim=dims, keepdim=True) + 1e-8
                alpha = num / den
                return alpha * uncond + scale * (cond - alpha * uncond)

            if not compat.set_post_cfg(m, post_cfg):
                _warn_kijai("CFGZeroStar")
                compat.set_transformer_option(m, "cfg_zero_star", True)
                compat.set_transformer_option(m, "wne_cfg_zero_star", {"zero_init_steps": zero_init_steps})
            return (m,)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(humanise(exc)) from exc

    @classmethod
    def IS_CHANGED(cls, zero_init_steps, **kw):
        return f"cfgzero-{zero_init_steps}"


class TangentialDampingCFG:
    """Tangential-Damping CFG (TCFG family, arXiv:2503.18137).

    Damps the *tangential* (orthogonal) component of the unconditional
    prediction — the part misaligned with the conditional one, which injects
    off-prompt artifacts — while keeping the aligned (parallel) component. At
    ``damping``=1 the negative branch collapses onto its projection onto cond.
    ``damping`` = how much of the tangential part to remove.

    Credits: refined from Kijai's ``use_tcfg`` (tangential CFG) option.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "damping": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = CATEGORY
    DESCRIPTION = "Tangential-damping CFG: damp the radial part of the negative branch."

    def patch(self, model, damping):
        try:
            m = compat.clone_model(model)

            def post_cfg(args):
                cond = args["cond_denoised"]
                uncond = args["uncond_denoised"]
                scale = args["cond_scale"]
                _par, orth = _project(uncond, cond)  # orth = uncond's tangential (off-cond) part
                uncond_damped = uncond - damping * orth
                return uncond_damped + scale * (cond - uncond_damped)

            if not compat.set_post_cfg(m, post_cfg):
                _warn_kijai("TangentialDampingCFG")
                compat.set_transformer_option(m, "use_tcfg", True)
                compat.set_transformer_option(m, "wne_tcfg", {"damping": damping})
            return (m,)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(humanise(exc)) from exc

    @classmethod
    def IS_CHANGED(cls, damping, **kw):
        return f"tcfg-{damping}"


NODE_CLASS_MAPPINGS = {
    "WNE_APGGuidance": APGGuidance,
    "WNE_ZeResFDGuidance": ZeResFDGuidance,
    "WNE_STGGuidance": STGGuidance,
    "WNE_MoECrossFadeRouter": MoECrossFadeRouter,
    "WNE_T5AttentionAmplifier": T5AttentionAmplifier,
    "WNE_HighFreqLatentInjector": HighFreqLatentInjector,
    "WNE_RescaleCFG": RescaleCFG,
    "WNE_CFGZeroStar": CFGZeroStar,
    "WNE_TangentialDampingCFG": TangentialDampingCFG,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WNE_APGGuidance": "APG Guidance (WNE)",
    "WNE_ZeResFDGuidance": "ZeResFDG Guidance · CADE 2.5 (WNE)",
    "WNE_STGGuidance": "STG Spatiotemporal Skip Guidance (WNE)",
    "WNE_MoECrossFadeRouter": "MoE Cross-Fade Router (WNE)",
    "WNE_T5AttentionAmplifier": "T5 Token Attention Amplifier (WNE)",
    "WNE_HighFreqLatentInjector": "High-Frequency Latent Injector (WNE)",
    "WNE_RescaleCFG": "Rescale CFG · anti-burn (WNE)",
    "WNE_CFGZeroStar": "CFG-Zero★ (WNE)",
    "WNE_TangentialDampingCFG": "Tangential-Damping CFG (WNE)",
}
