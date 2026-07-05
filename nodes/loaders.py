"""
loaders.py — unified Wan 2.2 model loaders (Phase 2).

Forked/refined from the loading patterns in **Kijai**'s ComfyUI-WanVideoWrapper
(`nodes_model_loading.py`, `gguf/`) and **wuwukaka**'s ComfyUI-WanAnimatePlus,
both Apache-2.0. Studied to extract: base-precision/fp8 dtype handling, GGUF
dequant, VACE module loading, and the Wan 2.2 high/low-noise MoE expert split.

Design choice (deliberate, documented in MODIFICATIONS.md): these loaders emit
the **native ComfyUI `MODEL`** type via ``comfy.sd`` rather than Kijai's private
``WANVIDEOMODEL``. That keeps the pack truly standalone (no diffsynth / no
vendored model module) and 100% compatible with native ``KSampler`` *and* every
node in this pack — while my dual-mode ``compat`` layer still maps onto Kijai's
``WANVIDEOMODEL`` when one is passed in elsewhere.

GPU: weights load straight to the offload device where possible; fp8 dtypes and
an explicit ``force_offload`` (soft cache empty + gc) cover low-VRAM use.

All ``comfy`` / ``folder_paths`` imports are lazy so this module imports cleanly
without a running ComfyUI (for the standalone import check).
"""

from __future__ import annotations

import gc

import torch

from .compat import log, humanise

CATEGORY = "WanNodeExperiments/Loaders"

_DTYPES = ["default", "fp16", "bf16", "fp8_e4m3fn", "fp8_e5m2"]


def _folder_list(*keys):
    """Filenames for given folder_paths keys; safe if folder_paths is absent."""
    try:
        import folder_paths

        out = []
        for k in keys:
            try:
                out += folder_paths.get_filename_list(k)
            except Exception:  # noqa: BLE001
                pass
        # de-dupe, keep order
        seen = set()
        return [x for x in out if not (x in seen or seen.add(x))]
    except Exception:  # noqa: BLE001
        return []


def _model_field(*keys):
    lst = _folder_list(*keys)
    if lst:
        return (lst, {"tooltip": "Model file from ComfyUI/models/diffusion_models (or unet)."})
    return ("STRING", {"default": "", "tooltip": "Path/name of the diffusion model file."})


def _dtype_options(name):
    """Map a dtype label to comfy load_diffusion_model model_options."""
    opts = {}
    if name == "fp16":
        opts["dtype"] = torch.float16
    elif name == "bf16":
        opts["dtype"] = torch.bfloat16
    elif name == "fp8_e4m3fn":
        opts["dtype"] = torch.float8_e4m3fn
        opts["fp8_optimizations"] = True
    elif name == "fp8_e5m2":
        opts["dtype"] = torch.float8_e5m2
        opts["fp8_optimizations"] = True
    return opts


def _resolve_path(name):
    import folder_paths

    for key in ("diffusion_models", "unet", "unet_gguf"):
        try:
            p = folder_paths.get_full_path(key, name)
            if p:
                return p
        except Exception:  # noqa: BLE001
            pass
    # already an absolute/relative path?
    import os

    if os.path.exists(name):
        return name
    raise FileNotFoundError(name)


def _clean_vram():
    try:
        import comfy.model_management as mm

        mm.soft_empty_cache()
    except Exception:  # noqa: BLE001
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_native(name, weight_dtype, force_offload):
    """Load a diffusion model file → native ComfyUI MODEL (ModelPatcher)."""
    import comfy.sd

    path = _resolve_path(name)
    model_options = _dtype_options(weight_dtype)
    with torch.inference_mode():
        model = comfy.sd.load_diffusion_model(path, model_options=model_options)
    if force_offload:
        _clean_vram()
    log.info("[WanNodeExperiments] loaded Wan model '%s' (dtype=%s)", name, weight_dtype)
    return model


# --------------------------------------------------------------------------- #
# Unified T2V / I2V loader
# --------------------------------------------------------------------------- #
class WanModelLoader:
    """Unified Wan 2.2 T2V/I2V model loader (native MODEL).

    Credits: loading patterns from Kijai (ComfyUI-WanVideoWrapper) & wuwukaka
    (ComfyUI-WanAnimatePlus). T2V vs I2V is a conditioning difference, not a
    loading one — both load the same diffusion transformer here.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": _model_field("diffusion_models", "unet"),
                "weight_dtype": (_DTYPES, {"default": "default"}),
            },
            "optional": {
                "force_offload": ("BOOLEAN", {"default": False,
                                              "tooltip": "Empty VRAM cache after load (low-VRAM)."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = "Load a Wan 2.2 T2V/I2V diffusion model as native ComfyUI MODEL."

    def load(self, model_name, weight_dtype, force_offload=False):
        try:
            return (_load_native(model_name, weight_dtype, force_offload),)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(humanise(exc)) from exc


# --------------------------------------------------------------------------- #
# Wan 2.2 MoE expert loader (high-noise + low-noise)
# --------------------------------------------------------------------------- #
class WanMoEExpertLoader:
    """Load the Wan 2.2 MoE pair: high-noise + low-noise expert models.

    Outputs two native MODELs ready for ``WanMoECache`` / ``MoECrossFadeRouter``.

    Credits: Wan 2.2 MoE high/low-noise split studied from Kijai & wuwukaka.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "high_noise_model": _model_field("diffusion_models", "unet"),
                "low_noise_model": _model_field("diffusion_models", "unet"),
                "weight_dtype": (_DTYPES, {"default": "default"}),
            },
            "optional": {
                "force_offload": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("MODEL", "MODEL")
    RETURN_NAMES = ("model_high", "model_low")
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = "Load Wan 2.2 high-noise + low-noise MoE expert models."

    def load(self, high_noise_model, low_noise_model, weight_dtype, force_offload=True):
        try:
            mh = _load_native(high_noise_model, weight_dtype, force_offload)
            ml = _load_native(low_noise_model, weight_dtype, force_offload)
            return (mh, ml)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(humanise(exc)) from exc


# --------------------------------------------------------------------------- #
# VACE module loader
# --------------------------------------------------------------------------- #
class WanVACELoader:
    """Load a Wan VACE control module file as native MODEL.

    VACE adds control blocks to Wan; here it is loaded as a diffusion model so it
    can be used standalone or merged downstream.

    TODO: optional merge of VACE blocks into a base model's state dict (Kijai
    prefixes ``vace_blocks.`` → ``diffusion_model.vace_blocks.``) for a single
    combined MODEL; for now the VACE model is returned on its own.

    Credits: VACE handling studied from Kijai (ComfyUI-WanVideoWrapper).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vace_model": _model_field("diffusion_models", "unet"),
                "weight_dtype": (_DTYPES, {"default": "default"}),
            },
            "optional": {"force_offload": ("BOOLEAN", {"default": False})},
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = "Load a Wan VACE control module as native ComfyUI MODEL."

    def load(self, vace_model, weight_dtype, force_offload=False):
        try:
            return (_load_native(vace_model, weight_dtype, force_offload),)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(humanise(exc)) from exc


# --------------------------------------------------------------------------- #
# GGUF loader
# --------------------------------------------------------------------------- #
class WanGGUFModelLoader:
    """Load a GGUF-quantized Wan model → native MODEL.

    Reads the GGUF with the ``gguf`` library, dequantizes tensors to fp16, and
    builds a native model via ``comfy.sd.load_diffusion_model_state_dict``. If
    the ``gguf`` package's dequantizer is unavailable for a given quant type, a
    clear error is raised (install/route through ComfyUI-GGUF for K-quants).

    Credits: GGUF approach adapted from Kijai's gguf path (after city96's GGUF
    nodes), Apache-2.0.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gguf_name": _model_field("unet_gguf", "diffusion_models"),
            },
            "optional": {"force_offload": ("BOOLEAN", {"default": True})},
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = "Load a GGUF-quantized Wan model as native ComfyUI MODEL."

    def load(self, gguf_name, force_offload=True):
        try:
            import comfy.sd
            import gguf as _gguf
            import numpy as np

            path = _resolve_path(gguf_name)
            reader = _gguf.GGUFReader(path)
            sd = {}
            for tensor in reader.tensors:
                name = str(tensor.name)
                qtype = tensor.tensor_type
                if qtype in (_gguf.GGMLQuantizationType.F32, _gguf.GGMLQuantizationType.F16):
                    arr = np.array(tensor.data)
                else:
                    try:
                        arr = _gguf.quants.dequantize(np.array(tensor.data), qtype)
                    except Exception as exc:  # noqa: BLE001
                        raise RuntimeError(
                            f"ModuleNotFoundError: GGUF quant {qtype} not dequantizable by the "
                            f"installed gguf lib — install/route through ComfyUI-GGUF. ({exc})")
                t = torch.from_numpy(arr.astype(np.float16, copy=False))
                sd[name] = t.reshape(tuple(reversed(tensor.shape)))
            with torch.inference_mode():
                model = comfy.sd.load_diffusion_model_state_dict(sd)
            if model is None:
                raise RuntimeError("RuntimeError: could not detect a diffusion model in the GGUF.")
            if force_offload:
                _clean_vram()
            log.info("[WanNodeExperiments] loaded GGUF Wan model '%s' (%d tensors)", gguf_name, len(sd))
            return (model,)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(humanise(exc)) from exc


NODE_CLASS_MAPPINGS = {
    "WNE_WanModelLoader": WanModelLoader,
    "WNE_WanMoEExpertLoader": WanMoEExpertLoader,
    "WNE_WanVACELoader": WanVACELoader,
    "WNE_WanGGUFModelLoader": WanGGUFModelLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WNE_WanModelLoader": "Wan Model Loader · T2V/I2V (WNE)",
    "WNE_WanMoEExpertLoader": "Wan MoE Expert Loader · high/low (WNE)",
    "WNE_WanVACELoader": "Wan VACE Loader (WNE)",
    "WNE_WanGGUFModelLoader": "Wan GGUF Model Loader (WNE)",
}
