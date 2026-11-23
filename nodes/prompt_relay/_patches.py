# Prompt Relay — model patching backends.
#
# Three backends:
#   1. ``patch_native(model_clone, mask_fn)`` — native ComfyUI ``MODEL``
#      objects (Wan + LTX). Uses ``ModelPatcher.add_object_patch`` which every
#      native sampler honours, including third-party custom samplers built on
#      top of ``SamplerCustomAdvanced``.
#   2. ``patch_kijai(patcher, mask_fn)`` — Kijai ``WanVideoWrapper`` models.
#      Kijai's ``WanVideoSampler`` bypasses ``object_patches`` entirely, so we
#      rebind ``block.cross_attn.forward`` directly on the live ``WanModel``
#      instance. Idempotent and restorable.
#   3. ``patch_generic(model_clone, mask_fn)`` — fallback that introspects any
#      third-party diffusion model: looks for ``blocks[i].cross_attn`` modules
#      and wires the same mask-aware forward.
#
# All three reuse the masked-attention call defined here.

from __future__ import annotations

import logging
import types
from typing import Callable, Optional, Tuple

import torch

log = logging.getLogger("C2C.PromptRelay.patches")


def _masked_attention(q, k, v, heads, mask, transformer_options=None, **kwargs):
    """Force the masked path through ``attention_pytorch`` — sage/flash kernels often
    ignore additive masks. ``_inside_attn_wrapper=True`` skips the wrap_attn dispatch."""
    import comfy.ldm.modules.attention as _attn
    return _attn.attention_pytorch(
        q, k, v, heads,
        mask=mask,
        _inside_attn_wrapper=True,
        transformer_options=transformer_options or {},
        **kwargs,
    )


# ────────────────────────────────────────────────────────────────────────
# Forward implementations — Wan T2V / Wan I2V / LTX
# ────────────────────────────────────────────────────────────────────────

def _wan_t2v_forward(self_module, mask_fn, x, context, transformer_options=None, **kwargs):
    import comfy.ldm.modules.attention as _attn
    transformer_options = transformer_options or {}
    q = self_module.norm_q(self_module.q(x))
    k = self_module.norm_k(self_module.k(context))
    v = self_module.v(context)
    mask = mask_fn(q, k, transformer_options)
    if mask is not None:
        out = _masked_attention(q, k, v, heads=self_module.num_heads, mask=mask,
                                transformer_options=transformer_options)
    else:
        out = _attn.optimized_attention(q, k, v, heads=self_module.num_heads,
                                        transformer_options=transformer_options)
    return self_module.o(out)


def _wan_i2v_forward(self_module, mask_fn, x, context, context_img_len,
                     transformer_options=None, **kwargs):
    import comfy.ldm.modules.attention as _attn
    transformer_options = transformer_options or {}
    context_img = context[:, :context_img_len]
    context_text = context[:, context_img_len:]
    q = self_module.norm_q(self_module.q(x))

    k_img = self_module.norm_k_img(self_module.k_img(context_img))
    v_img = self_module.v_img(context_img)
    img_x = _attn.optimized_attention(q, k_img, v_img, heads=self_module.num_heads,
                                       transformer_options=transformer_options)

    k = self_module.norm_k(self_module.k(context_text))
    v = self_module.v(context_text)
    mask = mask_fn(q, k, transformer_options)
    if mask is not None:
        out = _masked_attention(q, k, v, heads=self_module.num_heads, mask=mask,
                                transformer_options=transformer_options)
    else:
        out = _attn.optimized_attention(q, k, v, heads=self_module.num_heads,
                                        transformer_options=transformer_options)
    return self_module.o(out + img_x)


def _ltx_forward(self_module, mask_fn, x, context=None, mask=None, pe=None,
                 k_pe=None, transformer_options=None):
    import comfy.ldm.modules.attention as _attn
    from comfy.ldm.lightricks.model import apply_rotary_emb
    transformer_options = transformer_options or {}
    is_self_attn = context is None
    context = x if is_self_attn else context

    q = self_module.q_norm(self_module.to_q(x))
    k = self_module.k_norm(self_module.to_k(context))
    v = self_module.to_v(context)
    if pe is not None:
        q = apply_rotary_emb(q, pe)
        k = apply_rotary_emb(k, pe if k_pe is None else k_pe)

    if not is_self_attn:
        temporal_mask = mask_fn(q, k, transformer_options)
        if temporal_mask is not None:
            mask = temporal_mask if mask is None else mask + temporal_mask

    if mask is None:
        out = _attn.optimized_attention(q, k, v, self_module.heads,
                                        attn_precision=self_module.attn_precision,
                                        transformer_options=transformer_options)
    else:
        out = _masked_attention(q, k, v, self_module.heads, mask=mask,
                                attn_precision=self_module.attn_precision,
                                transformer_options=transformer_options)

    if self_module.to_gate_logits is not None:
        gate_logits = self_module.to_gate_logits(x)
        b, t, _ = out.shape
        out = out.view(b, t, self_module.heads, self_module.dim_head)
        out = out * (2.0 * torch.sigmoid(gate_logits)).unsqueeze(-1)
        out = out.view(b, t, self_module.heads * self_module.dim_head)
    return self_module.to_out(out)


# ────────────────────────────────────────────────────────────────────────
# Model architecture detection
# ────────────────────────────────────────────────────────────────────────

def detect_native_arch(model) -> Tuple[str, Tuple[int, int, int], int]:
    """Detect arch / patch-size / temporal stride for native ComfyUI MODEL objects.

    Returns ``(arch, patch_size, temporal_stride)``. Raises if the model is
    neither native Wan nor native LTX.
    """
    diff_model = model.model.diffusion_model

    # Native Wan: has patch_size attr, no patchifier.
    if hasattr(diff_model, "patch_size") and not hasattr(diff_model, "patchifier"):
        # Reject Kijai's WanModel — its module path lives outside comfy.ldm.
        mod_path = getattr(type(diff_model), "__module__", "") or ""
        if "comfy.ldm" in mod_path or mod_path.startswith("comfy"):
            return "wan", tuple(diff_model.patch_size), 4

    # Native LTX: has patchifier + vae_scale_factors.
    if hasattr(diff_model, "patchifier") and hasattr(diff_model, "vae_scale_factors"):
        return "ltx", (1, 1, 1), int(diff_model.vae_scale_factors[0])

    raise ValueError(
        f"Unsupported native model: {type(diff_model).__name__} from "
        f"{getattr(type(diff_model), '__module__', '?')}. "
        "Use the Kijai variant for WANVIDEOMODEL inputs, or rely on the "
        "generic-fallback patcher."
    )


# ────────────────────────────────────────────────────────────────────────
# Backend 1: native ComfyUI MODEL (uses ModelPatcher.add_object_patch)
# ────────────────────────────────────────────────────────────────────────

class _CrossAttnPatch:
    """Bound-method factory for ``add_object_patch`` — restores ``self`` to the patched module."""

    def __init__(self, impl: Callable, mask_fn: Callable):
        self.impl = impl
        self.mask_fn = mask_fn

    def __get__(self, obj, objtype=None):
        impl, mask_fn = self.impl, self.mask_fn

        def wrapped(self_module, *args, **kwargs):
            return impl(self_module, mask_fn, *args, **kwargs)

        return types.MethodType(wrapped, obj)


def _check_unpatched(model_clone, key: str) -> None:
    if key in getattr(model_clone, "object_patches", {}):
        raise RuntimeError(
            f"PromptRelay: cross-attention forward at {key!r} is already patched "
            "by another node (e.g. KJNodes NAG). Stacking is not supported — "
            "remove the conflicting node."
        )


def patch_native(model_clone, arch: str, mask_fn: Callable) -> None:
    """Apply Prompt Relay to a native ComfyUI ``MODEL`` via ``add_object_patch``."""
    diffusion_model = model_clone.get_model_object("diffusion_model")

    if arch == "wan":
        from comfy.ldm.wan.model import WanI2VCrossAttention
        for idx, block in enumerate(diffusion_model.blocks):
            key = f"diffusion_model.blocks.{idx}.cross_attn.forward"
            _check_unpatched(model_clone, key)
            cross_attn = block.cross_attn
            impl = _wan_i2v_forward if isinstance(cross_attn, WanI2VCrossAttention) else _wan_t2v_forward
            model_clone.add_object_patch(
                key, _CrossAttnPatch(impl, mask_fn).__get__(cross_attn, cross_attn.__class__)
            )
        return

    if arch == "ltx":
        for idx, block in enumerate(diffusion_model.transformer_blocks):
            for attr in ("attn2", "audio_attn2"):
                module = getattr(block, attr, None)
                if module is None:
                    continue
                key = f"diffusion_model.transformer_blocks.{idx}.{attr}.forward"
                _check_unpatched(model_clone, key)
                model_clone.add_object_patch(
                    key, _CrossAttnPatch(_ltx_forward, mask_fn).__get__(module, module.__class__)
                )
        return

    raise ValueError(f"Unknown native arch: {arch}")


# ────────────────────────────────────────────────────────────────────────
# Backend 2: Kijai WanVideoWrapper (in-place rebind, idempotent)
# ────────────────────────────────────────────────────────────────────────

_KIJAI_ORIGINAL_ATTR = "_c2c_prompt_relay_orig_forward"
_KIJAI_MARKER_ATTR = "_c2c_prompt_relay_active"


def _kijai_class_dispatch(cross_attn) -> Callable:
    """Pick the right forward impl based on Kijai class name (T2V / I2V / HuMo)."""
    name = type(cross_attn).__name__
    if name in ("WanI2VCrossAttention",):
        return _wan_i2v_forward
    if name in ("WanT2VCrossAttention", "AudioCrossAttention", "WanHuMoCrossAttention"):
        return _wan_t2v_forward
    # Default to T2V signature — works for any subclass of WanSelfAttention.
    return _wan_t2v_forward


_KIJAI_PORTED = ("WanT2VCrossAttention", "WanI2VCrossAttention")


def _kijai_layout(cross_attn) -> bool:
    """True when the module's forward takes Kijai's (x, context, grid_sizes, clip_embed, ...) layout."""
    import inspect
    try:
        params = list(inspect.signature(type(cross_attn).forward).parameters)
    except (TypeError, ValueError):
        return False
    third = params[3] if len(params) > 3 else None
    return third == "grid_sizes" or "clip_embed" in params


def _kijai_signature_problem(cross_attn) -> Optional[str]:
    """L7.128: a Kijai-layout cross-attention Prompt Relay has no port for (the T2V / I2V ones are ported below)."""
    if not _kijai_layout(cross_attn) or type(cross_attn).__name__ in _KIJAI_PORTED:
        return None
    return f"Prompt Relay has no Kijai port for {type(cross_attn).__name__}; those blocks run unchanged."


# -- L7.128: Prompt Relay on Kijai's own cross-attention layout ---------------------------------------------------
# Kijai's WanAttentionBlock calls cross_attn(x, context, grid_sizes, clip_embed=..., ...). The port below reproduces
# WanT2VCrossAttention / WanI2VCrossAttention.forward's PLAIN path (text attention, plus the CLIP image attention for
# I2V) with Kijai's own projection order and attention(); only the text attention takes the Prompt Relay mask. Every
# other Kijai feature path (NAG, LongCat, EchoShot RoPE, Lynx, audio, FantasyPortrait adapter, Ovi fusion) is handed
# to Kijai's original forward unchanged (relay off for that call, said once in the log). With no mask the port is
# bit-identical to Kijai's forward (tests/test_l7128_kijai_relay_port.py runs Kijai's real classes on CPU).
_KIJAI_STATE_ATTR = "_c2c_prompt_relay_state"
_KIJAI_TFWD_ATTR = "_c2c_prompt_relay_orig_tforward"
_RELAY_OFF_LOGGED: set = set()


def _kijai_relay_off_reason(self_module, x, kwargs) -> Optional[str]:
    if kwargs.get("nag_context") is not None:
        return "NAG"
    if x.shape[-1] == 4096:
        return "LongCat"
    if kwargs.get("inner_t") is not None and kwargs.get("cross_freqs") is not None:
        return "EchoShot"
    if kwargs.get("lynx_x_ip") is not None:
        return "Lynx"
    if kwargs.get("audio_proj") is not None:
        return "audio"
    if kwargs.get("adapter_proj") is not None:
        return "FantasyPortrait adapter"
    if getattr(self_module, "k_fusion", None) is not None:
        return "Ovi fusion"
    if not hasattr(getattr(self_module, "norm_q", None), "weight") or \
            not hasattr(getattr(self_module, "norm_k", None), "weight"):
        return "no q/k norm"
    return None


def _kijai_grid(grid_sizes):
    """Kijai passes grid_sizes as a [B, 3] tensor (or a list of (F, H, W)); the mask wants one (F, H, W)."""
    if grid_sizes is None:
        return None
    try:
        if torch.is_tensor(grid_sizes):
            g = grid_sizes[0] if grid_sizes.ndim == 2 else grid_sizes
            return tuple(int(v) for v in g.tolist())
        first = grid_sizes[0]
        g = first if (isinstance(first, (list, tuple)) or torch.is_tensor(first)) else grid_sizes
        return tuple(int(v) for v in g)
    except Exception:  # noqa: BLE001 - an unknown layout means "use the fallback tokens per frame"
        return None


def _masked_sdpa(q, k, v, mask):
    """q, k, v in Kijai's [B, L, heads, dim] layout; mask additive [Lq, Lk]. Returns [B, Lq, heads*dim]."""
    out = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask.to(q.dtype))
    return out.transpose(1, 2).flatten(2)


def _kijai_cross_forward(self_module, mask_fn, x, context, grid_sizes=None, clip_embed=None, **kwargs):
    import sys
    original = getattr(self_module, _KIJAI_ORIGINAL_ATTR)
    attention = getattr(sys.modules.get(type(self_module).__module__), "attention", None)
    reason = _kijai_relay_off_reason(self_module, x, kwargs) or (None if attention else "no attention()")
    i2v = type(self_module).__name__ == "WanI2VCrossAttention"
    if i2v and clip_embed is not None and not hasattr(getattr(self_module, "norm_k_img", None), "weight"):
        reason = reason or "no image k norm"
    if reason:
        key = (type(self_module).__name__, reason)
        if key not in _RELAY_OFF_LOGGED:
            _RELAY_OFF_LOGGED.add(key)
            log.warning("Prompt Relay is off for %s calls using %s (Kijai feature path); they run unchanged.",
                        type(self_module).__name__, reason)
        return original(x, context, grid_sizes, clip_embed=clip_embed, **kwargs)

    b, n, d = x.size(0), self_module.num_heads, self_module.head_dim
    chunks = 2 if kwargs.get("rope_func", "comfy") == "comfy_chunked" else 1
    q = self_module.norm_q(self_module.q(x).to(self_module.norm_q.weight.dtype), num_chunks=chunks)
    q = q.view(b, -1, n, d).to(x.dtype) if i2v else q.to(x.dtype).view(b, -1, n, d)
    k = self_module.norm_k(self_module.k(context).to(self_module.norm_k.weight.dtype))
    k = k.view(b, -1, n, d).to(x.dtype) if i2v else k.to(x.dtype).view(b, -1, n, d)
    v = self_module.v(context).view(b, -1, n, d)

    state = getattr(self_module, _KIJAI_STATE_ATTR, None) or {}
    opts = {"cond_or_uncond": [1] if state.get("uncond") else [0]}
    grid = _kijai_grid(grid_sizes)
    if grid is not None:
        opts["grid_sizes"] = grid
    mask = mask_fn(q, k, opts)
    if mask is not None:
        out = _masked_sdpa(q, k, v, mask)
    else:
        out = attention(q, k, v, attention_mode=self_module.attention_mode, heads=n).flatten(2)

    if i2v and clip_embed is not None:
        k_img = self_module.norm_k_img(self_module.k_img(clip_embed).to(self_module.norm_k_img.weight.dtype))
        k_img = k_img.view(b, -1, n, d).to(x.dtype)
        v_img = self_module.v_img(clip_embed).view(b, -1, n, d)
        out.add_(attention(q, k_img, v_img, attention_mode=self_module.attention_mode, heads=n).flatten(2))
    return self_module.o(out)


def _track_uncond(transformer) -> dict:
    """Kijai's WanModel.forward gets is_uncond; its cross-attention does not. Note it per call so the mask keeps
    Prompt Relay's rule of never penalising the negative prompt (core: transformer_options['cond_or_uncond'])."""
    state = getattr(transformer, _KIJAI_STATE_ATTR, None)
    if state is not None:
        return state
    state = {"uncond": False}
    orig = getattr(transformer, "forward", None)
    if not callable(orig):                     # nothing to wrap: every call counts as the conditional pass
        try:
            setattr(transformer, _KIJAI_STATE_ATTR, state)
        except AttributeError:
            pass
        return state
    had_own = "forward" in getattr(transformer, "__dict__", {})

    def forward(*args, **kwargs):
        state["uncond"] = bool(kwargs.get("is_uncond", False))
        return orig(*args, **kwargs)

    setattr(transformer, _KIJAI_TFWD_ATTR, (orig if had_own else None,))
    setattr(transformer, _KIJAI_STATE_ATTR, state)
    transformer.forward = forward
    return state


def patch_kijai(patcher, mask_fn: Callable) -> int:
    """Patch a Kijai ``WanVideoModel`` in place. Returns the number of blocks patched.

    The Kijai sampler does ``transformer = patcher.model.diffusion_model`` and
    calls it directly, bypassing ``ModelPatcher.object_patches``. We rebind
    ``block.cross_attn.forward`` on the live transformer with idempotent
    guards so repeated queues don't stack patches.
    """
    model = getattr(patcher, "model", patcher)
    transformer = getattr(model, "diffusion_model", None)
    if transformer is None:
        raise RuntimeError(
            "Kijai patch path: model.diffusion_model is missing. "
            f"Got type(model)={type(model).__name__}."
        )
    blocks = getattr(transformer, "blocks", None)
    if blocks is None:
        raise RuntimeError("Kijai patch path: transformer.blocks is missing.")

    state = None
    patched = 0
    skipped: set = set()
    for idx, block in enumerate(blocks):
        cross_attn = getattr(block, "cross_attn", None)
        if cross_attn is None:
            continue
        if _kijai_layout(cross_attn):
            if type(cross_attn).__name__ not in _KIJAI_PORTED:      # L7.128: no port - leave the block untouched
                skipped.add(type(cross_attn).__name__)
                continue
            impl = _kijai_cross_forward
            if state is None:
                state = _track_uncond(transformer)
            setattr(cross_attn, _KIJAI_STATE_ATTR, state)
        else:
            impl = _kijai_class_dispatch(cross_attn)

        # Save original on first patch; on re-patch, overwrite the bound forward
        # but keep the original-saved one intact so restore_kijai still works.
        if not hasattr(cross_attn, _KIJAI_ORIGINAL_ATTR):
            setattr(cross_attn, _KIJAI_ORIGINAL_ATTR, cross_attn.forward)

        def _make_bound(impl_=impl, mfn_=mask_fn, ca_=cross_attn):
            def _bound(self_module, *args, **kwargs):
                return impl_(self_module, mfn_, *args, **kwargs)
            return types.MethodType(_bound, ca_)

        cross_attn.forward = _make_bound()
        setattr(cross_attn, _KIJAI_MARKER_ATTR, True)
        patched += 1

    if skipped:
        log.warning("Prompt Relay has no Kijai port for %s; those blocks run unchanged.", ", ".join(sorted(skipped)))
    if not patched and skipped:
        raise RuntimeError(f"Prompt Relay cannot drive this Kijai model: no block has a ported cross-attention "
                           f"({', '.join(sorted(skipped))}). Nothing was patched.")
    log.info("Kijai patch path: rebound forward on %d cross_attn blocks", patched)
    return patched


def restore_kijai(patcher) -> int:
    """Reverse :func:`patch_kijai`. Returns the number of blocks restored."""
    model = getattr(patcher, "model", patcher)
    transformer = getattr(model, "diffusion_model", None)
    if transformer is None:
        return 0
    blocks = getattr(transformer, "blocks", None) or []
    restored = 0
    for block in blocks:
        cross_attn = getattr(block, "cross_attn", None)
        if cross_attn is None:
            continue
        orig = getattr(cross_attn, _KIJAI_ORIGINAL_ATTR, None)
        if orig is not None:
            if getattr(orig, "__func__", None) is getattr(type(cross_attn), "forward", None):
                cross_attn.__dict__.pop("forward", None)          # back to the class's own forward, no override
            else:
                cross_attn.forward = orig
            try:
                delattr(cross_attn, _KIJAI_ORIGINAL_ATTR)
            except AttributeError:
                pass
            for attr in (_KIJAI_MARKER_ATTR, _KIJAI_STATE_ATTR):
                if hasattr(cross_attn, attr):
                    try:
                        delattr(cross_attn, attr)
                    except AttributeError:
                        pass
            restored += 1
    saved = getattr(transformer, _KIJAI_TFWD_ATTR, None)
    if saved is not None:
        if saved[0] is None:
            transformer.__dict__.pop("forward", None)
        else:
            transformer.forward = saved[0]
        for attr in (_KIJAI_TFWD_ATTR, _KIJAI_STATE_ATTR):
            try:
                delattr(transformer, attr)
            except AttributeError:
                pass
    log.info("Kijai restore path: %d blocks restored", restored)
    return restored


# ────────────────────────────────────────────────────────────────────────
# Backend 3: generic introspective fallback
# ────────────────────────────────────────────────────────────────────────

def patch_generic(model_clone, mask_fn: Callable) -> int:
    """Best-effort patcher for unknown architectures.

    Walks ``diffusion_model.blocks`` (or ``transformer_blocks``) and patches
    any attribute named ``cross_attn``/``attn2``. Uses ``add_object_patch`` so
    it composes with any native sampler. Returns count patched; returns 0 if
    the model exposes neither block list.
    """
    try:
        diffusion_model = model_clone.get_model_object("diffusion_model")
    except Exception:
        return 0
    blocks = (
        getattr(diffusion_model, "blocks", None)
        or getattr(diffusion_model, "transformer_blocks", None)
    )
    if blocks is None:
        return 0

    block_path = "blocks" if hasattr(diffusion_model, "blocks") else "transformer_blocks"
    patched = 0
    for idx, block in enumerate(blocks):
        for attr in ("cross_attn", "attn2"):
            module = getattr(block, attr, None)
            if module is None:
                continue
            key = f"diffusion_model.{block_path}.{idx}.{attr}.forward"
            _check_unpatched(model_clone, key)
            impl = _kijai_class_dispatch(module) if attr == "cross_attn" else _ltx_forward
            model_clone.add_object_patch(
                key, _CrossAttnPatch(impl, mask_fn).__get__(module, module.__class__)
            )
            patched += 1
    if patched:
        log.info("Generic patch path: patched %d cross-attention modules", patched)
    return patched
