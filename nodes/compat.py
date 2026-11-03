"""
compat.py — dual-mode compatibility layer for ComfyUI-WanNodeExperiments.

Credits: the model-patching strategy here was informed by studying the
foundational Wan 2.2 wrappers by **Kijai** (ComfyUI-WanVideoWrapper) and
**wuwukaka** (ComfyUI-WanAnimatePlus). No code is imported from them.

This module gives every model-patching node ONE place to:
  * detect whether it was handed a native ComfyUI ``MODEL`` (a
    ``comfy.model_patcher.ModelPatcher``) or Kijai's ``WANVIDEOMODEL``
    (a ``WanVideoModelPatcher``);
  * clone the model and attach guidance / unet / attention hooks against
    whichever ecosystem it is;
  * intercept attention (``optimized_attention``) for cross-cutting effects
    (logit decay, token amplification, layer skipping, sparse attention)
    on the native path, with a clear console note + graceful fallback on the
    Kijai path where his sampler owns the forward pass.

Nothing here raises if ComfyUI is not importable (e.g. during a standalone
import check) — everything that touches ``comfy`` is guarded.
"""

from __future__ import annotations

import contextlib
import threading
import logging

import torch

log = logging.getLogger("WanNodeExperiments")
if not log.handlers:
    logging.basicConfig(level=logging.INFO)

# --------------------------------------------------------------------------- #
# Optional ComfyUI imports (guarded so the pack imports without a live ComfyUI)
# --------------------------------------------------------------------------- #
HAS_COMFY = False
_attention_mod = None
try:  # pragma: no cover - depends on runtime
    import comfy.model_management as model_management  # noqa: F401
    import comfy.ldm.modules.attention as _attention_mod  # type: ignore
    HAS_COMFY = True
except Exception:  # noqa: BLE001
    model_management = None  # type: ignore


# --------------------------------------------------------------------------- #
# Human-readable errors (per project CLAUDE.md humanise() contract)
# --------------------------------------------------------------------------- #
_HUMANISE_RULES = [
    (r"NoneType.*attribute", "A required input is not connected."),
    (r"CUDA out of memory", "Not enough GPU memory. Lower resolution, frame count or tile size."),
    (r"FileNotFoundError", "A model/weight file is missing. Check the file path."),
    (r"RuntimeError.*size", "Tensor dimensions do not match between inputs."),
    (r"ModuleNotFoundError", "A required Python package is missing (see requirements.txt)."),
]


def humanise(raw) -> str:
    """Turn a raw exception/traceback into one plain-English line."""
    import re

    if raw is None:
        return "Something went wrong."
    text = str(raw)
    for pattern, msg in _HUMANISE_RULES:
        if re.search(pattern, text, re.IGNORECASE):
            return msg
    first = next((ln for ln in text.split("\n") if ln.strip() and not ln.startswith(" ")), "")
    if not (first or text.strip()):
        # L7.87 (2026-10-12): an exception without a message (a bare assert, KeyError(), ...) became just "Error: "
        # and the loaders failed with nothing to go on - name the exception and the line it came from.
        if not isinstance(raw, BaseException):
            return "Something went wrong."
        tb = raw.__traceback__
        while tb is not None and tb.tb_next is not None:
            tb = tb.tb_next
        import os

        where = f" in {os.path.basename(tb.tb_frame.f_code.co_filename)}:{tb.tb_lineno}" if tb is not None else ""
        return f"Error: {type(raw).__name__}{where} (no message)"
    return "Error: " + (first or text[:120])


# --------------------------------------------------------------------------- #
# Model detection / cloning
# --------------------------------------------------------------------------- #
def detect_model(model) -> str:
    """Return ``"native"``, ``"wanvideo"`` or ``"unknown"``.

    Kijai's ``WanVideoModelPatcher`` subclasses ComfyUI's ``ModelPatcher``, so
    we check the class name *first* before falling back to capability checks.
    """
    if model is None:
        return "unknown"
    cls = type(model).__name__
    mod = type(model).__module__ or ""
    if "WanVideo" in cls or "wanvideo" in mod.lower():
        return "wanvideo"
    if hasattr(model, "set_model_sampler_post_cfg_function") and hasattr(model, "model_options"):
        return "native"
    if hasattr(model, "model_options"):
        return "wanvideo"  # has the dict but not native hooks → Kijai-style
    return "unknown"


def clone_model(model):
    """Clone a model patcher without mutating the caller's object."""
    if hasattr(model, "clone"):
        return model.clone()
    import copy

    return copy.copy(model)


def transformer_options(m) -> dict:
    """Return (creating if needed) ``m.model_options['transformer_options']``."""
    opts = getattr(m, "model_options", None)
    if opts is None:
        m.model_options = {}
        opts = m.model_options
    if "transformer_options" not in opts:
        opts["transformer_options"] = {}
    return opts["transformer_options"]


def set_transformer_option(m, key, value) -> None:
    transformer_options(m)[key] = value


def set_post_cfg(m, fn) -> bool:
    """Attach a sampler ``post_cfg`` function (native API). Returns success.

    Kijai's ``WANVIDEOMODEL`` inherits ``set_model_sampler_post_cfg_function``
    from ``ModelPatcher`` but his sampler never calls it, so a plain ``hasattr``
    check reports a false success and the caller's Kijai fallback (which maps the
    effect onto ``transformer_options``) never runs. Gate on ``detect_model`` so
    only genuine native models take the hook; Kijai models return False.
    """
    if detect_model(m) == "wanvideo":
        return False
    if hasattr(m, "set_model_sampler_post_cfg_function"):
        m.set_model_sampler_post_cfg_function(fn)
        return True
    return False


def set_unet_wrapper(m, fn) -> bool:
    """Attach a unet function wrapper (native API). Returns success.

    Same false-success caveat as ``set_post_cfg`` on Kijai's WANVIDEOMODEL — gate
    on ``detect_model`` so the Kijai fallback stays reachable.
    """
    if detect_model(m) == "wanvideo":
        return False
    if hasattr(m, "set_model_unet_function_wrapper"):
        m.set_model_unet_function_wrapper(fn)
        return True
    return False


# --------------------------------------------------------------------------- #
# Attention interception
# --------------------------------------------------------------------------- #
# A single, well-contained monkeypatch of ComfyUI's global
# ``optimized_attention`` dispatch, gated by a thread-local stack of
# "modifiers". When no modifier is active the original function runs verbatim,
# so there is zero overhead and zero behavioural change outside our context.
#
# A modifier is an object with:
#   * ``bias(layer_idx, q, k) -> Optional[Tensor]``  additive logit bias mask
#       broadcastable to (..., q_len, k_len); used for UltraViCo logit decay and
#       T5 token amplification (scaling == additive log-bias on key columns).
#   * ``skip(layer_idx) -> bool``  if True, attention returns a value-passthrough
#       (the STG "skipped spatiotemporal layer" weak model).
#
# Layers are indexed by a per-forward counter so nodes can target specific
# block indices deterministically.
# --------------------------------------------------------------------------- #
_tls = threading.local()
_orig_optimized_attention = None
_patched = False


def _stack():
    s = getattr(_tls, "stack", None)
    if s is None:
        s = []
        _tls.stack = s
    return s


def _reset_layer_counter():
    _tls.layer = 0
    _tls.block = None
    _tls.block_call = 0


def _block_call(block) -> int:
    """0 for the first attention call inside a transformer block, 1 for the next, ... (reset when the block number
    changes). Core's Wan blocks run self-attention first, then cross-attention."""
    if getattr(_tls, "block", None) != block:
        _tls.block = block
        _tls.block_call = 0
    else:
        _tls.block_call = getattr(_tls, "block_call", 0) + 1
    return _tls.block_call


def _next_layer():
    n = getattr(_tls, "layer", 0)
    _tls.layer = n + 1
    return n


def _is_container(t) -> bool:
    """ComfyUI 0.39+ passes q / k / v as AttentionTensorContainer (single-owner; peek() reads, take() consumes).
    Duck-typed so this works whether or not comfy imported here."""
    return hasattr(t, "peek") and hasattr(t, "take") and not torch.is_tensor(t)


def _peek(t):
    return t.peek() if _is_container(t) else t


def _wrapped_optimized_attention(q, k, v, heads, *args, **kwargs):
    stack = _stack()
    if not stack:
        return _orig_optimized_attention(q, k, v, heads, *args, **kwargs)

    layer = _next_layer()
    mod = stack[-1]
    qt, kt = _peek(q), _peek(k)          # tensors for our own logic; the originals still go to attention

    # STG-style skip → a value passthrough (aligned weak model). Only SELF-attention is ever skipped (cross-attention's
    # v holds text tokens, so returning it broke the residual add), and the index is the transformer BLOCK when core
    # names it (transformer_options["block_index"], set by core's Wan / LTX / Flux loops) - the same numbers the Kijai
    # fallback passes as slg_layers. The per-call counter is only the fallback for models that do not name blocks.
    try:
        topts = kwargs.get("transformer_options") or {}
        block = topts.get("block_index") if isinstance(topts, dict) else None
        is_self = qt.dim() >= 2 and kt.dim() >= 2 and qt.shape[-2] == kt.shape[-2]
        if block is not None:
            # equal lengths alone are not enough: a latent with exactly as many tokens as the text (512) made
            # cross-attention look like self-attention. Only a block's FIRST attention call is its self-attention.
            is_self = is_self and _block_call(block) == 0
        if is_self and mod.skip(int(block) if block is not None else layer):
            # attention's output layout is (B, L, heads * dim_head) - exactly v's (B, L, heads * dim_head)
            return v.take() if _is_container(v) else v
    except Exception:  # noqa: BLE001
        pass

    # Additive logit bias (UltraViCo decay / T5 amplification). Modifiers read tensors, never containers (on core
    # 0.39+ q.dim() raised on the container, the error was swallowed and the modifier silently did nothing).
    bias = None
    try:
        bias = mod.bias(layer, qt, kt)
    except Exception:  # noqa: BLE001
        bias = None

    if bias is None:
        return _orig_optimized_attention(q, k, v, heads, *args, **kwargs)

    # Merge our bias into an existing additive mask if the caller passed one.
    mask = kwargs.pop("mask", None)
    if mask is None and len(args) >= 1:
        mask = args[0]
        args = args[1:]
    if mask is not None and mask.dtype == torch.bool:
        mask = torch.zeros_like(mask, dtype=qt.dtype).masked_fill(~mask, float("-inf"))
    merged = bias if mask is None else (mask + bias)
    try:
        return _orig_optimized_attention(q, k, v, heads, merged, *args, **kwargs)
    except TypeError:
        return _orig_optimized_attention(q, k, v, heads, mask=merged, *args, **kwargs)


def _install_patch() -> bool:
    global _orig_optimized_attention, _patched
    if _patched:
        return True
    if not HAS_COMFY or _attention_mod is None:
        return False
    if not hasattr(_attention_mod, "optimized_attention"):
        return False
    _orig_optimized_attention = _attention_mod.optimized_attention
    _attention_mod.optimized_attention = _wrapped_optimized_attention
    # DiT modules bind the symbol at import time (`from ...attention import
    # optimized_attention`), so patching only the source module is dead for
    # them — e.g. comfy.ldm.wan.model holds its own reference. Rebind every
    # already-imported module whose `optimized_attention` is still the original
    # so the interceptor actually reaches Wan's attention. (Modules imported
    # AFTER this runs pick up the wrapper automatically.)
    import sys
    rebound = 0
    for _mod in list(sys.modules.values()):
        if _mod is None or _mod is _attention_mod:
            continue
        try:
            if getattr(_mod, "optimized_attention", None) is _orig_optimized_attention:
                setattr(_mod, "optimized_attention", _wrapped_optimized_attention)
                rebound += 1
        except Exception:  # noqa: BLE001
            continue
    _patched = True
    log.info("[WanNodeExperiments] attention interception installed (+%d consumer modules rebound)", rebound)
    return True


@contextlib.contextmanager
def attention_modifier(mod):
    """Activate ``mod`` for the duration of a forward pass (native path).

    Yields ``True`` if interception is active, ``False`` if ComfyUI's attention
    symbol could not be patched (caller should warn + degrade gracefully).
    """
    ok = _install_patch()
    if not ok:
        yield False
        return
    _stack().append(mod)
    _reset_layer_counter()
    try:
        yield True
    finally:
        _stack().pop()
        _reset_layer_counter()


__all__ = [
    "HAS_COMFY",
    "log",
    "humanise",
    "detect_model",
    "clone_model",
    "transformer_options",
    "set_transformer_option",
    "set_post_cfg",
    "set_unet_wrapper",
    "attention_modifier",
]
