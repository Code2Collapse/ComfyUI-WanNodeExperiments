# ComfyUI-WanNodeExperiments — Design / Spec

> **R&D node pack for Wan 2.2 (MoE DiT) — guidance, sampling, and VAE-decode experiments.**
>
> **Credits.** This pack would not exist without the foundational Wan 2.2 wrapper work of
> **Kijai** (`ComfyUI-WanVideoWrapper`) and **wuwukaka** (`ComfyUI-WanAnimatePlus`).
> Their code in `third_party/` was studied to understand how Wan 2.2's MoE expert
> switching, VAE tiling, and context windows are implemented. **No code is imported
> from `third_party/`** — every node here is a standalone reimplementation against
> ComfyUI's native types so it chains with native nodes *and* (dual-mode) Kijai's nodes.

Date: 2026-06-20

---

## 1. Goals & non-goals

**Goal.** Ship a single registerable pack (`ComfyUI-WanNodeExperiments`) implementing the
18-node R&D catalog (Parts A/B/C/D). Every node registers, every `execute()` returns
valid tensors, no `NotImplementedError`/`pass` stubs.

**Compatibility.** Nodes accept/return standard ComfyUI types (`MODEL`, `LATENT`, `VAE`,
`IMAGE`, `CLIP`, `CONDITIONING`). Model-patching nodes are **dual-mode**: they work on
ComfyUI-native `MODEL` *and* on Kijai's `WANVIDEOMODEL`.

**Non-goal.** Shipping trained weights. Where a method needs distilled/trained weights
(AnyFlow flow-map, RefDecoder adapter, FreeLOC probe calibration), the node provides a
real functional fallback + a clearly-marked `TODO` for weight integration. It never
crashes or no-ops silently.

---

## 2. Dual-mode patching (core)

`compat.py` exposes:

- `detect_model(model) -> "native" | "wanvideo" | "unknown"`
  - native: `comfy.model_patcher.ModelPatcher` with `set_model_sampler_cfg_function`
  - wanvideo: object whose `model_options["transformer_options"]` exists and class name
    contains `WanVideo` / has `.model.diffusion_model` shaped like Kijai's patcher.
- Hook helpers that branch on detection:

| Capability | Native | Kijai `WANVIDEOMODEL` |
|---|---|---|
| Replace CFG combine | `set_model_sampler_cfg_function` | map to `transformer_options` experimental key; else module-`forward` wrapper |
| Post-CFG stack | `set_model_sampler_post_cfg_function` | module-`forward` wrapper |
| Per-call unet wrap | `set_model_unet_function_wrapper` | module-`forward` wrapper |
| Cross-attn patch | `set_model_attn2_replace` | `transformer_options["wne_*"]` read by forward wrapper |
| Free-form options | `model_options["transformer_options"][...]` | same dict (Kijai already uses it) |

**Honesty rule.** When the native hook is unavailable on the Kijai path and no equivalent
key exists, we install a **module-level `forward` wrapper** on the underlying diffusion
model (with a save/restore registry keyed by `id(module)`), which both samplers call. If
even that is impossible, the node logs a clear console warning naming the limitation and
returns the model unchanged — never a silent no-op.

**Kijai equivalents we map onto (found in `third_party`):** `decay_factor`/`block_size`
(≈ UltraViCo), `slg_args` (≈ STG skip-layer), `cfg_zero_star`/`use_tcfg`/`fresca`/`tsr`
(experimental_args), `teacache_args`/`cache_args` (≈ MoE cache).

**LATENT/IMAGE/VAE nodes need no dual-mode** — they are tensor-only and ecosystem-agnostic.

---

## 3. Module layout

```
__init__.py        credits, GUARDED submodule imports (record_failure per module), NODE_CLASS_MAPPINGS
compat.py          detect_model + hook helpers + humanise() + module-forward registry
freq_utils.py      vectorized FreeInit LPF (gaussian/butterworth/ideal/box), freq_mix_3d, 3D Laplacian pyramid, 3D gaussian blur
color_utils.py     reinhard/mkl/hm-mvgd color match, DWT (PyWavelets), RGB<->YCbCr
guidance_nodes.py  A2 APG, A3 ZeResFDG, A7 STG, C1 MoECrossFade, C2 T5AttnAmplifier, C3 HFLI
sampling_nodes.py  A1 AnyFlow, A4 WanMoECache, A5 UltraViCo, A6 FreeLOC, A8 FreeInit, C4 MotionMax
vae_nodes.py       B1 RefDecoder, B2 WanLockedDecode, B3 WaveletColorLock, D1 TemporalCausalVAEDecode, D2 YUVColorLock, D3 Latent3DAntiAlias
requirements.txt   numpy, PyWavelets, opencv-python, torchvision (hard reqs; imported at module tops)
setup_thirdparty.sh  init dirs + clone the two study repos
.gitignore         third_party/, __pycache__/
```

`__init__.py` wraps each submodule import in `try/except` + `record_failure(name, exc)`
so a missing hard-dep disables only that module's nodes (with a console message) instead
of breaking all of ComfyUI. Deps are still "required" (listed + imported at module top).

---

## 4. Node fidelity (18 nodes)

**Tier 1 — fully real now (no external weights):**
A2 APG · A3 ZeResFDG · A7 STG · A8 FreeInit · A5 UltraViCo · A4 WanMoECache · C1 MoECrossFade ·
C2 T5AttnAmplifier · C3 HFLI · B2 WanLockedDecode · B3 WaveletColorLock · D1 TemporalCausalVAEDecode ·
D2 YUVColorLock · D3 Latent3DAntiAlias

**Tier 2 — full node + real fallback, weight/model integration is a marked TODO:**
A1 AnyFlow (loads/validates flow-map ckpt path, live NFE schedule) · A6 FreeLOC (real RoPE
re-index VRPR + tiered-sparse-attn scaffold) · B1 RefDecoder (reference feature + detail/color
transfer fallback) · C4 MotionMax (torchvision RAFT optical flow → noise warping)

Research grounding: each node's algorithm is cross-checked against the paper + public
reference code (arXiv / GitHub / HuggingFace) before implementation. Methods with
unverifiable/future arXiv IDs are reconstructed faithfully from the described mechanism
and labelled as such in the class docstring.

---

## 5. Contracts & conventions

- `RETURN_TYPES` always a tuple; `execute()` returns a tuple.
- `IS_CHANGED` returns a real content hash (never `float("nan")`).
- IMAGE `[B,H,W,3]` 0–1, MASK `[B,H,W]`, LATENT `dict{"samples":[B,C,T?,H,W]}` (Wan latents are 5D `[B,16,T,H//8,W//8]`).
- `CATEGORY = "WanNodeExperiments/<Part>"`.
- All user-facing errors via `humanise()`.

## 6. Checks (loop protocol, run every iteration)

1. `py_compile` every module.
2. import `__init__`, assert `NODE_CLASS_MAPPINGS` non-empty and no swallowed failures.
3. structural lint: each class has `INPUT_TYPES`/`RETURN_TYPES`/`FUNCTION`/`CATEGORY`; FUNCTION method exists.

Report passing output as proof. Stop after 5 attempts or a repeated identical error.
