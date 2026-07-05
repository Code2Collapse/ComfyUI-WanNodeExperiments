# MODIFICATIONS & ATTRIBUTION

This document records, per the Apache-2.0 terms and the upstream `NOTICE`
requirements of ComfyUI-WanAnimatePlus, what code in this pack is derived from
external projects and how it was modified.

## License of this pack
Apache License 2.0 (see `LICENSE`, `NOTICE`).

## Derived-from / affected modules

### From Kijai — ComfyUI-WanVideoWrapper (Apache-2.0)
- **`nodes/loaders.py`** *(Phase 2)* — Wan 2.2 T2V/I2V/VACE/MoE model loading,
  GGUF dequant, fp8/low-VRAM paths. Reimplemented standalone; refined for GPU
  utilization (`torch.inference_mode()`, explicit `gc`/`empty_cache`, einops).
- **`nodes/samplers.py`** *(Phase 2)* — sampler/scheduler logic.
- **`nodes/vae_nodes.py`** *(Phase 4)* — 3D VAE tiling / context-window decode.
- Modification summary: extracted architectural patterns, rewrote as standalone
  implementations using native ComfyUI types; no upstream files imported.

### From wuwukaka — ComfyUI-WanAnimatePlus (Apache-2.0, itself a fork of the above)
- Wan Animate extensions and related loader/sampler refinements folded into
  `nodes/loaders.py` / `nodes/samplers.py` *(Phase 2)*.
- This NOTICE/attribution is retained per the upstream NOTICE requirement.

### From WhatDreamsCost — LTX Director 2.0 (upstream GPL-3.0; used WITH PERMISSION)
- **`nodes/director.py` + `web/js/wan_director*.js`** *(Phase 5)* — WanDirector
  timeline editor brought to LTX Director 2.0 feature parity (timeline editing,
  keyframes, custom audio + inpainting, video import/split/extend, IC-LoRA track,
  timeline save/load, UI overhaul, QoL suite).
- **Permission record:** the LTX Director author (WhatDreamsCost) granted
  Halohues Studios explicit permission to copy/adapt LTX Director code and
  redistribute it under this pack's Apache-2.0 license. Absent that grant the
  code would be GPL-3.0; the grant is what permits Apache redistribution here.
- WanDirector base (`WanDirectorC2C`) originates in the author's own
  ComfyUI-CustomNodePacks (C2C/MEC) and is relocated here.

### Research methods
Implemented from public papers/reference code; credited per-node in source
docstrings (see `NOTICE` for the list).

## WanDirector — Wan target & variant coverage
WanDirector is the **Wan** counterpart of LTX Director 2.0: same timeline-editor
feature set, but driving Wan models (not LTX). It supports the full Wan family via
a flag-driven variant table: wan2.1 t2v/i2v, wan2.2 t2v/i2v (dual-cfg), Fun
(inpaint/control), Animate (+ EverAnimate long-horizon), Move, VACE, InfiniteTalk,
MultiTalk, S2V, Phantom. LTX 2.0 feature parity (timeline editing QoL, save/load,
IC-LoRA track, audio inpainting, UI overhaul) is being ported to the Wan timeline
in browser-verified slices.

## Phase log
- **Phase 1 (2026-06-21):** package layout (`nodes/`), master `__init__.py`,
  frontend `web/js/main.js`, Apache `LICENSE`/`NOTICE`/this file. The 18 custom
  R&D nodes (built earlier) re-homed under `nodes/` (`guidance.py`, `samplers.py`,
  `vae_nodes.py`).
- **Phase 2:** `nodes/loaders.py` (WanModelLoader/MoE/VACE/GGUF) + forked
  scheduler nodes in `samplers.py`. **Phase 3:** refined CFG (RescaleCFG,
  CFG-Zero★, TCFG) in `guidance.py`. **Phase 4:** forked tiled decode + encode in
  `vae_nodes.py`. (29 nodes, verified.)
- **Phase 5a (2026-06-21):** shifted the WanDirector subsystem in
  (`nodes/wan_director/` + `prompt_relay/` + `asymflow_sampler.py` +
  `_is_changed_util.py`, JS `wan_director_*.js` + `_c2c_theme/_c2c_report`).
  Verified in a live ComfyUI: 30 nodes, 0 import failures.
- **Phase 5b (in progress):** extended Wan variant coverage to 14
  (added move, vace, infinitetalk, multitalk, s2v, phantom). LTX-2.0 timeline
  feature port underway in verified slices.
