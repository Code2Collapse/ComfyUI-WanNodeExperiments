# MODIFICATIONS & ATTRIBUTION

This document records, per the Apache-2.0 terms and the upstream `NOTICE`
requirements of ComfyUI-WanAnimatePlus, what code in this pack is derived from
external projects and how it was modified.

## License of this pack
Apache License 2.0 (see `LICENSE`, `NOTICE`).

## Derived-from / affected modules

### From Kijai — ComfyUI-WanVideoWrapper (Apache-2.0)
- **`wanwrapper/`** — a FULL COPY of Kijai's node suite, committed in-repo as
  normal tracked files (no submodule/gitlink, no external dependency) so the
  pack is self-contained on any clone. Registered under a `WNE_` prefix with
  skip-if-upstream-installed logic. Upstream LICENSE retained in the folder.
  All credit for this code: **Kijai**.
- **`nodes/loaders.py`** *(Phase 2)* — Wan 2.2 T2V/I2V/VACE/MoE model loading,
  GGUF dequant, fp8/low-VRAM paths. Reimplemented standalone; refined for GPU
  utilization (`torch.inference_mode()`, explicit `gc`/`empty_cache`, einops).
- **`nodes/samplers.py`** *(Phase 2)* — sampler/scheduler logic.
- **`nodes/vae_nodes.py`** *(Phase 4)* — 3D VAE tiling / context-window decode.
- Modification summary for `nodes/`: extracted architectural patterns, rewrote
  as standalone implementations using native ComfyUI types. (An earlier
  revision of this file claimed "no upstream files imported" — that was true
  of `nodes/` but omitted the full `wanwrapper/` copy; corrected 2026-07-18.)

### From wuwukaka — ComfyUI-WanAnimatePlus (Apache-2.0, itself a fork of the above)
- The FULL Animate-Plus codebase is ported INTO this pack as first-class
  tracked files (2026-07-18). History of how it got here: originally a git
  SUBMODULE POINTER (empty on GitHub/fresh clones), then briefly a
  `wananimateplus/` folder copy, now DISSOLVED into the pack root — no
  directory named after the upstream project remains:
  * its subpackages live at the pack root: `fantasyportrait/`, `multitalk/`,
    `unianimate/`, `wanvideo/`, `Ovi/`, `diffsynth/`, `HuMo/`, `SCAIL/`,
    `mocha/`, `lynx/`, and the rest (37 folders, incl. the fantasyportrait
    ONNX face models);
  * its glue modules live at the pack root: `animateplus.py` (node
    aggregation, formerly its `__init__.py`), `nodes_animate.py` (formerly
    its `nodes.py` — renamed because this pack already has a `nodes/`
    package), `nodes_model_loading.py`, `nodes_sampler.py`,
    `nodes_utility.py`, `utils.py`, `custom_linear.py`, etc.;
  * upstream license/attribution retained at root as
    `LICENSE_WANANIMATEPLUS` + `NOTICE_WANANIMATEPLUS` (and the bundled
    components' licenses — FantasyPortrait, Ovi/BigVGAN, diffsynth — remain
    in their subfolders); upstream READMEs kept under `docs/`.
  All credit for this code: **wuwukaka** (fork) and **Kijai** (base).
  Registered under a `WNE_AP_` prefix, skipped when the genuine upstream
  pack is installed.
- Wan Animate extensions and related loader/sampler refinements folded into
  `nodes/loaders.py` / `nodes/samplers.py` *(Phase 2)*.
- This NOTICE/attribution is retained per the upstream NOTICE requirement.
- `nodes/audio_lipsync.py` imports the fantasyportrait ONNX face detector from
  the ported `fantasyportrait/` package and the MultiTalk/InfiniteTalk
  pipeline from `wanwrapper/` (delegation to real upstream code, credited
  above — not reimplementations).

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
