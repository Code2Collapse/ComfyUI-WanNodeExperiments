# Wan (Tongyi / Alibaba) Model Family — Research & Improvement Plan

Research compiled 2026-06-24 to ground the WanDirector / ControlAOV / Wan-Animate
pipeline work. Goal: understand how each Wan model works, its limits, and concrete,
implementable fixes for the failure modes observed (color shift, pixel/spatial drift,
"damaged-pixel"/net artifacts, missing long-context, infinite length, EverAnimate).

> Correction baked in: **Wan has no IC-LoRA track** (that's an LTX concept). Wan's
> equivalents are **control video** conditioning (Wan-Fun Control, VACE) and
> **reference image** conditioning. The LTX-2.0 fork's "IC-LoRA track" must be
> repurposed as a **control-video track** for the Wan node.

---

## 1. The family — what each model is

| Model | Task | Conditioning | Notes |
|-------|------|--------------|-------|
| **Wan 2.1** (1.3B / 14B) | T2V, I2V | text (umT5), ref image (I2V) | Baseline DiT + 3D **causal** Wan-VAE. |
| **Wan 2.2** (27B total / 14B active) | T2V, I2V | text, ref image | **First MoE video diffusion** — high-noise expert (layout) + low-noise expert (detail), split by denoising timestep. Higher-compression VAE (4×16×16, ratio 64). Dual-CFG (separate guidance per expert). |
| **Wan-Fun InP** | I2V / first-last-frame | start (+ optional end) image | Inpaint-style temporal conditioning. |
| **Wan-Fun Control** | controlled T2V/V2V | **control video** (depth / pose / canny / etc.) + optional ref | The Wan analogue of ControlNet-for-video. |
| **Wan VACE** | unified control | ref image **+ control video + mask** | One model for editing/inpaint/extend/control. |
| **Wan 2.2 Animate** (14B) | character animation / replacement | **spatially-aligned skeleton** (body) + **implicit face features** (expression) + ref image; **relight LoRA** | Animation mode (animate ref char by driving video, keep bg) and Replacement mode (insert char into driving video's env, relight to match). |
| **Wan S2V / MultiTalk / InfiniteTalk** | audio-driven talking | ref image + audio (+ multi-speaker) | Long-form audio-driven; InfiniteTalk targets long duration. |

**Shared architecture:** umT5 text encoder → DiT (flow-matching) → **3D causal Wan-VAE**
(temporal causality, 4× temporal / 16×16 spatial compression in 2.2). The *causal* VAE is
the single most important property for long video — it permits **sliding-window decoding**
(decode chunks reusing prior temporal context) with bounded memory, which every infinite-
length method below exploits.

Sources: [Wan2.2 GitHub](https://github.com/Wan-Video/Wan2.2), [Wan2.2 explained](https://vast.ai/article/wan-2-2-explained-new-approach-ai-video-generation), [AIBase MoE](https://news.aibase.com/news/20029), [Wan-Animate paper 2509.14055](https://arxiv.org/html/2509.14055v1), [Wan2.2-Animate HF](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B), [ComfyUI Wan2.2 docs](https://docs.comfy.org/tutorials/video/wan/wan2_2).

---

## 2. Failure modes → root cause → fix

### 2.1 Color shift / color drift
**Where:** chaining clips (last-frame → next clip), I2V looping/continuation, long extension.
**Root cause:** **autoregressive error accumulation** — small per-frame errors compound, and
re-encoding a decoded last frame through the VAE injects a slight color/contrast bias each
hop (contrast/saturation creep over loops is the classic symptom). This is a known,
acknowledged Wan-2.2 community issue.
**Fixes (cheap → strong):**
1. **Color-match each new chunk to an anchor frame** (reference = first/clean frame), not just to the previous chunk — matching only to the previous chunk lets bias accumulate. (Histogram/mean-std match; wavelet color-fix.)
2. **Re-inject anchor features** (StreamingT2V "Appearance Preservation Module" idea): condition each chunk on a fixed anchor latent, not only the rolling last frame.
3. **Stable Video Infinity (SVI)** "error recycling" — explicitly trains/decodes to absorb accumulated error; SVI 2.0 Pro supports Wan 2.2 (HIGH/LOW LoRAs).
4. Feed the clean **latent** (not the decoded+re-encoded pixel frame) forward when possible — avoids the per-hop VAE round-trip bias.
Sources: [Wan2.2 #172 color difference](https://github.com/Wan-Video/Wan2.2/issues/172), [WanVideoWrapper #1541 loop contrast](https://github.com/kijai/ComfyUI-WanVideoWrapper/issues/1541), [Stable Video Infinity 2510.09212](https://arxiv.org/html/2510.09212v1), [SVI 2.0 Pro Wan2.2](https://comfyui-wiki.com/en/news/2025-12-27-svi-2-0-pro-wan-2-2-release), [Painter-I2V-AIO](https://github.com/LDNKS094/ComfyUI-Painter-I2V-AIO).

### 2.2 Pixel offset / spatial drift
**Root cause:** same autoregressive accumulation but spatial — the scene slowly translates/
warps across chunks; also caused by misaligned latent boundaries when stitching windows.
**Fixes:** anchor-latent conditioning (as 2.1.2); **overlapping context windows with blending**
(not hard cuts) so boundaries are averaged; keep a static reference latent across windows
("reference latents … maintaining a particular aesthetic throughout").
Sources: [WanVideoWrapper context windows](https://deepwiki.com/kijai/ComfyUI-WanVideoWrapper/6.5-context-windows-for-long-videos), [Relax Forcing 2603.21366](https://arxiv.org/pdf/2603.21366).

### 2.3 "Damaged pixels" / net / grid artifacts on output
**Root cause:** **VAE tiled-decode seams.** ComfyUI's tiled decode defaults to tile 512 /
overlap 64; the seam is where tiles cut and overlap-blend, producing a faint grid/net,
especially on flat regions. Low-bit (fp8/fp4) VAE decode worsens it.
**Fixes:**
1. **Increase tile overlap to ≥20%** of tile size (tile 512 → overlap ≥100); or **reduce
   tile size** (smaller tiles + moderate overlap blend better than huge tiles + max overlap).
2. **fp32 VAE decode** for the final pass (we already expose `vae_fp32_decode` in WanDirector — make it the default for final decode).
3. Decode the **whole latent untiled** when VRAM allows; only tile when forced.
Sources: [VAEDecodeTiled docs](https://comfyai.run/documentation/VAEDecodeTiled), [LTX-2 #202 grid tiles](https://github.com/Lightricks/LTX-2/issues/202), [LTX-2 #19 white grid](https://huggingface.co/Lightricks/LTX-2/discussions/19), [SeedVR2 artifact guide](https://apatero.com/blog/seedvr2-removing-artifacts-complete-guide-2025).

### 2.4 "Net issues on output"
If this means **network/inference artifacts** (blocky/over-sharpened/oversaturated): usually
**too-few steps + too-high CFG** with distillation LoRAs (CausVid / lightx2v) — they need
**low CFG (≈1)** and few steps; running them at CFG 5–7 produces the burnt/damaged look.
For 2.2 MoE, the high/low-noise experts want **different CFG** (our dual-CFG) — a single high
CFG across both burns detail. Recommendation: when a speed LoRA is active, force CFG→1 and
expose the dual-CFG split.

### 2.5 Context windows — "not built-in"
**True:** Wan has **no native long-context**; it's trained at a fixed frame count (e.g. 81).
Beyond that you must window. The community-standard is Kijai's **WanVideoContextOptions**:
splits the video into **overlapping context windows**, schedules them (Uniform / static),
blends overlaps, and can carry **reference latents** for aesthetic continuity. This is what
we should wire into the WanDirector long-form path (rather than inventing our own).
Sources: [WanVideoContextOptions](https://www.instasd.com/comfyui/custom-nodes/comfyui-wanvideowrapper/wanvideocontextoptions), [Context windows deepwiki](https://deepwiki.com/kijai/ComfyUI-WanVideoWrapper/6.5-context-windows-for-long-videos).

### 2.6 Infinite-length support
**Causal VAE is the enabler** (sliding-window decode, bounded memory). Methods, by approach:
- **Sliding-window / diffusion forcing** (SkyReels-V2): last few frames as historical context to predict the next segment.
- **CausVid:** KV-cache + diffusion for autoregressive inference (fast, but drifts).
- **Self-Forcing / Self-Forcing++ / LongLive:** train the model to predict from *its own* prior (error-containing) frames → much less drift; LongLive = real-time interactive long video, built on Wan-2.1's causal VAE.
- **Stable Video Infinity (SVI):** "error recycling" for truly infinite length with little degradation; **SVI 2.0 Pro supports Wan 2.2** as drop-in HIGH/LOW LoRAs in ComfyUI — the most practical path for us to expose.
- **RIFLEx:** "free-lunch" length **extrapolation** for video DiTs (rescales positional frequency so the model generates longer than trained without retraining) — we already expose `enable_riflex`; document that it's extrapolation, not true infinite.
- **FramePack:** compresses history into a fixed budget for long generation.
Sources: [SVI 2510.09212](https://arxiv.org/pdf/2510.09212), [LongLive 2509.22622](https://arxiv.org/html/2509.22622v1), [PackForcing 2603.25730](https://arxiv.org/pdf/2603.25730), [SVI 2.0 Pro](https://comfyui-wiki.com/en/news/2025-12-27-svi-2-0-pro-wan-2-2-release), [WAN2.2 SVI ComfyUI](https://www.patreon.com/posts/wan-2-2-svi-one-147707880).

### 2.7 EverAnimate for Wan-Animate (long-form character animation)
**What it is:** a **post-training method for minute-scale** human animation that fixes the
drift Wan-Animate shows over long durations. Wan-Animate is already the best long-range
animation baseline because of its **attention-sink** design (most tokens attend to the ref
frame) — but it **still degrades** over time (static backgrounds rot, identity drifts).
**EverAnimate's fix: Persistent Latent Propagation** — a **persistent latent context memory**
maintained *across chunks* that propagates identity + motion in latent space ("Latent Flow
Restoration"), anchoring long-horizon generation and mitigating temporal forgetting.
**Implication for us:** Wan-Animate + EverAnimate = chunked animation with a carried latent
memory (not just last-frame). Our `everanimate_*` widgets (stage / num_chunks / overlap /
anchor strategy) map directly onto this — wire them to a real chunked-with-latent-memory
runner.
Sources: [EverAnimate 2605.15042](https://arxiv.org/html/2605.15042v1), [Wan-Animate 2509.14055](https://arxiv.org/html/2509.14055v1), [SCAIL 2512.05905](https://arxiv.org/html/2512.05905v1).

---

## 3. Concrete improvement plan for our nodes (priority order)

**A. WanDirector — make the LTX-2.0 UI correct for Wan**
1. **Rename the "IC-LoRA" track → "Control Video" track** (Wan-Fun Control / VACE). Drag a
   video onto it → it becomes the control conditioning for that segment.
2. Keep the timeline → `timeline_data` contract; the control-video track writes control clips.
3. Expose **WanVideoContextOptions-style** long-form settings (window size, overlap, schedule,
   reference latent) for >81-frame timelines, instead of a home-grown approach.
4. Default **fp32 VAE decode** on the final pass; expose tile overlap ≥20% when tiling.
5. Speed-LoRA aware: when CausVid/lightx2v LoRA is on, force **CFG≈1** + few steps; surface dual-CFG for 2.2.

**B. Long / infinite length**
- Integrate **SVI 2.0 Pro (Wan 2.2)** HIGH/LOW LoRAs as a selectable "infinite length" mode.
- Anchor-frame **color-match + latent re-injection** per chunk (kills 2.1 color drift + 2.2 spatial drift).
- Keep **RIFLEx** as the no-extra-model length-extrapolation toggle (document its limits).

**C. Wan-Animate + EverAnimate**
- Chunked animation runner with **persistent latent memory** across chunks (EverAnimate),
  anchored to the ref frame (attention-sink) — wire the existing `everanimate_*` widgets.
- Use the 3D-pose (NLF) → ViTPose-format skeleton we built for ~exact body control.

**D. Artifact hygiene (applies to all)**
- Final decode: fp32 + untiled when VRAM allows; else tile overlap ≥100 (tile 512).
- Color: match every chunk to a fixed anchor, never only to the previous chunk.

---

## 4. Open verification items
- Confirm SVI 2.0 Pro Wan-2.2 LoRA filenames + load path for a selectable mode.
- Confirm Wan-Fun Control vs VACE control-video tensor format (what the model expects) before
  wiring the control-video track end-to-end.
- GPU-verify EverAnimate latent-memory runner (needs the Animate-14B weights; heavy).
