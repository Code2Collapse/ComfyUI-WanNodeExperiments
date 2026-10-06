# Wan Node Experiments (WNE) — Node Reference

*Auto-generated from the live `NODE_CLASS_MAPPINGS` on 2026-07-29 — 40 nodes. Every parameter description below is the node's own tooltip, so this file cannot drift from the code.*

Regenerate after changing any node's `INPUT_TYPES`.


## Contents

- **C2C/Wan_Director** (2)
  - [Wan Director](#wandirectorc2c)
  - [Wan Director — Inspector](#wandirectorinspector)
- **WanNodeExperiments/Audio** (1)
  - [Advanced Audio Separator (Vocals)](#wne-advancedaudioseparator)
- **WanNodeExperiments/Director** (1)
  - [Wan Director — Extra Args](#wandirectorextraargs)
- **WanNodeExperiments/Flow** (2)
  - [Flow Temporal Consistency — FlowVid/FRESCO (WNE)](#wne-flowtemporalconsistency)
  - [RAFT Optical Flow (WNE)](#wne-raftopticalflow)
- **WanNodeExperiments/Guidance** (8)
  - [APG Guidance (WNE)](#wne-apgguidance)
  - [CFG-Zero★ (WNE)](#wne-cfgzerostar)
  - [High-Frequency Latent Injector (WNE)](#wne-highfreqlatentinjector)
  - [Rescale CFG · anti-burn (WNE)](#wne-rescalecfg)
  - [STG Spatiotemporal Skip Guidance (WNE)](#wne-stgguidance)
  - [T5 Token Attention Amplifier (WNE)](#wne-t5attentionamplifier)
  - [Tangential-Damping CFG (WNE)](#wne-tangentialdampingcfg)
  - [ZeResFDG Guidance · CADE 2.5 (WNE)](#wne-zeresfdguidance)
- **WanNodeExperiments/LipSync** (1)
  - [InfiniteTalk V2V Lip-Sync](#wne-infinitetalkv2v)
- **WanNodeExperiments/Loaders** (4)
  - [Wan GGUF Model Loader (WNE)](#wne-wanggufmodelloader)
  - [Wan MoE Expert Loader · high/low (WNE)](#wne-wanmoeexpertloader)
  - [Wan Model Loader · T2V/I2V (WNE)](#wne-wanmodelloader)
  - [Wan VACE Loader (WNE)](#wne-wanvaceloader)
- **WanNodeExperiments/LongVideo** (2)
  - [FreeLOC OOD Correction (WNE)](#wne-freeloccorrection)
  - [RIFLEx + UltraViCo Attention Decay (WNE)](#wne-ultravicoattentiondecay)
- **WanNodeExperiments/Postprocess** (1)
  - [Wan Anchor Color-Match (anti-drift)](#wananchorcolormatchc2c)
- **WanNodeExperiments/Sampling** (7)
  - [AnyFlow Any-Step Loader (WNE)](#wne-anyflowmodelloader)
  - [FreeInit Noise Reinit (WNE)](#wne-freeinitnoise)
  - [MoE Cross-Fade Router (WNE)](#wne-moecrossfaderouter)
  - [MotionMax Flow Noise Init (WNE)](#wne-motionmaxnoiseinit)
  - [Wan Flow Scheduler · sigmas (WNE)](#wne-wanflowscheduler)
  - [Wan MoE-Aware Cache (WNE)](#wne-wanmoecache)
  - [Wan Model Sampling Shift (WNE)](#wne-wanmodelsamplingshift)
- **WanNodeExperiments/TextEncoders** (2)
  - [T5Gemma Encoder Loader (WNE)](#wne-want5gemmaloader)
  - [T5Gemma Text Encode (WNE)](#wne-want5gemmatextencode)
- **WanNodeExperiments/VAE** (8)
  - [3D Anti-Alias Latent (WNE)](#wne-latent3dantialias)
  - [RefDecoder VAE Decode (WNE)](#wne-refdecodervaedecode)
  - [Temporal Causal VAE Decode (WNE)](#wne-temporalcausalvaedecode)
  - [Wan Locked Decode + Colour-Lock (WNE)](#wne-wanlockeddecode)
  - [Wan VAE Decode · tiled (WNE)](#wne-wanvaedecodetiled)
  - [Wan VAE Encode · tiled (WNE)](#wne-wanvaeencode)
  - [Wavelet Colour-Lock · WF-VAE (WNE)](#wne-waveletcolorlock)
  - [YUV Colour-Lock Decode (WNE)](#wne-yuvcolorlockdecode)
- **WanNodeExperiments/Video** (1)
  - [Video Import — any format + EXR (WNE)](#wne-videoimport)


---

## C2C/Wan_Director


### WanDirectorC2C

**Shown in the menu as:** Wan Director

Visual timeline director for Wan 2.1 / 2.2 / Fun / Animate. Drag image, text and audio clips onto the timeline, choose a model_variant, and the node emits the matching CONDITIONING, LATENT, FPS and AUDIO bundle. Inspired by WhatDreamsCost / LTX Director (MIT), redesigned for the Wan VAE shape (16ch, /8 spatial, /4 temporal) and Wan-specific options (dual-CFG for 2.2, reference image for Animate, control track for Fun).


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `backend` | choice: `native`, `kijai` | default `"native"` | Which video-model stack to drive.    native — ComfyUI's built-in Wan implementation. Connect `model` + `clip`.   kijai  — Kijai's ComfyUI-WanVideoWrapper. Connect `wan_model` + `wan_t5` (optional sockets).  PromptRelay (if enabled) is applied to whichever backbone is active and falls back to the generic-introspection patcher for any third-party model. |
| `model_variant` | choice: `wan2.1_t2v`, `wan2.1_i2v`, `wan2.2_t2v`, `wan2.2_i2v`, `wan_fun_inp`, `wan_fun_control`, `wan_animate`, `wan2.2_animate_everanimate`, … (+6) | default `"wan2.1_i2v"` | Which Wan family / mode this timeline targets. Changes which optional sliders are visible and how the latent + conditioning are assembled. |
| `duration_frames` | `INT` | default `81`, range 1…10000, step 1 | Total timeline length in pixel-space frames. Wan 2.x defaults to 81 frames (≈ 5 s @ 16 fps). |
| `duration_seconds` | `FLOAT` | default `5.0`, range 0.1…1000.0, step 0.01 | Total timeline duration in seconds (synced from frames by the UI). |
| `frame_rate` | `FLOAT` | default `16.0`, range 1.0…240.0, step 1.0 | FPS. Wan 2.x is trained at 16 fps; raise for slow-motion-like output. |
| `global_prompt` | `STRING` | default `""`, multiline | Persistent context prepended to every per-clip prompt (characters, lighting, style anchors). |
| `timeline_data` | `STRING` | default `""`, multiline | — |
| `local_prompts` | `STRING` | default `""`, multiline | — |
| `negative_prompts` | `STRING` | default `""`, multiline | — |
| `segment_lengths` | `STRING` | default `""` | — |
| `guide_strength` | `STRING` | default `""` | — |
| `display_mode` | choice: `seconds`, `frames` | default `"seconds"` | — |
| `custom_width` | `INT` | default `832`, range 0…8192, step 8 | Target width. 0 = inherit from first image clip. |
| `custom_height` | `INT` | default `480`, range 0…8192, step 8 | Target height. 0 = inherit from first image clip. |
| `resize_method` | choice: `maintain aspect ratio`, `stretch to fit`, `pad`, `crop` | default `"maintain aspect ratio"` | — |
| `cfg_high_noise` | `FLOAT` | default `3.5`, range 0.0…20.0, step 0.1 | Wan 2.2 high-noise expert CFG. Ignored for non-2.2 variants. |
| `cfg_low_noise` | `FLOAT` | default `3.5`, range 0.0…20.0, step 0.1 | Wan 2.2 low-noise expert CFG. Ignored for non-2.2 variants. |
| `ref_strength` | `FLOAT` | default `1.0`, range 0.0…2.0, step 0.05 | Wan Animate reference-image influence. Ignored for other variants. |
| `audio_target` | choice: `music_44k_stereo`, `speech_16k_mono` | default `"music_44k_stereo"` | Output AUDIO format. Use `speech_16k_mono` if you intend to feed Wan-S2V or any speech-driven pipeline downstream. |
| `enable_prompt_relay` | `BOOLEAN` | default `True` | Internal PromptRelay: bias each backbone cross-attention block so the timeline's per-clip prompts only steer their own frame span. On by default — the whole point of the timeline. Works on native ComfyUI MODEL, Kijai WANVIDEOMODEL, and arbitrary video-diffusion models (auto-falls back to generic introspection). With 2+ text/image clips the per-clip prompts become the local prompts and `global_prompt` is the anchor; with 0 or 1 clip it is a no-op. Turn off to encode one flat prompt for the whole clip. |
| `prompt_relay_epsilon` | `FLOAT` | default `0.001`, range 1e-06…0.99, step 0.0001 | PromptRelay penalty decay. <0.1 = sharp boundaries; ≥0.5 softer. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `model` | `MODEL` |  | Native Wan MODEL (2.1 / 2.2 / Fun / Animate). Required when backend='native'. |
| `clip` | `CLIP` |  | Text encoder paired with the Wan model (UMT5 for 2.x). Required when backend='native'. |
| `vae` | `VAE` |  | Wan VAE. If connected, the Director encodes reference images for i2v and forces fp32 when vae_fp32_decode=True. |
| `clip_vision` | `CLIP_VISION` |  | CLIP Vision model for image embeddings (Kijai i2v/Animate). If not connected, image_embeds output is None. |
| `optional_latent` | `LATENT` |  | Override the auto-built empty latent. |
| `control_video` | `IMAGE` |  | For Wan Fun / Animate: control sequence (depth/pose/canny). Passed through to control_video output. |
| `control_mask` | `MASK` |  | For Wan Fun Inpaint: per-frame mask track. Passed through to control_mask output. |
| `wan_model` | `WANVIDEOMODEL` |  | Kijai WanVideoWrapper model patcher (required when backend='kijai'). |
| `wan_t5` | `WANTEXTENCODER` |  | Kijai T5 text encoder (required when backend='kijai'). |
| `t5gemma` | `T5GEMMA_ENCODER` |  | R&D: connect a 'T5Gemma Encoder Loader (WNE)' to encode the Director's composed prompt through T5Gemma instead of CLIP (backend='native'). NOTE: T5Gemma hidden states are a different space/width from Wan's UMT5-XXL — this only works on a Wan model finetuned/adapted for T5Gemma; a stock Wan checkpoint will error or produce garbage. |
| `extra_args` | `WAN_DIRECTOR_EXTRA` |  | Optional: connect a 'WanDirector Extra Args' node to drive the advanced quality stack (PAG / NAG / SLG / RIFLEx / FETA / cache / dynamic-CFG / phase-shift / multi-clip / prompt-relay / EverAnimate). Leave unconnected to use clean defaults. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `model` | `MODEL` | Native MODEL (patched with PromptRelay + NAG + PAG + Dynamic CFG + AsymFlow as enabled). When backend='kijai', passthrough of input `model` if connected. |
| 1 | `positive` | `CONDITIONING` | Positive CONDITIONING (native branch) with guide_strength embedded. Empty list when backend='kijai'. |
| 2 | `negative` | `CONDITIONING` | Negative CONDITIONING (native branch). Empty list when backend='kijai'. |
| 3 | `video_latent` | `LATENT` | Wan latent: VAE-encoded reference for i2v (if VAE connected) or empty latent. Channels=16, /8 spatial, /4 temporal. |
| 4 | `frame_rate` | `FLOAT` | Frame rate echoed for downstream sampler/saver nodes. |
| 5 | `combined_audio` | `AUDIO` | Audio waveform mixed from the timeline's audio segments. |
| 6 | `reference_image` | `IMAGE` | Reference image (first image clip) — used by Wan I2V/Animate as the start/reference frame. Black image if none. |
| 7 | `info` | `STRING` | JSON: resolved backend, variant, latent shape, segment count, audio sample rate, prompt-relay status, quality stack status, warnings. |
| 8 | `wan_model` | `WANVIDEOMODEL` | Kijai WANVIDEOMODEL (only populated when backend='kijai'; PromptRelay-patched in place if enabled). |
| 9 | `wan_text_embeds` | `WANVIDEOTEXTEMBEDS` | Kijai WANVIDEOTEXTEMBEDS dict (only populated when backend='kijai'). Feed directly into WanVideoSampler. |
| 10 | `tracks_program` | `STRING` | JSON: timeline schema_version + normalised lora/camera/seed/pose tracks for downstream applier nodes. |
| 11 | `control_video` | `IMAGE` | Control video passthrough (IMAGE) for Wan Fun/Animate. None if not connected. |
| 12 | `control_mask` | `MASK` | Control mask passthrough (MASK) for Wan Fun Inpaint. None if not connected. |
| 13 | `guide_data` | `STRING` | JSON: per-segment guide_strength values for downstream guide applier nodes. |
| 14 | `quality_recipe` | `STRING` | JSON: quality recipe config (SLG, FETA, RIFLEx, cache, FreeInit, phase-shift) for downstream sampler. |


### WanDirectorInspector

**Shown in the menu as:** Wan Director — Inspector

Reads the Wan Director's JSON status outputs (info / quality_recipe / tracks_program / guide_data) and renders a human-readable plan summary plus typed scalars (segment count, frame rate, prompt-relay flag, active-feature list). Display/inspection only — it does not re-apply quality features (the Director already patched the model).


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `info` | `STRING` | **connection-only** | — |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `quality_recipe` | `STRING` | **connection-only** | — |
| `tracks_program` | `STRING` | **connection-only** | — |
| `guide_data` | `STRING` | **connection-only** | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `summary` | `STRING` | Plain-English multi-line plan summary (feed to a Show-Text node). |
| 1 | `segment_count` | `INT` | Number of text/prompt segments on the timeline. |
| 2 | `frame_rate` | `FLOAT` | Resolved frame rate. |
| 3 | `prompt_relay_applied` | `BOOLEAN` | True if PromptRelay was actually applied (not just enabled). |
| 4 | `active_features` | `STRING` | Comma-separated list of active quality features, e.g. 'pag(scale=2.0), nag(scale=11.0)'. |


---

## WanNodeExperiments/Audio


### WNE_AdvancedAudioSeparator

**Shown in the menu as:** Advanced Audio Separator (Vocals)

Isolate clean vocals (BS-RoFormer/Demucs/MDX) for lip-sync driving.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `audio` | `AUDIO` |  | Mixed audio ({'waveform':[B,C,S],'sample_rate'}). |
| `model` | choice: `bs_roformer_viperx`, `flow_bs_roformer`, `htdemucs_v4`, `uvr_mdx_net` | default `"bs_roformer_viperx"` | bs_roformer_viperx = Band-Split RoFormer (viperx ckpt, clean vocal isolation + mid-range retention). flow_bs_roformer = roformer variant tuned for compressed inputs. htdemucs_v4 = hybrid transformer (heavy acoustic/synth bleed). uvr_mdx_net = classic MDX-Net frequency filtering. Backends: pip install audio-separator (roformer/mdx) / demucs (htdemucs). |
| `msr_hifi_restore` | `BOOLEAN` | default `False` | Multi-Stage Restoration: pipe the separated vocals through a HiFi++ GAN artifact restorer. Requires a hifi_pp checkpoint at ComfyUI/models/audio_restore/hifi_pp.ckpt — if absent the stage warns and passes vocals through unchanged (never fakes it). |

**Hidden inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `unique_id` | `UNIQUE_ID` |  | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `vocals` | `AUDIO` | — |
| 1 | `instrumental` | `AUDIO` | — |
| 2 | `info` | `STRING` | — |


---

## WanNodeExperiments/Director


### WanDirectorExtraArgs

**Shown in the menu as:** Wan Director — Extra Args

Advanced quality stack for WanDirector (PAG/NAG/SLG/RIFLEx/FETA/cache/dynamic-CFG/phase-shift/multi-clip/prompt-relay/EverAnimate). Each block's params show only when its toggle is on. Wire into WanDirector's 'extra_args'.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `everanimate_stage` | choice: `stage1_480p`, `stage2_480p`, `stage3_720p_beta` | default `"stage2_480p"` | Which EverAnimate LoRA checkpoint to apply on top of Wan2.2-Animate-14B:   stage1_480p     — base motion fidelity (480p training).   stage2_480p     — Restorative Flow Matching, sharper temporal coherence (recommended).   stage3_720p_beta — 720p beta with higher detail; needs more VRAM. Ignored for non-EverAnimate variants. |
| `everanimate_num_chunks` | `INT` | default `1`, range 1…50, step 1 | Long-horizon chunk count. 1 = single ~5 s clip (standard Wan2.2-Animate). ≥2 enables EverAnimate's Persistent Latent Propagation across anchor frames for minute-scale animation. Ignored for non-EverAnimate variants. |
| `everanimate_overlap_frames` | `INT` | default `4`, range 0…16, step 1 | Frames of latent overlap between consecutive chunks (anchor padding). Higher = smoother seams but slower. Ignored if num_chunks=1 or non-EverAnimate variant. |
| `everanimate_lora_strength` | `FLOAT` | default `1.0`, range 0.0…2.0, step 0.05 | EverAnimate rank-32 LoRA strength. 1.0 = paper default. Ignored for non-EverAnimate variants. |
| `everanimate_anchor_strategy` | choice: `auto`, `first_only`, `first_plus_random_3` | default `"auto"` | Anchor-frame selection for chunks 2+:   auto                 — first chunk uses first frame only, later chunks use first + 3 random.   first_only           — always 1 anchor (faster, slight quality loss).   first_plus_random_3  — always 4 anchors (paper-default; best quality). Ignored for non-EverAnimate variants. |
| `enable_dynamic_cfg` | `BOOLEAN` | default `False` | Cosine-ramped dynamic CFG across denoising steps. Early steps get 1.2× CFG (stronger structure), late steps get 0.7× (softer detail). Prevents oversaturation and improves quality. |
| `guidance_rescale_phi` | `FLOAT` | default `0.0`, range 0.0…1.0, step 0.05 | Guidance rescale (phi). Rescales guided output to match conditional std-deviation, preventing color oversaturation at high CFG. 0=off, 0.7=recommended for Wan 2.2. Requires enable_dynamic_cfg=True. |
| `pag_scale` | `FLOAT` | default `0.0`, range 0.0…5.0, step 0.1 | Perturbed Attention Guidance scale. Improves prompt adherence by guiding away from identity-attention outputs. 0=off, 1.0–3.0 typical. |
| `enable_phase_shift` | `BOOLEAN` | default `False` | Phase-shift sampling: Euler for early steps (structure), DPM++ 2M for late steps (detail). Uses smooth sigma crossfade. |
| `phase_shift_pct` | `FLOAT` | default `0.7`, range 0.3…0.95, step 0.05 | Step fraction where phase-shift transitions from Euler to DPM++. |
| `vae_fp32_decode` | `BOOLEAN` | default `True` | Force VAE decode in fp32 for maximum quality. Wan VAE produces significantly better results in fp32 (recommended by HuggingFace). Uses more VRAM during decode only. |
| `enable_multi_clip` | `BOOLEAN` | default `False` | Multi-slot CLIP conditioning. Split prompts into structure (early) and detail (late) phases for finer control over generation. |
| `structure_prompt` | `STRING` | default `""`, multiline | Structure prompt (active during early denoising, 0–35%). Focus on composition, layout, camera angles, scene description. Only used when enable_multi_clip=True. |
| `detail_prompt` | `STRING` | default `""`, multiline | Detail prompt (active during late denoising, 55–100%). Focus on textures, materials, lighting, color grading. Only used when enable_multi_clip=True. |
| `enable_nag` | `BOOLEAN` | default `False` | Normalized Attention Guidance: boosts prompt adherence via attention-space CFG. |
| `nag_scale` | `FLOAT` | default `11.0`, range 0.0…30.0, step 0.5 | NAG guidance scale. Higher = stronger guidance. |
| `enable_asymflow` | `BOOLEAN` | default `False` | AsymFlow time-shift for improved temporal consistency. |
| `asymflow_shift` | `FLOAT` | default `3.0`, range 0.1…20.0, step 0.1 | AsymFlow shift parameter. |
| `cache_type` | choice: `none`, `teacache`, `magcache`, `easycache` | default `"none"` | Inference caching strategy. Speeds up generation by skipping redundant transformer passes. |
| `cache_threshold` | `FLOAT` | default `0.1`, range 0.0…1.0, step 0.01 | Cache skip threshold. Lower = more aggressive caching (faster but less accurate). |
| `enable_slg` | `BOOLEAN` | default `False` | Skip-Layer Guidance: run a second pass with layers removed for quality boost. |
| `slg_layers` | `STRING` | default `""` | Comma-separated layer indices to skip (e.g. '7,8,9'). Empty = auto-select. |
| `slg_scale` | `FLOAT` | default `0.7`, range 0.0…2.0, step 0.05 | SLG guidance scale. |
| `enable_feta` | `BOOLEAN` | default `False` | Frequency-Enhanced Temporal Attention for better frame coherence. |
| `feta_scale` | `FLOAT` | default `0.5`, range 0.0…2.0, step 0.05 | FETA scale. 0 = off. |
| `enable_riflex` | `BOOLEAN` | default `False` | RIFLEx RoPE rescaling for length extrapolation beyond training length. |
| `riflex_k` | `INT` | default `2`, range 1…8, step 1 | Number of lowest RoPE frequencies to rescale. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `extra_args` | `WAN_DIRECTOR_EXTRA` | — |


---

## WanNodeExperiments/Flow


### WNE_FlowTemporalConsistency

**Shown in the menu as:** Flow Temporal Consistency — FlowVid/FRESCO (WNE)

Lock frame-to-frame geometry / kill jitter: blends every frame with the occlusion-masked, flow-aligned PREVIOUS output frame (FlowVid/FRESCO forward-backward-consistency propagation). Tracked content stays stable; occluded/disoccluded regions fall back to the per-frame image (no smearing). Pass a precomputed WAN_FLOW, or let it run RAFT internally.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `images` | `IMAGE` |  | — |
| `strength` | `FLOAT` | default `0.6`, range 0.0…1.0, step 0.01 | How strongly each frame snaps to the flow-aligned previous frame (1 = max stability, may ghost). |
| `occlusion_gate` | `BOOLEAN` | default `True` | Only propagate where the forward-backward flow check is reliable (recommended). |
| `model_size` | choice: `large`, `small` | default `"large"` | — |
| `iterations` | `INT` | default `12`, range 1…32 | — |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `flow` | `WAN_FLOW` |  | Precomputed flow from WNE_RAFTOpticalFlow (skips recompute). |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `images` | `IMAGE` | — |


### WNE_RAFTOpticalFlow

**Shown in the menu as:** RAFT Optical Flow (WNE)

RAFT optical flow (torchvision, official) for an IMAGE video batch — forward+backward flow and forward-backward-consistency occlusion masks, the shared foundation for the FlowVid/FRESCO/TokenFlow consistency stages.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `images` | `IMAGE` |  | — |
| `model_size` | choice: `large`, `small` | default `"large"` | RAFT backbone. large = most accurate; small = faster/less VRAM. |
| `iterations` | `INT` | default `12`, range 1…32 | RAFT refinement iterations (more = sharper flow, slower). |
| `device` | choice: `auto`, `gpu`, `cpu` | default `"auto"` | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `flow` | `WAN_FLOW` | — |
| 1 | `flow_preview` | `IMAGE` | — |


---

## WanNodeExperiments/Guidance


### WNE_APGGuidance

**Shown in the menu as:** APG Guidance (WNE)

Adaptive Projected Guidance: eliminate oversaturation at high CFG.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `model` | `MODEL` |  | — |
| `eta` | `FLOAT` | default `0.0`, range -2.0…2.0, step 0.01 | Reverse momentum on the guidance running average. |
| `norm_threshold` | `FLOAT` | default `15.0`, range 0.0…100.0, step 0.1 | Clamp guidance L2 norm (0 = off). |
| `parallel_weight` | `FLOAT` | default `0.0`, range 0.0…1.0, step 0.01 | Keep this fraction of the parallel component (0 = full APG). |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `positive` | `CONDITIONING` |  | — |
| `negative` | `CONDITIONING` |  | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `model` | `MODEL` | — |


### WNE_CFGZeroStar

**Shown in the menu as:** CFG-Zero★ (WNE)

CFG-Zero*: optimized uncond scale + early-step zero-init.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `model` | `MODEL` |  | — |
| `zero_init_steps` | `INT` | default `1`, range 0…100 | Number of first steps to zero-init. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `model` | `MODEL` | — |


### WNE_HighFreqLatentInjector

**Shown in the menu as:** High-Frequency Latent Injector (WNE)

Inject conditional high-frequency texture during final steps.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `model` | `MODEL` |  | — |
| `start_step` | `INT` | default `15`, range 0…1000 | — |
| `end_step` | `INT` | default `1000`, range 0…1000 | — |
| `frequency_weight` | `FLOAT` | default `0.15`, range 0.0…2.0, step 0.01 | — |
| `blur_kernel_size` | `INT` | default `7`, range 3…31, step 2 | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `model` | `MODEL` | — |


### WNE_RescaleCFG

**Shown in the menu as:** Rescale CFG · anti-burn (WNE)

Rescale CFG to the conditional std (anti-burn).


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `model` | `MODEL` |  | — |
| `multiplier` | `FLOAT` | default `0.7`, range 0.0…1.0, step 0.01 | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `model` | `MODEL` | — |


### WNE_STGGuidance

**Shown in the menu as:** STG Spatiotemporal Skip Guidance (WNE)

Spatiotemporal Skip Guidance: self-perturbation weak model.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `model` | `MODEL` |  | — |
| `skip_mode` | choice: `attention`, `residual` | default `"attention"` | — |
| `stg_scale` | `FLOAT` | default `1.0`, range 0.0…10.0, step 0.05 | — |
| `layer_indices` | `STRING` | default `"8,9"` | Comma-separated attention layer indices to skip. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `model` | `MODEL` | — |


### WNE_T5AttentionAmplifier

**Shown in the menu as:** T5 Token Attention Amplifier (WNE)

Amplify important T5 tokens / suppress stop-words in cross-attention.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `model` | `MODEL` |  | — |
| `token_amplify_factor` | `FLOAT` | default `1.5`, range 0.1…5.0, step 0.05 | — |
| `suppress_stopwords` | `BOOLEAN` | default `True` | — |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `clip` | `CLIP` |  | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `model` | `MODEL` | — |


### WNE_TangentialDampingCFG

**Shown in the menu as:** Tangential-Damping CFG (WNE)

Tangential-damping CFG: damp the radial part of the negative branch.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `model` | `MODEL` |  | — |
| `damping` | `FLOAT` | default `0.5`, range 0.0…1.0, step 0.01 | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `model` | `MODEL` | — |


### WNE_ZeResFDGuidance

**Shown in the menu as:** ZeResFDG Guidance · CADE 2.5 (WNE)

Frequency-decoupled + rescaled + zero-projected guidance (CADE 2.5).


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `model` | `MODEL` |  | — |
| `freq_split` | `FLOAT` | default `0.7`, range 0.0…1.0, step 0.01 | Low-frequency gain (high gain = 2 - this). |
| `energy_rescale` | `FLOAT` | default `0.7`, range 0.0…1.0, step 0.01 | Blend toward magnitude-matched prediction. |
| `zero_proj_steps` | `INT` | default `3`, range 0…1000 | Apply zero-projection for this many early steps. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `positive` | `CONDITIONING` |  | — |
| `negative` | `CONDITIONING` |  | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `model` | `MODEL` | — |


---

## WanNodeExperiments/LipSync


### WNE_InfiniteTalkV2V

**Shown in the menu as:** InfiniteTalk V2V Lip-Sync

InfiniteTalk V2V lip-sync. Auto face mask when none provided; audio embeds + generation delegate to the vendored MultiTalk pipeline (wire the Wan model stack).


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `images` | `IMAGE` |  | Source video frames [B,H,W,C] 0-1. |
| `audio` | `AUDIO` |  | Driving vocals (from Advanced Audio Separator). |
| `fps` | `FLOAT` | default `25.0`, range 1.0…120.0, step 0.5 | Video fps — aligns audio embeds to frames. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `mask` | `MASK` |  | Face region [B,H,W]. Omit = automatic ONNX face-detection bounding-box mask (in-repo). |
| `wav2vec_model` | `WAV2VECMODEL` |  | From the vendored 'Wav2Vec Model Loader' node (wanwrapper/multitalk). |
| `wan_model` | `WANVIDEOMODEL` |  | Wan video model with the InfiniteTalk/MultiTalk weights loaded (vendored loader). |
| `vae` | `WANVAE` |  | Wan VAE (vendored loader). |

**Hidden inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `unique_id` | `UNIQUE_ID` |  | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `images` | `IMAGE` | — |
| 1 | `face_mask` | `MASK` | — |


---

## WanNodeExperiments/Loaders


### WNE_WanGGUFModelLoader

**Shown in the menu as:** Wan GGUF Model Loader (WNE)

Load a GGUF-quantized Wan model as native ComfyUI MODEL.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `gguf_name` | `STRING` | default `""` | Path/name of the diffusion model file. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `force_offload` | `BOOLEAN` | default `True` | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `model` | `MODEL` | — |


### WNE_WanMoEExpertLoader

**Shown in the menu as:** Wan MoE Expert Loader · high/low (WNE)

Load Wan 2.2 high-noise + low-noise MoE expert models.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `high_noise_model` | `STRING` | default `""` | Path/name of the diffusion model file. |
| `low_noise_model` | `STRING` | default `""` | Path/name of the diffusion model file. |
| `weight_dtype` | choice: `default`, `fp16`, `bf16`, `fp8_e4m3fn`, `fp8_e5m2` | default `"default"` | — |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `force_offload` | `BOOLEAN` | default `True` | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `model_high` | `MODEL` | — |
| 1 | `model_low` | `MODEL` | — |


### WNE_WanModelLoader

**Shown in the menu as:** Wan Model Loader · T2V/I2V (WNE)

Load a Wan 2.2 T2V/I2V diffusion model as native ComfyUI MODEL.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `model_name` | `STRING` | default `""` | Path/name of the diffusion model file. |
| `weight_dtype` | choice: `default`, `fp16`, `bf16`, `fp8_e4m3fn`, `fp8_e5m2` | default `"default"` | — |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `force_offload` | `BOOLEAN` | default `False` | Empty VRAM cache after load (low-VRAM). |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `model` | `MODEL` | — |


### WNE_WanVACELoader

**Shown in the menu as:** Wan VACE Loader (WNE)

Load a Wan VACE control module as native ComfyUI MODEL.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `vace_model` | `STRING` | default `""` | Path/name of the diffusion model file. |
| `weight_dtype` | choice: `default`, `fp16`, `bf16`, `fp8_e4m3fn`, `fp8_e5m2` | default `"default"` | — |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `force_offload` | `BOOLEAN` | default `False` | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `model` | `MODEL` | — |


---

## WanNodeExperiments/LongVideo


### WNE_FreeLOCCorrection

**Shown in the menu as:** FreeLOC OOD Correction (WNE)

FreeLOC VRPR + tiered sparse attention for long video.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `model` | `MODEL` |  | — |
| `enable_vrpr` | `BOOLEAN` | default `True` | — |
| `enable_tsa` | `BOOLEAN` | default `True` | — |
| `probe_calibration_path` | `STRING` | default `""` | — |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `local_window` | `INT` | default `512`, range 16…100000 | — |
| `sparse_stride` | `INT` | default `4`, range 2…64 | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `model` | `MODEL` | — |


### WNE_UltraViCoAttentionDecay

**Shown in the menu as:** RIFLEx + UltraViCo Attention Decay (WNE)

UltraViCo logit decay + RIFLEx RoPE for long-video extrapolation.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `model` | `MODEL` |  | — |
| `decay_factor` | `FLOAT` | default `0.5`, range 0.0…1.0, step 0.01 | — |
| `training_window` | `INT` | default `81`, range 1…100000 | Training window (tokens on native path / frames on Kijai). |
| `first_frame_decay_override` | `FLOAT` | default `0.0`, range -1.0…0.0, step 0.01 | Stronger (negative) decay for first-frame pairs; 0 = off. |
| `riflex_freq_index` | `INT` | default `0`, range 0…16 | RIFLEx RoPE frequency index (0 = off). |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `model` | `MODEL` | — |


---

## WanNodeExperiments/Postprocess


### WanAnchorColorMatchC2C

**Shown in the menu as:** Wan Anchor Color-Match (anti-drift)

Remove Wan colour/contrast DRIFT in long / extended / looped video by matching every frame to a fixed anchor (first frame or a wired reference), not to the previous frame — so bias never accumulates. Wire after the VAE decode. mean_std = fast Reinhard transfer; histogram = stronger (needs cv2).


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `images` | `IMAGE` |  | — |
| `anchor` | choice: `first_frame`, `last_frame`, `reference` | default `"first_frame"` | Colour target every frame is matched to. 'reference' uses the wired reference image. |
| `method` | choice: `mean_std`, `histogram` | default `"mean_std"` | mean_std = fast per-channel Reinhard transfer; histogram = per-channel CDF match (stronger, needs OpenCV). |
| `strength` | `FLOAT` | default `1.0`, range 0.0…1.0, step 0.01 | 0 = no change, 1 = full match to the anchor. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `reference` | `IMAGE` |  | Anchor frame when anchor='reference' (uses its first frame). |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `images` | `IMAGE` | — |


---

## WanNodeExperiments/Sampling


### WNE_AnyFlowModelLoader

**Shown in the menu as:** AnyFlow Any-Step Loader (WNE)

AnyFlow flow-map any-step schedule (4→32 NFE, monotonic).


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `model` | `MODEL` |  | — |
| `flow_map_checkpoint` | `STRING` | default `""` | Path to distilled flow-map checkpoint (optional). |
| `target_nfe` | `INT` | default `8`, range 1…128 | Number of function evals; live-adjustable per queue. |
| `causal_mode` | `BOOLEAN` | default `False` | FAR/streaming causal sampling order. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `model` | `MODEL` | — |


### WNE_FreeInitNoise

**Shown in the menu as:** FreeInit Noise Reinit (WNE)

FreeInit low-frequency noise reinitialization.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `latent` | `LATENT` |  | — |
| `num_iters` | `INT` | default `1`, range 1…10 | — |
| `filter_method` | choice: `gaussian`, `butterworth`, `ideal`, `box` | default `"butterworth"` | — |
| `d_s` | `FLOAT` | default `0.25`, range 0.0…1.0, step 0.01 | — |
| `d_t` | `FLOAT` | default `0.25`, range 0.0…1.0, step 0.01 | — |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `seed` | `INT` | default `0`, range 0…18446744073709551615 | — |
| `butterworth_n` | `INT` | default `4`, range 1…16 | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `latent` | `LATENT` | — |


### WNE_MoECrossFadeRouter

**Shown in the menu as:** MoE Cross-Fade Router (WNE)

Cross-fade across the MoE expert switch to remove boundary pops.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `model` | `MODEL` |  | — |
| `switch_step` | `INT` | default `10`, range 0…1000 | — |
| `fade_window` | `INT` | default `3`, range 1…50 | — |
| `fade_curve` | choice: `linear`, `cubic` | default `"linear"` | — |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `model_low` | `MODEL` |  | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `model` | `MODEL` | — |


### WNE_MotionMaxNoiseInit

**Shown in the menu as:** MotionMax Flow Noise Init (WNE)

Flow-guided noise structuring along reference motion (RAFT).


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `latent` | `LATENT` |  | — |
| `reference_video` | `IMAGE` |  | — |
| `flow_strength` | `FLOAT` | default `0.5`, range 0.0…1.0, step 0.01 | — |
| `motion_extrapolation` | `INT` | default `0`, range 0…64 | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `latent` | `LATENT` | — |


### WNE_WanFlowScheduler

**Shown in the menu as:** Wan Flow Scheduler · sigmas (WNE)

Wan flow-matching sigma schedule (shift) as native SIGMAS.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `steps` | `INT` | default `20`, range 1…1000 | — |
| `shift` | `FLOAT` | default `5.0`, range 0.0…100.0, step 0.1 | Flow-matching shift (Wan 2.2 typically 3-8). |
| `denoise` | `FLOAT` | default `1.0`, range 0.0…1.0, step 0.01 | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `sigmas` | `SIGMAS` | — |


### WNE_WanMoECache

**Shown in the menu as:** Wan MoE-Aware Cache (WNE)

Per-expert TeaCache/MagCache with high→low boundary reset.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `model_high` | `MODEL` |  | — |
| `model_low` | `MODEL` |  | — |
| `switch_step` | `INT` | default `10`, range 0…1000 | — |
| `cache_mode` | choice: `teacache`, `magcache` | default `"teacache"` | — |
| `thresh_high` | `FLOAT` | default `0.15`, range 0.0…2.0, step 0.01 | — |
| `thresh_low` | `FLOAT` | default `0.2`, range 0.0…2.0, step 0.01 | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `model_high` | `MODEL` | — |
| 1 | `model_low` | `MODEL` | — |


### WNE_WanModelSamplingShift

**Shown in the menu as:** Wan Model Sampling Shift (WNE)

Apply Wan/SD3 flow-matching shift to the model sampling.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `model` | `MODEL` |  | — |
| `shift` | `FLOAT` | default `5.0`, range 0.0…100.0, step 0.1 | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `model` | `MODEL` | — |


---

## WanNodeExperiments/TextEncoders


### WNE_WanT5GemmaLoader

**Shown in the menu as:** T5Gemma Encoder Loader (WNE)

Load a converted T5Gemma encoder (safetensors or GGUF Q8_0) with selectable attention backend.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `model_file` | choice: `<run tools/convert_t5gemma_encoder.py first>` |  | Converted single-file T5Gemma encoder from models/text_encoders (made by tools/convert_t5gemma_encoder.py). |
| `attention_backend` | choice: `auto`, `sdpa`, `eager`, `flash_attention_2` | default `"auto"` | sdpa = PyTorch fused attention (default), eager = reference math, flash_attention_2 = requires flash-attn package (fails loudly if absent). |
| `dtype` | choice: `auto`, `bf16`, `fp16`, `fp32` | default `"auto"` | Compute dtype (auto = bf16 on CUDA, fp32 on CPU). |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `t5gemma` | `T5GEMMA_ENCODER` | — |


### WNE_WanT5GemmaTextEncode

**Shown in the menu as:** T5Gemma Text Encode (WNE)

T5Gemma text encoding → CONDITIONING (last hidden states + mask + mean-pooled). NOTE: stock Wan checkpoints were trained on UMT5-XXL embeddings — T5Gemma states are a different space/width, for adapter/finetune R&D, not a drop-in swap on pretrained Wan.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `t5gemma` | `T5GEMMA_ENCODER` |  | — |
| `text` | `STRING` | default `""`, multiline | — |
| `max_tokens` | `INT` | default `512`, range 8…8192 | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `conditioning` | `CONDITIONING` | — |
| 1 | `info` | `STRING` | — |


---

## WanNodeExperiments/VAE


### WNE_Latent3DAntiAlias

**Shown in the menu as:** 3D Anti-Alias Latent (WNE)

Edge-aware 3D anti-alias filter in latent space.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `samples` | `LATENT` |  | — |
| `spatial_sigma` | `FLOAT` | default `0.6`, range 0.0…5.0, step 0.05 | — |
| `temporal_sigma` | `FLOAT` | default `0.4`, range 0.0…5.0, step 0.05 | — |
| `threshold` | `FLOAT` | default `0.05`, range 0.0…1.0, step 0.01 | Edge preservation: higher keeps more edges sharp. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `latent` | `LATENT` | — |


### WNE_RefDecoderVAEDecode

**Shown in the menu as:** RefDecoder VAE Decode (WNE)

Reference-conditioned decode: recover detail/colour from a reference frame.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `samples` | `LATENT` |  | — |
| `vae` | `VAE` |  | — |
| `reference_image` | `IMAGE` |  | — |
| `dropout_rate` | `FLOAT` | default `0.1`, range 0.0…1.0, step 0.01 | — |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `detail_weight` | `FLOAT` | default `0.25`, range 0.0…2.0, step 0.01 | — |
| `color_match_method` | choice: `reinhard`, `mkl`, `hm-mvgd` | default `"reinhard"` | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `image` | `IMAGE` | — |


### WNE_TemporalCausalVAEDecode

**Shown in the menu as:** Temporal Causal VAE Decode (WNE)

Temporal-overlap decode with pixel-space cross-fade (no strobing).


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `samples` | `LATENT` |  | — |
| `vae` | `VAE` |  | — |
| `temporal_overlap` | `INT` | default `9`, range 0…64 | Overlapping pixel frames cross-faded between windows. |
| `spatial_tile_size` | `INT` | default `512`, range 0…4096, step 64 | — |
| `spatial_overlap` | `INT` | default `64`, range 0…512, step 16 | — |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `latent_window` | `INT` | default `16`, range 2…256 | Latent frames decoded per window. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `image` | `IMAGE` | — |


### WNE_WanLockedDecode

**Shown in the menu as:** Wan Locked Decode + Colour-Lock (WNE)

Largest-tile decode with per-window colour-lock to a reference.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `samples` | `LATENT` |  | — |
| `vae` | `VAE` |  | — |
| `reference_frame` | `IMAGE` |  | — |
| `max_tile_size` | `INT` | default `0`, range 0…4096, step 64 | 0 = fully non-tiled (largest). |
| `color_match_method` | choice: `reinhard`, `mkl`, `hm-mvgd` | default `"reinhard"` | — |
| `apply_every_window` | `BOOLEAN` | default `True` | — |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `window_size` | `INT` | default `16`, range 1…1024 | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `image` | `IMAGE` | — |


### WNE_WanVAEDecodeTiled

**Shown in the menu as:** Wan VAE Decode · tiled (WNE)

Spatial+temporal tiled Wan VAE decode with graceful fallback.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `samples` | `LATENT` |  | — |
| `vae` | `VAE` |  | — |
| `tile_size` | `INT` | default `512`, range 0…4096, step 64 | 0 = no spatial tiling. |
| `overlap` | `INT` | default `64`, range 0…512, step 16 | — |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `temporal_tile` | `INT` | default `0`, range 0…256 | Latent frames per temporal tile (0 = off). |
| `temporal_overlap` | `INT` | default `8`, range 0…64 | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `image` | `IMAGE` | — |


### WNE_WanVAEEncode

**Shown in the menu as:** Wan VAE Encode · tiled (WNE)

Encode video frames to a Wan latent (tiled, V2V/I2V).


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `pixels` | `IMAGE` |  | — |
| `vae` | `VAE` |  | — |
| `tile_size` | `INT` | default `0`, range 0…4096, step 64 | 0 = no spatial tiling. |
| `overlap` | `INT` | default `64`, range 0…512, step 16 | — |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `frames` | choice: `as core`, `pad to 4n+1` | default `"as core"` | The Wan VAE keeps 4n+1 frames and silently drops the rest (10, 11 or 12 frames in -> 9 out). as core: unchanged behaviour, with a warning naming the frames dropped. pad to 4n+1: repeat the last frame up to the next 4n+1; WNE Wan VAE Decode (tiled) and C2C VAE Quality Decode trim the result back to the source length. Core VAE Decode does not know the padding and keeps the extra frames. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `latent` | `LATENT` | — |


### WNE_WaveletColorLock

**Shown in the menu as:** Wavelet Colour-Lock · WF-VAE (WNE)

Lock low-frequency wavelet subbands to a reference (kill tiling seams).


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `reference_frame` | `IMAGE` |  | — |
| `wavelet_levels` | `INT` | default `2`, range 1…5 | — |
| `lock_subbands` | choice: `LLL only`, `LLL+LLH`, `all-low` | default `"LLL only"` | — |
| `cache_mode` | `BOOLEAN` | default `False` | — |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `images` | `IMAGE` |  | — |
| `samples` | `LATENT` |  | — |
| `vae` | `VAE` |  | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `image` | `IMAGE` | — |


### WNE_YUVColorLockDecode

**Shown in the menu as:** YUV Colour-Lock Decode (WNE)

Lock luma, colour-match chroma (or vice-versa) in YCbCr space.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `samples` | `LATENT` |  | — |
| `vae` | `VAE` |  | — |
| `reference_frame` | `IMAGE` |  | — |
| `yuv_lock_mode` | choice: `luma_only`, `chroma_only`, `both` | default `"luma_only"` | — |
| `chroma_blend` | `FLOAT` | default `0.7`, range 0.0…1.0, step 0.01 | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `image` | `IMAGE` | — |


---

## WanNodeExperiments/Video


### WNE_VideoImport

**Shown in the menu as:** Video Import — any format + EXR (WNE)

Import ANY video (.mp4/.mov/.webm/.mkv/.avi/.gif…) or an OpenEXR file/sequence (linear-HDR, view-transformed to sRGB) or an image-sequence folder, as an IMAGE batch — the source for WanDirector control-video / the flow vid2vid editor. Drop files in ComfyUI/input (or use the upload button).


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `video` | choice: `(put a video/EXR in ComfyUI/input)` |  | A video / .exr / sequence-folder in ComfyUI/input (use the '📁 upload video / EXR' button to add one). |
| `frame_load_cap` | `INT` | default `0`, range 0…100000 | Max frames to load (0 = all). |
| `skip_first_frames` | `INT` | default `0`, range 0…100000 | — |
| `select_every_nth` | `INT` | default `1`, range 1…100 | — |
| `force_rate` | `FLOAT` | default `0.0`, range 0.0…240.0, step 0.01 | Override output frame rate (0 = use the file's native rate / 24 for EXR seq). |
| `target_width` | `INT` | default `0`, range 0…8192, step 8 | Resize width (0 = native; if only one of w/h set, keep aspect). |
| `target_height` | `INT` | default `0`, range 0…8192, step 8 | — |
| `exr_view_transform` | choice: `sRGB`, `gamma2.2`, `linear` | default `"sRGB"` | How to map linear-HDR EXR to 0..1 for display/editing. |
| `exr_exposure` | `FLOAT` | default `0.0`, range -10.0…10.0, step 0.1 | Exposure (stops) applied before the EXR view transform. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `images` | `IMAGE` | — |
| 1 | `frame_rate` | `FLOAT` | — |
| 2 | `frame_count` | `INT` | — |
| 3 | `width` | `INT` | — |
| 4 | `height` | `INT` | — |
