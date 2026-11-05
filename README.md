# ComfyUI-WanNodeExperiments (WNE)

An R&D node pack for **Wan 2.2** (MoE DiT) — model loaders, text encoders, VAE,
samplers, guidance experiments, long-video strategies, optical-flow temporal
consistency, audio/lip-sync, and the **WanDirector** multi-track timeline.

> **Full per-parameter documentation lives in [NODE_REFERENCE.md](NODE_REFERENCE.md)**
> — every node, every input, its type, default, valid range and what it does,
> generated directly from the code so it cannot drift out of date.
> This README is the orientation layer: what the groups are *for* and which
> node to reach for.

**License:** Apache-2.0. Built on work by **Kijai** (ComfyUI-WanVideoWrapper),
**wuwukaka** (ComfyUI-WanAnimatePlus) and **WhatDreamsCost** (LTX Director,
used with permission). See `NOTICE` and `MODIFICATIONS.md`.

---

## Is this the pack you want?

This is the **experimental / research** pack. Nodes here explore techniques
that may change shape between versions.

| If you want… | Use |
|---|---|
| Wan-Animate **preprocessing** (pose, face crops, gaze, landmarks) | `ComfyUI-WanAnimatePreprocessV2` |
| General VFX / compositing / masking utilities | `ComfyUI-CustomNodePacks` |
| Nuke-style compositing operators | `ComfyUI-NukeMaxNodes` |
| Wan 2.2 model loading, sampling, guidance R&D, the Director timeline | **this pack** |

> **Load-order note:** WanDirector exists in both this pack and
> `ComfyUI-CustomNodePacks`. **WNE wins the load order** — if you edit the
> Director and see no change, you edited the CNP copy. Keep both in sync.

---

## Node groups

### `WanNodeExperiments/Loaders`, `/TextEncoders`, `/VAE`
Getting a Wan 2.2 model, its text encoder and VAE into memory. Includes the
**T5Gemma** encoder path (research — T5Gemma does *not* share UMT5's embedding
space, so it is not a drop-in replacement for the stock encoder).

### `WanNodeExperiments/Sampling`, `/Guidance`
The sampler and the guidance stack (APG, TCFG, rescaling, attention
interception). Guidance nodes are where most of the R&D churn happens —
read each node's description before wiring it into a production graph.
On ComfyUI 0.39 and later (fixed 2026-11-03), STG, T5 Attention Amplifier, UltraViCo and FreeLOC reach the model's
attention again:
- Core now passes attention inputs in a wrapper that these nodes did not read. STG crashed and the other three
  silently did nothing.
- The fix is tested on CPU with stand-in attention; it has **not yet been run with real Wan weights on a GPU**.
- UltraViCo and FreeLOC build a full token-by-token bias. On long or large clips this can need several GB per layer,
  so start with short, small clips.
- STG's `layer_indices` are transformer **block** numbers. Only the self-attention of those blocks is skipped.
- Chaining (fixed 2026-11-04): two guidance or sampler nodes that wrap the model, or one of them after another
  pack's model wrapper (TeaCache, MagCache ...), now all run. Before, the last one silently replaced the others.
- CFG-Zero* and Tangential-Damping CFG (fixed 2026-11-04) work on the noise / velocity prediction, as core's
  versions and the papers do. Before, they worked on the denoised image. CFG-Zero*'s zero-init now leaves the
  first step's latent as it is; it used to push that step toward an all-zero latent.
- ZeResFDG (fixed 2026-11-04) leaves the prediction alone at cfg 1. At cfg 1 core skips the negative pass, and the
  node re-weighted the whole prediction's frequencies, changing every cfg-1 (distilled / lightx2v) image.
- MoE Cross-Fade Router (fixed 2026-11-04) with `model_low` connected hands over to the low expert after the fade
  window. It kept the high one. A failing low expert now stops the run with the reason; it used to fall back
  silently to single-model smoothing.

- Wan MoE Cache (fixed 2026-11-05) keeps a separate cache for the positive and the negative pass. When ComfyUI
  ran them as two model calls (low memory, or prompts that cannot be batched), the negative pass got the positive
  pass's cached result, so CFG did nothing. Steps are counted by sigma, so the high-to-low reset lands on
  `switch_step`; the same applies to the guidance nodes' step counter (MoE Cross-Fade Router window).

These guidance fixes are tested on CPU with stand-in predictions. They are **not run with real Wan weights on a
GPU**, and no image-quality A/B was made. `model_low` is called without being loaded through ComfyUI's model
management first, which on a GPU may be the failure that used to be hidden (NOT VERIFIED).

The VAE decode nodes (fixed 2026-11-04) tile in pixels, like core's VAE Decode (Tiled). Before, a 512 px tile was
passed to the VAE as 512 latent cells (4096 px), so a large decode was not tiled at all. The VAE nodes' cache
check no longer fails on every run, so an unchanged re-queue can reuse their result (unit tested; not yet run
live with a real Wan VAE).

### `WanNodeExperiments/LongVideo`
Strategies for generating past the model's native window: segment splicing,
context handoff, and the parameters governing how neighbouring segments blend.

### `WanNodeExperiments/Flow`
Optical-flow temporal consistency (RAFT-based warping, forward-backward
occlusion, FlowVid/FRESCO-style deflicker) for editing *existing* video rather
than generating from scratch.

### `WanNodeExperiments/Audio`, `/LipSync`
Audio separation and lip-sync driving (InfiniteTalk-style). The lip-sync progress bar also follows nodes that sit inside subgraphs.
**InfiniteTalk V2V** passes the whole source clip and the driving audio to the bundled MultiTalk pipeline (fixed
2026-11-03; before, every run stopped with a TypeError).

### `WanNodeExperiments/Video`, `/Postprocess`
Video import/export helpers and post passes.

### `WanNodeExperiments/Director` + `C2C/Wan_Director`
**WanDirector** — a multi-track timeline for authoring a shot: a Scene track
(image/text segments), an Audio track, a Control-Video track, and four
automation lanes (LoRA, Camera, Seed, Pose). All seven tracks render as real
DOM elements. `WanDirectorExtraArgs` carries the advanced quality-stack
options as a single connectable bundle so the main node stays readable.

The Director's preview proxies (2026-10-26) now show true colours from RGB sources such as PNG-coded clips:
- Before, ffmpeg converted those with its default BT.601 matrix into an untagged file, and players show that as
  BT.709, so pure green was 40/255 off.
- Now they convert with BT.709 in limited range, and the file is tagged.
- A YUV source keeps its own colour tags, as before.

`vae_fp32_decode` (fixed 2026-11-05) no longer changes the VAE you wired in. The Director encodes its reference
image with the VAE in fp32 and then puts the VAE back as it was. Before, it converted the shared VAE for the whole
session (every other node using it was changed too) and left its input precision at bf16. Tested with a stand-in
VAE; not yet run with a real Wan VAE.

---

## Two things that will save you time

**1. `cfg` does not control face-expression strength at `cfg = 1.0`.**
Kijai's sampler skips the unconditional pass entirely when `cfg == 1.0`
(`if math.isclose(cfg_scale, 1.0): return noise_pred_cond`), which is the
regime any distilled few-step workflow runs in (e.g. a 4-step lightx2v LoRA).
The lever that works in **every** CFG regime is
`WanVideoAnimateEmbeds.face_strength` (and `pose_strength` for pose), because
it multiplies the adapter residual unconditionally.

**2. Wan-Animate's face conditioning is 100% pixel-driven.**
Landmarks are used *only* to place the face-crop rectangle; the pixels inside
are encoded directly by a LIA-style implicit motion encoder. Editing landmark
data alone changes nothing downstream — an expression edit only reaches the
model if it also rewrites the crop's actual pixels.

---

## Regenerating the reference

`NODE_REFERENCE.md` is generated from the live `NODE_CLASS_MAPPINGS`, so it
always matches the installed code. Regenerate it after changing any node's
`INPUT_TYPES` — the parameter tables are the node tooltips themselves, which
means the tooltip you see in ComfyUI and the docs here are the same string.
