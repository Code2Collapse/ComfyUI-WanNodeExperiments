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

### `WanNodeExperiments/LongVideo`
Strategies for generating past the model's native window: segment splicing,
context handoff, and the parameters governing how neighbouring segments blend.

### `WanNodeExperiments/Flow`
Optical-flow temporal consistency (RAFT-based warping, forward-backward
occlusion, FlowVid/FRESCO-style deflicker) for editing *existing* video rather
than generating from scratch.

### `WanNodeExperiments/Audio`, `/LipSync`
Audio separation and lip-sync driving (InfiniteTalk-style).

### `WanNodeExperiments/Video`, `/Postprocess`
Video import/export helpers and post passes.

### `WanNodeExperiments/Director` + `C2C/Wan_Director`
**WanDirector** — a multi-track timeline for authoring a shot: a Scene track
(image/text segments), an Audio track, a Control-Video track, and four
automation lanes (LoRA, Camera, Seed, Pose). All seven tracks render as real
DOM elements. `WanDirectorExtraArgs` carries the advanced quality-stack
options as a single connectable bundle so the main node stays readable.

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
