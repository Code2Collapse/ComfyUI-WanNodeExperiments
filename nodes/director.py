"""
director.py — WanDirector (Phase 5), shifted into ComfyUI-WanNodeExperiments.

``WanDirectorC2C`` is a single-node visual multi-shot timeline for Wan video
models (the C2C/MEC implementation, authored by Code2Collapse / Halohues
Studios). It is relocated here intact along with its dependency closure
(``prompt_relay/``, ``wan_director/`` + ``features/``, ``asymflow_sampler``).

The companion frontend lives in ``web/js/wan_director_*.js`` (timeline editor,
player, compact, variant gate) with shared helpers ``_c2c_theme.js`` /
``_c2c_report.js``.

Roadmap (Phase 5b+): bring the timeline editor to LTX Director 2.0 feature
parity — timeline editing, arbitrary keyframes, custom audio + inpainting,
video import/split/extend, IC-LoRA track, timeline save/load, UI overhaul, QoL
suite. The LTX-derived features are used with the express permission of the LTX
Director author (WhatDreamsCost); see NOTICE / MODIFICATIONS.md.

Credits: WanDirector inspired by WhatDreamsCost's LTX Director and Kijai's
Prompt Relay; Wan backends interop with Kijai (ComfyUI-WanVideoWrapper) and
wuwukaka (ComfyUI-WanAnimatePlus).
"""

from __future__ import annotations

from .wan_director import (  # noqa: F401
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
