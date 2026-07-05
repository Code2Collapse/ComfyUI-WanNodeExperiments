"""
nodes/ — backend package for ComfyUI-WanNodeExperiments.

Aggregates NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS from each phase
module with per-module guards, so a syntax error or a missing dependency in one
module disables only that module's nodes instead of breaking the whole pack.

Phase modules (built incrementally):
    guidance     — APG, ZeResFDG, STG, MoECrossFade, T5AmpAttn, HFLI            [Phase 3]
    samplers     — AnyFlow, WanMoECache, UltraViCo, FreeLOC, FreeInit, MotionMax [Phase 2]
    vae_nodes    — RefDecoder, WanLockedDecode, WaveletColorLock, ...            [Phase 4]
    loaders      — forked Wan 2.2 T2V/I2V/VACE/MoE loaders + GGUF               [Phase 2]
    director     — WanDirector (LTXDirector-2.0 parity)                         [Phase 5]

Modules that do not exist yet are skipped silently; a module that exists but
fails to import is recorded in IMPORT_FAILURES.

Credits: Wan 2.2 wrapper foundations by Kijai (ComfyUI-WanVideoWrapper) and
wuwukaka (ComfyUI-WanAnimatePlus); WanDirector timeline features derived from
WhatDreamsCost (LTX Director), used with the author's permission.
"""

from __future__ import annotations

import importlib
import logging

log = logging.getLogger("WanNodeExperiments")

NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}
IMPORT_FAILURES: dict = {}

# Order matters only for readability; all are independent.
_SUBMODULES = ("loaders", "samplers", "guidance", "vae_nodes", "director", "flow", "video")


def _record_failure(name, exc):
    IMPORT_FAILURES[name] = repr(exc)
    log.error("[WanNodeExperiments] backend module '%s' failed to load: %s", name, exc)


def _load():
    for name in _SUBMODULES:
        try:
            mod = importlib.import_module(f".{name}", __name__)
        except ModuleNotFoundError as exc:
            # The phase module itself isn't built yet → skip silently.
            if exc.name and exc.name.split(".")[-1] == name:
                continue
            _record_failure(name, exc)  # a dependency *inside* the module is missing
            continue
        except Exception as exc:  # noqa: BLE001
            _record_failure(name, exc)
            continue
        NODE_CLASS_MAPPINGS.update(getattr(mod, "NODE_CLASS_MAPPINGS", {}))
        NODE_DISPLAY_NAME_MAPPINGS.update(getattr(mod, "NODE_DISPLAY_NAME_MAPPINGS", {}))


_load()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "IMPORT_FAILURES"]
