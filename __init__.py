"""
ComfyUI-WanNodeExperiments
==========================
A unified, GPU-optimized R&D node pack for Wan 2.2 (MoE DiT): model loaders,
samplers, guidance, VAE-decode experiments, and a timeline Director — all using
standard ComfyUI types so they chain with native nodes and the Kijai / wuwukaka
ecosystems.

License: Apache-2.0 (see LICENSE, NOTICE, MODIFICATIONS.md).

CREDITS
-------
This pack stands on the work of:
  * **Kijai**           — ComfyUI-WanVideoWrapper   (Apache-2.0) — Wan 2.2 wrappers, loaders, samplers, VAE
  * **wuwukaka**        — ComfyUI-WanAnimatePlus     (Apache-2.0) — Wan Animate fork & extensions
  * **WhatDreamsCost**  — LTX Director               (used with author's permission) — timeline editor features
  * **gordonchen19 / Kijai** — Prompt Relay          — per-segment prompt control (Director)
  * R&D method authors credited per-node: APG, ZeResFDG/CADE 2.5, STG, FreeInit,
    AnyFlow, UltraViCo, FreeLOC, RefDecoder, WF-VAE, RIFLEx, TeaCache/MagCache.

Forked code is reimplemented/refined standalone; upstream copyright headers and
attribution are retained in derived files (see MODIFICATIONS.md).
"""

import logging
import os

# Enable OpenCV's OpenEXR reader as early as possible (must be set before cv2 is
# first imported in the process). Best-effort: if another pack already imported
# cv2 first, media_probe / the Director fall back to ffmpeg for EXR/DPX.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

log = logging.getLogger("WanNodeExperiments")
if not log.handlers:
    logging.basicConfig(level=logging.INFO)


def _custom_nodes_dir():
    # this file lives at <custom_nodes>/ComfyUI-WanNodeExperiments/__init__.py
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _upstream_present(*folder_names):
    """Return the first sibling pack folder that is already installed, else None.

    We ship full in-repo COPIES of Kijai's WanVideoWrapper and wuwukaka's
    WanAnimatePlus so this pack is self-contained on a fresh box (e.g. Linux).
    But when the genuine upstream pack is already installed alongside us, loading
    our copy too means importing the entire ~40k-line Wan pipeline a SECOND time
    in the same process — wasteful and, on low-RAM boxes, enough extra heap
    pressure to tip a later builtin's deep pydantic schema recursion into a native
    access-violation at boot. So: load our copy only when upstream is absent.
    """
    cn = _custom_nodes_dir()
    for name in folder_names:
        p = os.path.join(cn, name)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "__init__.py")):
            return name
    return None

__version__ = "0.2.0"
CREDITS = ("ComfyUI-WanNodeExperiments — built on Kijai (ComfyUI-WanVideoWrapper), "
           "wuwukaka (ComfyUI-WanAnimatePlus), and WhatDreamsCost (LTX Director, "
           "used with permission). Apache-2.0.")

# Frontend served at /extensions/ComfyUI-WanNodeExperiments/
WEB_DIRECTORY = "./web"

try:
    from .nodes import (
        NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS,
        IMPORT_FAILURES,
    )
except Exception as exc:  # noqa: BLE001 - never hard-crash ComfyUI on load
    log.exception("[WanNodeExperiments] backend package failed to import: %s", exc)
    NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, IMPORT_FAILURES = {}, {}, {"nodes": repr(exc)}

# ── Additive Wan improvements (our own nodes; safe, standalone) ──────────────
try:
    from .nodes.wan_color_match import (
        NODE_CLASS_MAPPINGS as _cm_map,
        NODE_DISPLAY_NAME_MAPPINGS as _cm_disp,
    )
    NODE_CLASS_MAPPINGS.update(_cm_map)
    NODE_DISPLAY_NAME_MAPPINGS.update(_cm_disp)
except Exception as _cm_exc:  # noqa: BLE001
    log.warning("[WanNodeExperiments] color-match node not registered: %s", _cm_exc)
    try:
        IMPORT_FAILURES["wan_color_match"] = repr(_cm_exc)
    except Exception:  # noqa: BLE001
        pass

# ── Vendored Kijai ComfyUI-WanVideoWrapper (Apache-2.0; LICENSE kept in wanwrapper/) ──
# Direct copy of his full node suite so this pack is self-contained (ships to the
# Linux box without a separate install). Registered under a "WNE_" prefix so it can
# never collide with a separately-installed WanVideoWrapper, and fully guarded so a
# load failure here cannot take down the rest of the pack (per the import-cascade rule).
_wvw_upstream = _upstream_present("ComfyUI-WanVideoWrapper")
if _wvw_upstream:
    log.info("[WanNodeExperiments] upstream %s already installed — skipping our vendored "
             "wanwrapper copy (avoids a redundant second Wan-pipeline load).", _wvw_upstream)
else:
    try:
        from . import wanwrapper as _wvw  # noqa: F401  (triggers its node aggregation)
        _wvw_map = getattr(_wvw, "NODE_CLASS_MAPPINGS", {}) or {}
        _wvw_disp = getattr(_wvw, "NODE_DISPLAY_NAME_MAPPINGS", {}) or {}
        _added = 0
        for _k, _v in _wvw_map.items():
            _pk = "WNE_" + _k
            if _pk not in NODE_CLASS_MAPPINGS:
                NODE_CLASS_MAPPINGS[_pk] = _v
                NODE_DISPLAY_NAME_MAPPINGS[_pk] = str(_wvw_disp.get(_k, _k)) + " (WNE)"
                _added += 1
        log.info("[WanNodeExperiments] vendored WanVideoWrapper: +%d nodes (WNE_ prefix)", _added)
    except Exception as _wvw_exc:  # noqa: BLE001 - never hard-crash on the big copy
        log.warning("[WanNodeExperiments] wanwrapper copy not registered: %s", _wvw_exc)
        try:
            IMPORT_FAILURES["wanwrapper"] = repr(_wvw_exc)
        except Exception:  # noqa: BLE001
            pass

# ── Vendored wuwukaka ComfyUI-WanAnimatePlus (Apache-2.0; LICENSE kept in wananimateplus/) ──
# Direct copy of the Animate-Plus fork so this pack is self-contained. It reuses
# most WanVideoWrapper classes plus unique Animate nodes (EverAnimate, Bernini,
# SCAIL2, v2 samplers). Registered under "WNE_AP_" so it never collides, fully guarded.
# Same load-aware rule: skip if the genuine upstream pack is already installed.
_wap_upstream = _upstream_present("ComfyUI-WanAnimatePlus")
if _wap_upstream:
    log.info("[WanNodeExperiments] upstream %s already installed — skipping our vendored "
             "wananimateplus copy.", _wap_upstream)
else:
    try:
        from . import wananimateplus as _wap  # noqa: F401
        _wap_map = getattr(_wap, "NODE_CLASS_MAPPINGS", {}) or {}
        _wap_disp = getattr(_wap, "NODE_DISPLAY_NAME_MAPPINGS", {}) or {}
        _ap_added = 0
        for _k, _v in _wap_map.items():
            _pk = "WNE_AP_" + _k
            if _pk not in NODE_CLASS_MAPPINGS:
                NODE_CLASS_MAPPINGS[_pk] = _v
                NODE_DISPLAY_NAME_MAPPINGS[_pk] = str(_wap_disp.get(_k, _k)) + " (WNE)"
                _ap_added += 1
        log.info("[WanNodeExperiments] vendored WanAnimatePlus: +%d nodes (WNE_AP_ prefix)", _ap_added)
    except Exception as _wap_exc:  # noqa: BLE001
        log.warning("[WanNodeExperiments] wananimateplus copy not registered: %s", _wap_exc)
        try:
            IMPORT_FAILURES["wananimateplus"] = repr(_wap_exc)
        except Exception:  # noqa: BLE001
            pass

log.info("[WanNodeExperiments] v%s loaded %d nodes (%d module failures). %s",
         __version__, len(NODE_CLASS_MAPPINGS), len(IMPORT_FAILURES), CREDITS)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY",
           "IMPORT_FAILURES", "CREDITS", "__version__"]
