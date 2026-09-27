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

# ── Audio Separation + InfiniteTalk V2V lip-sync suite ───────────────────────
try:
    from .nodes.audio_lipsync import (
        NODE_CLASS_MAPPINGS as _al_map,
        NODE_DISPLAY_NAME_MAPPINGS as _al_disp,
    )
    NODE_CLASS_MAPPINGS.update(_al_map)
    NODE_DISPLAY_NAME_MAPPINGS.update(_al_disp)
except Exception as _al_exc:  # noqa: BLE001
    log.warning("[WanNodeExperiments] audio/lip-sync suite not registered: %s", _al_exc)
    try:
        IMPORT_FAILURES["audio_lipsync"] = repr(_al_exc)
    except Exception:  # noqa: BLE001
        pass

# ── T5Gemma encoder loader (files from tools/convert_t5gemma_encoder.py) ──
try:
    from .nodes.wan_t5gemma_loader import (
        NODE_CLASS_MAPPINGS as _t5g_map,
        NODE_DISPLAY_NAME_MAPPINGS as _t5g_disp,
    )
    NODE_CLASS_MAPPINGS.update(_t5g_map)
    NODE_DISPLAY_NAME_MAPPINGS.update(_t5g_disp)
except Exception as _t5g_exc:  # noqa: BLE001
    log.warning("[WanNodeExperiments] T5Gemma loader not registered: %s", _t5g_exc)
    try:
        IMPORT_FAILURES["wan_t5gemma_loader"] = repr(_t5g_exc)
    except Exception:  # noqa: BLE001
        pass

# ── Vendored Kijai ComfyUI-WanVideoWrapper (Apache-2.0; LICENSE kept in wanwrapper/) ──
# Direct copy of his full node suite so this pack is self-contained (ships to the
# Linux box without a separate install). Registered under a "WNE_" prefix so it can
# never collide with a separately-installed WanVideoWrapper, and fully guarded so a
# load failure here cannot take down the rest of the pack (per the import-cascade rule).
def _label_vendored(cls, origin: str):
    """Give a vendored node a description, where upstream left it blank.

    98 of the re-registered nodes carry no DESCRIPTION, so they show a blank
    tooltip in the menu - the one place someone looks before wiring one up.

    What is NOT done here: inventing a description of what each node does.
    That would mean asserting behaviour nobody verified, on someone else's
    code, in a pack where a wrong claim is worse than no claim. Provenance is
    true, useful, and does not pretend to knowledge this pack does not have.
    """
    try:
        if (getattr(cls, "DESCRIPTION", "") or "").strip():
            return
        cls.DESCRIPTION = (
            f"Vendored from {origin}, re-registered here so this pack is "
            "self-contained. Upstream ships no description for this node - see "
            f"the {origin} documentation for what it does. It is prefixed and "
            "namespaced so it cannot collide with a separately installed copy."
        )
    except Exception:  # noqa: BLE001 - a read-only class must not break loading
        pass


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
                _label_vendored(_v, "Kijai's ComfyUI-WanVideoWrapper")
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

# ── wuwukaka's ComfyUI-WanAnimatePlus, ported INTO the pack root (Apache-2.0;
# LICENSE_WANANIMATEPLUS + NOTICE_WANANIMATEPLUS at root, full credit retained).
# Formerly a wananimateplus/ folder (and before that a broken submodule pointer);
# dissolved 2026-07-18: its subpackages (fantasyportrait, multitalk, unianimate,
# wanvideo, …) now live at the pack root, aggregated by animateplus.py. It adds
# unique Animate nodes (EverAnimate, Bernini, SCAIL2, v2 samplers). Registered
# under "WNE_AP_" so it never collides, fully guarded.
# Same load-aware rule: skip if the genuine upstream pack is already installed.
_wap_upstream = _upstream_present("ComfyUI-WanAnimatePlus")
if _wap_upstream:
    log.info("[WanNodeExperiments] upstream %s already installed — skipping our "
             "in-repo Animate-Plus port.", _wap_upstream)
else:
    try:
        from . import animateplus as _wap  # noqa: F401
        _wap_map = getattr(_wap, "NODE_CLASS_MAPPINGS", {}) or {}
        _wap_disp = getattr(_wap, "NODE_DISPLAY_NAME_MAPPINGS", {}) or {}
        _ap_added = 0
        for _k, _v in _wap_map.items():
            _pk = "WNE_AP_" + _k
            if _pk not in NODE_CLASS_MAPPINGS:
                _label_vendored(_v, "wuwukaka's ComfyUI-WanAnimatePlus")
                NODE_CLASS_MAPPINGS[_pk] = _v
                NODE_DISPLAY_NAME_MAPPINGS[_pk] = str(_wap_disp.get(_k, _k)) + " (WNE)"
                _ap_added += 1
        log.info("[WanNodeExperiments] Animate-Plus port: +%d nodes (WNE_AP_ prefix)", _ap_added)
    except Exception as _wap_exc:  # noqa: BLE001
        log.warning("[WanNodeExperiments] Animate-Plus port not registered: %s", _wap_exc)
        try:
            IMPORT_FAILURES["animateplus"] = repr(_wap_exc)
        except Exception:  # noqa: BLE001
            pass

log.info("[WanNodeExperiments] v%s loaded %d nodes (%d module failures). %s",
         __version__, len(NODE_CLASS_MAPPINGS), len(IMPORT_FAILURES), CREDITS)

# ── One menu root for every Code2Collapse pack ─────────────────────────────
# Every node lands under "🐺 C2C/<pack>/<family>" in the Add Node menu and the
# node library (see _c2c_menu.py). Node ids are untouched, so saved workflows
# are unaffected. Guarded: a menu placement must never cost the pack its nodes.
try:
    from ._c2c_menu import rebrand_v1 as _c2c_menu_rebrand

    _c2c_menu_rebrand(
        NODE_CLASS_MAPPINGS, "\U0001F30A Wan Experiments",
        strip=("WanNodeExperiments", "C2C"),
        rename={
            "WanVideoWrapper": "Wan Video Wrapper", "WanAnimatePlus": "Wan Animate Plus",
            "Wan_Director": "Director", "LongVideo": "Long Video", "LipSync": "Lip Sync",
            "TextEncoders": "Text Encoders",
            "ControlNet Preprocessors/Pose Keypoint Postprocess": "Postprocess",
            "KJNodes/masking": "Mask",
        },
    )
except Exception as _c2c_menu_exc:  # noqa: BLE001
    import logging as _c2c_menu_log

    _c2c_menu_log.getLogger(__name__).warning("C2C menu root not applied: %s", _c2c_menu_exc)


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY",
           "IMPORT_FAILURES", "CREDITS", "__version__"]
