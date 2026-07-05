"""
Wan Director Inspector — consumes the Director's JSON status outputs.

``WanDirectorC2C`` emits four JSON STRING outputs (``info``,
``tracks_program``, ``guide_data``, ``quality_recipe``) describing the
resolved plan: backend, variant, latent shape, segment counts, the active
quality stack, prompt-relay status and per-track programs. The quality
features themselves are applied INLINE on the Director's MODEL/CONDITIONING
outputs, so these strings are descriptors — previously they were dead ends
with no consumer.

This node terminates those ends: it parses the JSON and surfaces a clean
human-readable report (rendered in the node, like a Show-Text terminator)
plus a few typed scalars (segment count, frame rate, relay flag, active
feature list) so they can drive branches or be displayed. It does NOT
re-apply any quality feature — that would double-apply what the Director
already patched onto the model.

Author: Code2Collapse. Licensed under the Apache License, Version 2.0.
"""
from __future__ import annotations

import json

CATEGORY = "C2C/Wan_Director"


def _safe_load(s):
    """Parse a JSON string, tolerating empty/blank/None and bad JSON."""
    if not s or not str(s).strip():
        return None
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        return None


def _fmt_features(recipe: dict) -> list:
    """Return ['pag(scale=2.0)', 'nag(scale=11.0)', ...] for enabled features."""
    out = []
    if not isinstance(recipe, dict):
        return out
    for key, val in recipe.items():
        if not isinstance(val, dict) or not val.get("enabled"):
            # `cache` has a "type" instead of "enabled"
            if key == "cache" and isinstance(val, dict) and val.get("type") not in (None, "", "none"):
                out.append(f"cache(type={val.get('type')})")
            continue
        scale = val.get("scale")
        if scale is not None:
            out.append(f"{key}(scale={scale})")
        else:
            out.append(key)
    return out


class WanDirectorInspector:
    """Parse + display the Wan Director's JSON status outputs.

    Wire the Director's ``info`` (and optionally ``quality_recipe`` /
    ``tracks_program`` / ``guide_data``) here to read a plain-English plan
    summary in the node and pull out typed values for downstream logic.
    """

    CATEGORY = CATEGORY
    FUNCTION = "inspect"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Reads the Wan Director's JSON status outputs (info / quality_recipe "
        "/ tracks_program / guide_data) and renders a human-readable plan "
        "summary plus typed scalars (segment count, frame rate, prompt-relay "
        "flag, active-feature list). Display/inspection only — it does not "
        "re-apply quality features (the Director already patched the model)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "info": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "quality_recipe": ("STRING", {"forceInput": True}),
                "tracks_program": ("STRING", {"forceInput": True}),
                "guide_data": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "FLOAT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("summary", "segment_count", "frame_rate",
                    "prompt_relay_applied", "active_features")
    OUTPUT_TOOLTIPS = (
        "Plain-English multi-line plan summary (feed to a Show-Text node).",
        "Number of text/prompt segments on the timeline.",
        "Resolved frame rate.",
        "True if PromptRelay was actually applied (not just enabled).",
        "Comma-separated list of active quality features, e.g. 'pag(scale=2.0), nag(scale=11.0)'.",
    )

    @classmethod
    def IS_CHANGED(cls, info, quality_recipe="", tracks_program="", guide_data="", **kw):
        import hashlib
        blob = "\x1f".join([str(info), str(quality_recipe),
                            str(tracks_program), str(guide_data)])
        return hashlib.md5(blob.encode("utf-8", "replace")).hexdigest()

    def inspect(self, info, quality_recipe="", tracks_program="", guide_data=""):
        meta = _safe_load(info) or {}
        recipe = _safe_load(quality_recipe)
        if recipe is None:
            recipe = meta.get("quality_recipe") if isinstance(meta, dict) else None
        tracks = _safe_load(tracks_program) or {}
        guides = _safe_load(guide_data) or {}

        if not isinstance(meta, dict) or not meta:
            summary = ("Wan Director Inspector: could not parse `info` JSON "
                       "(is the Director's `info` output connected?).")
            return {"ui": {"text": [summary]},
                    "result": (summary, 0, 0.0, False, "")}

        # ── typed scalars ──────────────────────────────────────────────
        seg_count = int(meta.get("n_text", 0) or 0)
        fps = float(meta.get("fps", 0.0) or 0.0)
        relay = meta.get("prompt_relay", {}) if isinstance(meta.get("prompt_relay"), dict) else {}
        relay_applied = bool(relay.get("applied", False))
        feats = _fmt_features(recipe) if isinstance(recipe, dict) else list(meta.get("quality_features", []))
        feats_str = ", ".join(feats)

        # ── human-readable report ──────────────────────────────────────
        out_size = meta.get("out_size", ["?", "?"])
        lat = meta.get("latent_shape", [])
        lines = [
            "── Wan Director plan ──────────────────────────────",
            f"backend         : {meta.get('backend', '?')}",
            f"variant         : {meta.get('variant', '?')}  ({meta.get('label', '')})",
            f"output size     : {out_size[0]} × {out_size[1]}  @ {fps:g} fps",
            f"frames          : {meta.get('frames', '?')}   latent {lat}",
            f"vae_encoded     : {meta.get('vae_encoded', False)}",
            f"segments        : {seg_count} text / {meta.get('n_neg', 0)} neg / {meta.get('n_image', 0)} image",
            f"audio sr        : {meta.get('audio_sr', '?')} Hz",
        ]

        # track programs (lora / camera / seed / pose / everanimate)
        track_counts = []
        for key in ("lora", "camera", "seed", "pose"):
            n = len(tracks.get(key, []) or []) if isinstance(tracks, dict) else 0
            if n:
                track_counts.append(f"{key}×{n}")
        ea = tracks.get("everanimate") if isinstance(tracks, dict) else None
        if isinstance(ea, dict) and ea.get("active"):
            track_counts.append("everanimate")
        if track_counts:
            lines.append(f"track programs  : {', '.join(track_counts)}")

        if isinstance(guides, dict) and guides.get("per_segment"):
            lines.append(f"guide strengths : {guides.get('per_segment')}")

        lines.append(
            f"prompt relay    : {'APPLIED' if relay_applied else 'off'}"
            + (f"  ({relay.get('note')})" if relay.get("note") else "")
        )
        lines.append(f"quality stack   : {feats_str or '(none active)'}")

        warns = meta.get("warnings", []) or []
        twarns = meta.get("track_warnings", []) or []
        if warns or twarns:
            lines.append("── warnings ───────────────────────────────────────")
            for w in (list(warns) + list(twarns))[:20]:
                lines.append(f"  • {w}")

        summary = "\n".join(lines)
        return {"ui": {"text": [summary]},
                "result": (summary, seg_count, fps, relay_applied, feats_str)}


NODE_CLASS_MAPPINGS = {"WanDirectorInspector": WanDirectorInspector}
NODE_DISPLAY_NAME_MAPPINGS = {"WanDirectorInspector": "Wan Director — Inspector"}
