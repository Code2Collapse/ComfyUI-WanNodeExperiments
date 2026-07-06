"""video — import/decode helpers (any video format + EXR) for the Wan pipeline."""
from __future__ import annotations

try:
    from .video_import import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except Exception:  # noqa: BLE001 - never hard-crash the pack on a video import error
    NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = {}, {}

# Server-side media thumbnail/metadata endpoint (any format incl 4K/EXR/DCP).
# Importing it registers GET /wne/media_probe; guarded so a failure here never
# disables the video nodes.
try:
    from . import media_probe  # noqa: F401
except Exception:  # noqa: BLE001
    pass

# Edit-proxy service (Resolve/Premiere pattern): GET /wne/media_proxy
# transcodes ProRes/EXR/MXF/… once to a frame-accurate H.264 proxy so the
# Director player + Video Comparer can PLAY them in real time. Guarded.
try:
    from . import media_proxy  # noqa: F401
except Exception:  # noqa: BLE001
    pass
