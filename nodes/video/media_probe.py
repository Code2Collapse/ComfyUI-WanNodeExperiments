"""
media_probe.py — server-side media thumbnail + metadata endpoint.

WHY THIS EXISTS (system design)
-------------------------------
The browser's <video> element can only decode web-delivery codecs
(H.264/VP9/AV1 in mp4/webm). It CANNOT decode the formats a film/VFX pipeline
actually uses — OpenEXR, DPX, ProRes, DNxHD, JPEG2000/MXF (DCP), 4K/8K masters.
So the WanDirector timeline could not draw a filmstrip for those clips (it showed
a "decoding video…" placeholder forever — the reported "add button not working").

The industry-standard fix (Mux/Frame.io/DaVinci proxy workflows) is to decode on
the SERVER, where ffmpeg + OpenCV live, and hand the browser only small
downscaled thumbnails. ffmpeg (full build, on PATH) decodes essentially every
format incl. EXR/ProRes/DNxHD/JPEG2000; OpenCV (OPENCV_IO_ENABLE_OPENEXR) gives
proper scene-linear→sRGB tone-mapping for EXR. 4K is a non-issue because we
downscale to ~80px tall server-side, so the payload is a few KB regardless of
source resolution, and we use *fast seek* (`-ss` before `-i`) so extraction time
is independent of clip length.

Endpoint
--------
GET /wne/media_probe?file=<input-relative>&count=8&height=80&exr_view=sRGB&exposure=0
→ { ok, kind, durationSec, fps, frameCount, width, height, thumbs:[dataURL,…] }

`file` is resolved INSIDE ComfyUI/input only (path-escape guarded).

Author: Code2Collapse. Apache-2.0.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import subprocess
import logging

log = logging.getLogger("WanNodeExperiments")

# OpenEXR reader is gated behind this env var; set BEFORE cv2 is used.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

_VIDEO_EXTS = (".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".gif", ".mpg",
               ".mpeg", ".wmv", ".mxf", ".m2v", ".ts", ".mts", ".prores", ".dnxhd")
_EXR_EXTS = (".exr",)
_IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".dpx",
             ".j2k", ".jp2", ".jpc")


def _input_dir() -> str:
    try:
        import folder_paths
        return folder_paths.get_input_directory()
    except Exception:  # noqa: BLE001
        return os.getcwd()


def _resolve_safe(name: str) -> str | None:
    """Resolve `name` strictly inside the input dir (no path-escape)."""
    base = os.path.realpath(_input_dir())
    p = os.path.realpath(os.path.join(base, str(name).strip().lstrip("/\\")))
    if p == base or p.startswith(base + os.sep):
        return p
    return None


def _ffprobe_meta(path: str) -> dict:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
           "-show_entries", "format=duration", "-of", "json", path]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    data = json.loads(out.stdout or "{}")
    st = (data.get("streams") or [{}])[0]
    fmt = data.get("format") or {}
    w = int(st.get("width") or 0)
    h = int(st.get("height") or 0)

    def _rate(s):
        try:
            n, d = str(s).split("/")
            return float(n) / float(d) if float(d) else 0.0
        except Exception:  # noqa: BLE001
            return 0.0
    fps = _rate(st.get("r_frame_rate")) or _rate(st.get("avg_frame_rate"))
    dur = 0.0
    for v in (st.get("duration"), fmt.get("duration")):
        try:
            dur = float(v)
            if dur > 0:
                break
        except Exception:  # noqa: BLE001
            continue
    nb = st.get("nb_frames")
    frames = int(nb) if (nb and str(nb).isdigit() and int(nb) > 0) else (round(dur * fps) if dur and fps else 0)
    return {"width": w, "height": h, "fps": fps or 0.0, "durationSec": dur, "frameCount": frames}


def _ffmpeg_thumb(path: str, t: float, height: int) -> bytes:
    """Fast-seek to time `t` and grab ONE frame, scaled to `height`px, as JPEG."""
    cmd = ["ffmpeg", "-v", "error", "-noaccurate_seek",
           "-ss", f"{max(0.0, t):.3f}", "-i", path,
           "-frames:v", "1", "-vf", f"scale=-2:{int(height)}:flags=area",
           "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "5", "pipe:1"]
    out = subprocess.run(cmd, capture_output=True, timeout=45)
    return out.stdout or b""


def _exr_view_bytes(path: str, height: int, transform: str, exposure: float) -> bytes:
    """Decode a single EXR/HDR/DPX image via OpenCV and tone-map to a JPEG."""
    import cv2          # type: ignore
    import numpy as np  # type: ignore
    from .video_import import _exr_view
    arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if arr is None:
        return b""
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    rgb = arr[..., :3][..., ::-1]                       # BGR→RGB
    if arr.dtype in (np.float32, np.float16) or path.lower().endswith(_EXR_EXTS):
        disp = _exr_view(np.ascontiguousarray(rgb.astype(np.float32)), transform, exposure)
    else:
        m = 65535.0 if arr.dtype == np.uint16 else 255.0
        disp = np.clip(rgb.astype(np.float32) / m, 0, 1)
    h0, w0 = disp.shape[:2]
    tw = max(1, round(w0 * height / max(1, h0)))
    small = cv2.resize((disp * 255.0).astype(np.uint8), (tw, int(height)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", small[..., ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    return buf.tobytes() if ok else b""


def _b64(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")


def _probe_blocking(path: str, count: int, height: int, transform: str, exposure: float) -> dict:
    lower = path.lower()

    # ── EXR / image sequence folder ──────────────────────────────────────
    if os.path.isdir(path):
        files = [os.path.join(path, f) for f in sorted(os.listdir(path))
                 if f.lower().endswith(_EXR_EXTS + _IMG_EXTS)]
        n = len(files)
        if n == 0:
            return {"ok": False, "error": "folder has no EXR/image frames"}
        idxs = [min(n - 1, int((k + 0.5) * n / count)) for k in range(min(count, n))]
        thumbs = []
        for i in idxs:
            b = b""
            try:
                b = _exr_view_bytes(files[i], height, transform, exposure)
            except Exception:  # noqa: BLE001
                b = b""
            if not b:
                b = _ffmpeg_thumb(files[i], 0.0, height)   # ffmpeg fallback (no OpenCV EXR dep)
            if b:
                thumbs.append(_b64(b))
        w0 = h0 = 0
        try:
            m0 = _ffprobe_meta(files[0])
            w0, h0 = m0.get("width", 0), m0.get("height", 0)
        except Exception:  # noqa: BLE001
            pass
        return {"ok": True, "kind": "exr_seq" if files[0].lower().endswith(_EXR_EXTS) else "image_seq",
                "durationSec": n / 24.0, "fps": 24.0, "frameCount": n,
                "width": int(w0), "height": int(h0), "thumbs": thumbs}

    # ── single EXR / DPX / HDR still ─────────────────────────────────────
    if lower.endswith(_EXR_EXTS) or lower.endswith((".dpx", ".hdr")):
        # cv2 gives the best scene-linear→sRGB tone-map, but its OpenEXR codec is
        # only enabled if OPENCV_IO_ENABLE_OPENEXR was set BEFORE cv2 was first
        # imported anywhere in the process — not guaranteed on a live server. So
        # fall back to ffmpeg (decodes EXR/DPX regardless of OpenCV's build flag).
        b = b""
        try:
            b = _exr_view_bytes(path, height, transform, exposure)
        except Exception:  # noqa: BLE001
            b = b""
        if not b:
            b = _ffmpeg_thumb(path, 0.0, height)
        w0 = h0 = 0
        try:
            meta = _ffprobe_meta(path)
            w0, h0 = meta.get("width", 0), meta.get("height", 0)
        except Exception:  # noqa: BLE001
            pass
        return {"ok": bool(b), "kind": "exr", "durationSec": 0.0, "fps": 0.0,
                "frameCount": 1, "width": int(w0), "height": int(h0),
                "thumbs": [_b64(b)] if b else []}

    # ── everything else: ffmpeg (video / ProRes / DNxHD / MXF / DCP / 4K) ─
    meta = _ffprobe_meta(path)
    dur = meta["durationSec"]
    thumbs = []
    if dur and dur > 0:
        times = [dur * (k + 0.5) / count for k in range(count)]
    else:
        times = [0.0]                                   # single-frame fallback
    for t in times:
        jpeg = _ffmpeg_thumb(path, t, height)
        if jpeg:
            thumbs.append(_b64(jpeg))
    if not thumbs:                                       # last-ditch: cv2 first frame
        try:
            import cv2  # type: ignore
            cap = cv2.VideoCapture(path)
            ok, fr = cap.read(); cap.release()
            if ok:
                hh = int(height); ww = max(1, round(fr.shape[1] * hh / max(1, fr.shape[0])))
                small = cv2.resize(fr, (ww, hh))
                ok2, buf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
                if ok2:
                    thumbs.append(_b64(buf.tobytes()))
        except Exception:  # noqa: BLE001
            pass
    return {"ok": bool(thumbs), "kind": "video", **meta, "thumbs": thumbs}


def _register_route():
    try:
        from server import PromptServer          # type: ignore
        from aiohttp import web
    except Exception as exc:  # noqa: BLE001
        log.warning("[WanNodeExperiments] media_probe route not registered (no server): %s", exc)
        return

    routes = PromptServer.instance.routes

    @routes.get("/wne/media_probe")
    async def media_probe(request):              # noqa: ANN001
        q = request.rel_url.query
        name = q.get("file", "")
        path = _resolve_safe(name)
        if not path or not os.path.exists(path):
            return web.json_response({"ok": False, "error": f"not found: {name}"}, status=404)
        try:
            count = max(1, min(24, int(q.get("count", "8"))))
            height = max(24, min(240, int(q.get("height", "80"))))
            transform = q.get("exr_view", "sRGB")
            exposure = float(q.get("exposure", "0") or 0)
        except Exception:  # noqa: BLE001
            count, height, transform, exposure = 8, 80, "sRGB", 0.0
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, _probe_blocking, path, count, height, transform, exposure)
            return web.json_response(result)
        except Exception as exc:  # noqa: BLE001
            log.warning("[WanNodeExperiments] media_probe failed for %s: %s", name, exc)
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

    log.info("[WanNodeExperiments] media_probe endpoint registered: GET /wne/media_probe")


_register_route()
