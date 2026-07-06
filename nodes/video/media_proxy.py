"""
media_proxy.py — server-side edit-proxy generation (Resolve/Premiere pattern).

WHY THIS EXISTS (system design)
-------------------------------
The browser can only PLAY web codecs (H.264/VP9/AV1). media_probe.py already
gives the timeline thumbnails + metadata for ProRes/EXR/MXF/DNxHD sources,
but real-time PLAYBACK of those sources in the WanDirector player and the
Video Comparer needs an actual playable stream. The industry answer is the
proxy workflow: transcode once, in the background, to a small frame-accurate
H.264 mp4; every UI player uses the proxy; the RENDER path keeps reading the
untouched original. Source media is never modified.

Endpoint
--------
GET /wne/media_proxy?file=<input-relative>&height=720
→ { ok, status: "ready"|"building"|"error", url?, progress?, error?, key }

- `file` resolves strictly inside ComfyUI/input (same guard as media_probe).
- Proxies land in ComfyUI temp under `wne_proxies/` and are served through
  ComfyUI's own /view endpoint (type=temp), so no new static route is needed.
- Cache key = sha1(source realpath + mtime + size + height): editing or
  replacing the source automatically invalidates its proxy.
- Frame-accuracy: fps is NOT changed; only spatial scale (and codec) change,
  so frame N of the proxy == frame N of the original.
- Disk ceiling: WNE_PROXY_CACHE_MB env var (default 2048 MB). Oldest proxies
  are evicted before a new job when over the ceiling.

Author: Code2Collapse. Apache-2.0.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import threading

log = logging.getLogger("WanNodeExperiments")

_SUBDIR = "wne_proxies"
_CACHE_MB = float(os.environ.get("WNE_PROXY_CACHE_MB", "2048"))

# Extensions the browser can (almost always) play natively — no proxy needed
# unless the container hides a non-web codec; the client may still request one
# if native playback errors out.
WEB_SAFE_EXTS = (".mp4", ".webm", ".m4v", ".ogv")

_jobs: dict = {}
_jobs_lock = threading.Lock()


def _input_dir() -> str:
    try:
        import folder_paths
        return folder_paths.get_input_directory()
    except Exception:  # noqa: BLE001
        return os.getcwd()


def _proxy_dir() -> str:
    try:
        import folder_paths
        base = folder_paths.get_temp_directory()
    except Exception:  # noqa: BLE001
        base = os.path.join(os.getcwd(), "temp")
    d = os.path.join(base, _SUBDIR)
    os.makedirs(d, exist_ok=True)
    return d


def _resolve_safe(name: str) -> str | None:
    base = os.path.realpath(_input_dir())
    p = os.path.realpath(os.path.join(base, str(name).strip().lstrip("/\\")))
    if p == base or p.startswith(base + os.sep):
        return p
    return None


def _key_for(path: str, height: int) -> str:
    st = os.stat(path)
    raw = f"{os.path.realpath(path)}|{st.st_mtime_ns}|{st.st_size}|{height}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]


def _evict_over_ceiling(incoming_hint_mb: float = 0.0) -> None:
    """Keep the proxy dir under WNE_PROXY_CACHE_MB by deleting oldest files."""
    d = _proxy_dir()
    files = []
    total = 0.0
    for n in os.listdir(d):
        p = os.path.join(d, n)
        try:
            st = os.stat(p)
        except OSError:
            continue
        files.append((st.st_mtime, st.st_size, p))
        total += st.st_size / 1e6
    budget = max(64.0, _CACHE_MB) - incoming_hint_mb
    if total <= budget:
        return
    for _, size, p in sorted(files):
        try:
            os.remove(p)
            total -= size / 1e6
            log.info("[WanNodeExperiments] proxy cache evicted %s", os.path.basename(p))
        except OSError:
            pass
        if total <= budget:
            break


def _duration_sec(path: str) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", path],
            capture_output=True, text=True, timeout=30)
        return float((json.loads(out.stdout or "{}").get("format") or {}).get("duration") or 0.0)
    except Exception:  # noqa: BLE001
        return 0.0


def _build_proxy(src: str, dst: str, height: int, key: str) -> None:
    """ffmpeg transcode in a worker thread; progress goes into _jobs[key]."""
    dur = _duration_sec(src)
    tmp = dst + ".part.mp4"
    # -vf scale only DOWN (never upscale); -2 keeps aspect + even dims. fps is
    # left untouched so the proxy stays frame-accurate to the original.
    vf = f"scale=-2:'min({int(height)},ih)':flags=bicubic,format=yuv420p"
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-i", src, "-vf", vf,
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-c:a", "aac", "-b:a", "128k",
           "-movflags", "+faststart",
           "-progress", "pipe:1", tmp]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, bufsize=1)
        for line in proc.stdout:  # type: ignore[union-attr]
            line = line.strip()
            if line.startswith("out_time_ms=") and dur > 0:
                try:
                    done = int(line.split("=", 1)[1]) / 1e6
                    with _jobs_lock:
                        _jobs[key]["progress"] = max(0.0, min(0.99, done / dur))
                except ValueError:
                    pass
        proc.wait(timeout=3600)
        if proc.returncode != 0:
            err = (proc.stderr.read() if proc.stderr else "")[-400:]
            raise RuntimeError(f"ffmpeg exited {proc.returncode}: {err}")
        os.replace(tmp, dst)
        with _jobs_lock:
            _jobs[key].update(status="ready", progress=1.0)
        log.info("[WanNodeExperiments] proxy ready: %s (%.1f MB)",
                 os.path.basename(dst), os.path.getsize(dst) / 1e6)
    except Exception as exc:  # noqa: BLE001
        try:
            os.remove(tmp)
        except OSError:
            pass
        with _jobs_lock:
            _jobs[key].update(status="error", error=str(exc)[:300])
        log.warning("[WanNodeExperiments] proxy build failed for %s: %s", src, exc)


def _view_url(fname: str) -> str:
    return f"/view?filename={fname}&subfolder={_SUBDIR}&type=temp"


def request_proxy(name: str, height: int = 720) -> dict:
    """Core logic (sync, cheap): return status for the given source file."""
    path = _resolve_safe(name)
    if not path or not os.path.isfile(path):
        return {"ok": False, "status": "error", "error": f"not found: {name}"}
    height = max(144, min(1080, int(height)))
    key = _key_for(path, height)
    fname = f"{key}.mp4"
    dst = os.path.join(_proxy_dir(), fname)
    if os.path.isfile(dst):
        return {"ok": True, "status": "ready", "url": _view_url(fname),
                "key": key, "progress": 1.0}
    with _jobs_lock:
        job = _jobs.get(key)
        if job and job.get("status") == "building":
            return {"ok": True, "status": "building", "key": key,
                    "progress": job.get("progress", 0.0)}
        if job and job.get("status") == "error":
            # allow retry: fall through and restart
            pass
        _jobs[key] = {"status": "building", "progress": 0.0}
    _evict_over_ceiling(incoming_hint_mb=200.0)
    threading.Thread(target=_build_proxy, args=(path, dst, height, key),
                     daemon=True, name=f"wne-proxy-{key[:8]}").start()
    return {"ok": True, "status": "building", "key": key, "progress": 0.0}


def _register_route() -> None:
    try:
        from server import PromptServer  # type: ignore
        from aiohttp import web
    except Exception as exc:  # noqa: BLE001
        log.warning("[WanNodeExperiments] media_proxy route not registered (no server): %s", exc)
        return

    routes = PromptServer.instance.routes

    @routes.get("/wne/media_proxy")
    async def media_proxy(request):  # noqa: ANN001
        q = request.rel_url.query
        name = q.get("file", "")
        try:
            height = int(q.get("height", "720"))
        except ValueError:
            height = 720
        try:
            result = request_proxy(name, height)
            status = 200 if result.get("ok") else 404
            return web.json_response(result, status=status)
        except Exception as exc:  # noqa: BLE001
            log.warning("[WanNodeExperiments] media_proxy failed for %s: %s", name, exc)
            return web.json_response({"ok": False, "status": "error",
                                      "error": str(exc)}, status=500)

    log.info("[WanNodeExperiments] media_proxy endpoint registered: GET /wne/media_proxy")


_register_route()
