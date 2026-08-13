"""Custom latent preview for WanNodeExperiments samplers.

WHY THIS EXISTS — WanNodeExperiments' samplers import this module via
``from .latent_preview import prepare_callback`` instead of ComfyUI's
native ``latent_preview``. The previous version was a wholesale copy of
ComfyUI's latent_preview with a VHS-style animated overlay bolted on, and
it had four real defects:

  1. ``print("latent_image shape: ", ...)`` ran on EVERY preview frame
     (console spam + latency).
  2. ``serv = server.PromptServer.instance`` executed at MODULE IMPORT
     time — crashes on Linux and whenever the server isn't ready yet
     (the "[fc3d_preview] PromptServer.instance not ready" warning).
  3. ``Thread(target=...).run()`` ran the preview thread SYNCHRONOUSLY,
     blocking the sampler callback instead of running it async.
  4. ``decode_latent_to_preview_image`` returned ``None`` always, so the
     native in-node preview got NOTHING unless the VHS frontend JS was
     loaded — i.e. the preview was invisible without VideoHelperSuite.

THE FIX — delegate the actual decode to ComfyUI's NATIVE
``latent_preview.prepare_callback``, which the C2C preview guard
(``_c2c_preview_guard.py``) patches with:
  - smart-Auto per-model previewer selection (TAESD for Wan/Hunyuan/LTX
    video, TAESD-for-images when a decoder file is present, else
    Latent2RGB, never blank),
  - a 24fps throttle (decode at most 24x/sec instead of every step —
    the "faster (24fps)" the user asked for).

So this module is now a thin, robust pass-through. The native in-node
preview works WITHOUT VHS installed, on Linux and Windows, for every
WanNodeExperiments sampler that imports this. The optional VHS animated
overlay is preserved as an enhancement that activates only when VHS is
actually present (so users who rely on it don't lose it), and it never
blocks the sampler or crashes on import.
"""
from __future__ import annotations

import logging
import time
from importlib.util import find_spec

log = logging.getLogger("wanexp.preview")

# 24fps for the optional VHS overlay (matches the guard's throttle).
_PREVIEW_FPS = 24


def _server():
    """Lazy PromptServer.instance — NEVER at import time (that was the
    Linux crash). Returns None if the server isn't ready yet."""
    try:
        import server
        inst = getattr(server, "PromptServer", None)
        return inst.instance if inst is not None else None
    except Exception:
        return None


def _vhs_present():
    """True iff ComfyUI-VideoHelperSuite (the VHS animated-preview frontend)
    is installed. Only then is the overlay worth sending."""
    return find_spec("videohelpersuite") is not None


def prepare_callback(model, steps, x0_output_dict=None):
    """Drop-in replacement for latent_preview.prepare_callback.

    Delegates the decode to the (guard-patched) native prepare_callback —
    so the previewer is the smart-Auto per-model one and the 24fps throttle
    the guard installs — then, IF VHS is installed, adds the animated
    scrolling-frame overlay on top. The native in-node preview works
    either way; the overlay is pure upside for VHS users.
    """
    import latent_preview as _native  # guard-patched: smart-Auto + 24fps
    native_cb = _native.prepare_callback(model, steps, x0_output_dict)
    if native_cb is None:
        return None

    serv = _server()
    if not (_vhs_present() and serv is not None):
        # No VHS overlay to add — native in-node preview is enough.
        return native_cb

    # VHS animated overlay: stream individual decoded frames over the
    # websocket as a scrolling preview, throttled to 24fps. We wrap the
    # native callback so the native in-node preview STILL gets its
    # preview_bytes on every (throttled) decode, and additionally emit
    # VHS frames. The overlay never blocks: it runs in a daemon thread.
    import io as _io
    import struct
    import threading
    from PIL import Image

    state = {"last": 0.0, "first": True, "n": 0}

    def _emit_vhs(x0, node_id):
        """Decode one small frame and send it as a VHS preview frame."""
        try:
            import latent_preview as _lp
            previewer = _lp.get_previewer(x0.device, model.model.latent_format)
            if previewer is None:
                return
            res = previewer.decode_latent_to_preview_image("JPEG", x0)
            if not res:
                return
            _fmt, img_or_bytes, _max = res
            # res may be ("JPEG", pil_image, max) from native or
            # ("JPEG", bytes, max) from the guard's safety net.
            if isinstance(img_or_bytes, (bytes, bytearray)):
                buf = _io.BytesIO(img_or_bytes)
                img = Image.open(buf)
            else:
                img = img_or_bytes
            message = _io.BytesIO()
            message.write((1).to_bytes(length=4, byteorder="big") * 2)
            message.write(state["n"].to_bytes(length=4, byteorder="big"))
            try:
                message.write(struct.pack("16p", (node_id or "").encode("ascii")))
            except Exception:
                pass
            img.save(message, format="JPEG", quality=92, compress_level=1)
            serv.send_sync(_lp.server.BinaryEventTypes.PREVIEW_IMAGE
                           if hasattr(_lp, "server") else 1,
                           message.getvalue(), serv.client_id)
            state["n"] = state["n"] + 1
        except Exception as exc:  # noqa: BLE001 — overlay must never break sampling
            log.debug("[wanexp.preview] VHS overlay frame skipped: %s", exc)

    def callback(step, x0, x, total_steps):
        # Native first: updates the in-node preview + progress bar (this
        # is the decode the guard throttles to 24fps).
        native_cb(step, x0, x, total_steps)
        # VHS overlay: independent 24fps throttle so it never spams.
        now = time.monotonic()
        if now - state["last"] < (1.0 / _PREVIEW_FPS) and not state["first"]:
            return
        state["last"] = now
        state["first"] = False
        node_id = getattr(serv, "last_node_id", "") or ""
        threading.Thread(target=_emit_vhs, args=(x0, node_id),
                          daemon=True).start()

    return callback
