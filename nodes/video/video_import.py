"""
video_import.py — WNE_VideoImport: load ANY video format (+ EXR) into the graph.

Decodes to a ComfyUI IMAGE batch [T,H,W,3] 0..1 plus frame-rate / count, so it
can feed WanDirector (control_video / reference) or the flow vid2vid stack as the
*source* video to edit.

Formats:
  * Standard video  — .mp4 .mov .webm .mkv .avi .m4v .gif  via cv2.VideoCapture
                      (ffmpeg backend; ffmpeg is on PATH).
  * OpenEXR         — a single .exr OR a numbered .exr SEQUENCE in a folder, via
                      cv2.imread(IMREAD_UNCHANGED) with OPENCV_IO_ENABLE_OPENEXR
                      (this OpenCV build ships OpenEXR). EXR is linear-HDR, so an
                      optional view transform (sRGB / gamma / none) is applied.
  * Image sequence  — a folder of .png/.jpg/.tiff frames.

Pure cv2 + numpy + torch (all present). No model, CPU-safe.

Author: Code2Collapse. Apache-2.0.
"""
from __future__ import annotations

import os

import numpy as np
import torch

# OpenCV's EXR reader is gated behind this env var; set BEFORE importing cv2 use.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
try:
    import cv2
    _HAVE_CV2 = True
except Exception:  # noqa: BLE001
    _HAVE_CV2 = False

_VIDEO_EXTS = (".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".gif", ".mpg", ".mpeg", ".wmv")
_EXR_EXTS = (".exr",)
_IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")


def _input_dir() -> str:
    try:
        import folder_paths
        return folder_paths.get_input_directory()
    except Exception:  # noqa: BLE001
        return os.getcwd()


def _list_importable() -> list[str]:
    d = _input_dir()
    out = []
    try:
        for f in sorted(os.listdir(d)):
            p = os.path.join(d, f)
            if os.path.isfile(p) and f.lower().endswith(_VIDEO_EXTS + _EXR_EXTS):
                out.append(f)
            elif os.path.isdir(p):
                # expose folders that look like EXR / image sequences
                try:
                    kids = os.listdir(p)
                    if any(k.lower().endswith(_EXR_EXTS + _IMG_EXTS) for k in kids):
                        out.append(f + "/")
                except Exception:  # noqa: BLE001
                    pass
    except Exception:  # noqa: BLE001
        pass
    return out or ["(put a video/EXR in ComfyUI/input)"]


def _resolve(name: str) -> str:
    name = str(name).rstrip("/")
    try:
        import folder_paths
        p = folder_paths.get_annotated_filepath(name)
        if p and os.path.exists(p):
            return p
    except Exception:  # noqa: BLE001
        pass
    return os.path.join(_input_dir(), name)


def _exr_view(rgb_lin: np.ndarray, transform: str, exposure: float) -> np.ndarray:
    """Linear HDR EXR → 0..1 display. rgb_lin float32 (may be >1)."""
    x = rgb_lin.astype(np.float32) * (2.0 ** float(exposure))
    if transform == "linear":
        return np.clip(x, 0.0, 1.0)
    if transform == "gamma2.2":
        return np.clip(np.power(np.clip(x, 0, None), 1.0 / 2.2), 0.0, 1.0)
    # sRGB (default)
    a = 0.055
    lo = x <= 0.0031308
    srgb = np.where(lo, x * 12.92, (1 + a) * np.power(np.clip(x, 0, None), 1 / 2.4) - a)
    return np.clip(srgb, 0.0, 1.0)


def _bgr_to_rgb01(frame: np.ndarray) -> np.ndarray:
    """cv2 BGR(A) uint8/uint16 → RGB float 0..1."""
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if frame.dtype == np.uint16:
        return frame.astype(np.float32) / 65535.0
    if frame.dtype == np.uint8:
        return frame.astype(np.float32) / 255.0
    return frame.astype(np.float32)


def _resize(frames: np.ndarray, tw: int, th: int) -> np.ndarray:
    if tw <= 0 and th <= 0:
        return frames
    h, w = frames.shape[1:3]
    if tw <= 0:
        tw = max(1, round(w * th / h))
    if th <= 0:
        th = max(1, round(h * tw / w))
    out = np.empty((frames.shape[0], th, tw, frames.shape[3]), dtype=frames.dtype)
    for i in range(frames.shape[0]):
        out[i] = cv2.resize(frames[i], (tw, th), interpolation=cv2.INTER_AREA)
    return out


class WNE_VideoImport:
    """Import a video (any format) or EXR sequence as an IMAGE batch."""

    CATEGORY = "WanNodeExperiments/Video"
    FUNCTION = "load"
    RETURN_TYPES = ("IMAGE", "FLOAT", "INT", "INT", "INT")
    RETURN_NAMES = ("images", "frame_rate", "frame_count", "width", "height")
    DESCRIPTION = (
        "Import ANY video (.mp4/.mov/.webm/.mkv/.avi/.gif…) or an OpenEXR file/sequence "
        "(linear-HDR, view-transformed to sRGB) or an image-sequence folder, as an IMAGE "
        "batch — the source for WanDirector control-video / the flow vid2vid editor. "
        "Drop files in ComfyUI/input (or use the upload button)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (_list_importable(), {
                    "tooltip": "A video / .exr / sequence-folder in ComfyUI/input "
                               "(use the '📁 upload video / EXR' button to add one)."}),
                "frame_load_cap": ("INT", {"default": 0, "min": 0, "max": 100000,
                    "tooltip": "Max frames to load (0 = all)."}),
                "skip_first_frames": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "max": 100}),
                "force_rate": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 240.0, "step": 0.01,
                    "tooltip": "Override output frame rate (0 = use the file's native rate / 24 for EXR seq)."}),
                "target_width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8,
                    "tooltip": "Resize width (0 = native; if only one of w/h set, keep aspect)."}),
                "target_height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                "exr_view_transform": (["sRGB", "gamma2.2", "linear"], {"default": "sRGB",
                    "tooltip": "How to map linear-HDR EXR to 0..1 for display/editing."}),
                "exr_exposure": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1,
                    "tooltip": "Exposure (stops) applied before the EXR view transform."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, video, **kw):
        p = _resolve(video)
        try:
            st = os.stat(p)
            return f"{p}-{st.st_mtime}-{st.st_size}-{sorted(kw.items())}"
        except Exception:  # noqa: BLE001
            return f"{p}-{sorted(kw.items())}"

    # ── loaders ──────────────────────────────────────────────────────────────
    def _load_standard(self, path, cap, skip, nth):
        vc = cv2.VideoCapture(path)
        if not vc.isOpened():
            raise RuntimeError(f"could not open video: {path}")
        native_fps = vc.get(cv2.CAP_PROP_FPS) or 0.0
        frames, idx, taken = [], 0, 0
        while True:
            ok, fr = vc.read()
            if not ok:
                break
            if idx >= skip and (idx - skip) % nth == 0:
                frames.append(_bgr_to_rgb01(fr))
                taken += 1
                if cap and taken >= cap:
                    break
            idx += 1
        vc.release()
        return frames, (native_fps if native_fps > 0 else 24.0)

    def _load_exr_or_seq(self, path, cap, skip, nth, transform, exposure):
        # gather file list (single file, or a folder sequence)
        if os.path.isdir(path):
            files = [os.path.join(path, f) for f in sorted(os.listdir(path))
                     if f.lower().endswith(_EXR_EXTS + _IMG_EXTS)]
        else:
            files = [path]
        files = files[skip::nth]
        if cap:
            files = files[:cap]
        frames = []
        for f in files:
            arr = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            if arr is None:
                continue
            if f.lower().endswith(_EXR_EXTS) or (arr.dtype == np.float32):
                if arr.ndim == 2:
                    arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
                rgb = arr[..., :3][..., ::-1]          # BGR->RGB, keep float HDR
                frames.append(_exr_view(np.ascontiguousarray(rgb), transform, exposure))
            else:
                frames.append(_bgr_to_rgb01(arr))
        return frames

    def load(self, video, frame_load_cap, skip_first_frames, select_every_nth,
             force_rate, target_width, target_height, exr_view_transform, exr_exposure):
        if not _HAVE_CV2:
            raise RuntimeError("WNE_VideoImport needs OpenCV (cv2). Install opencv-python.")
        path = _resolve(video)
        if not os.path.exists(path):
            raise FileNotFoundError(f"video not found: {path}")
        cap, skip, nth = int(frame_load_cap), int(skip_first_frames), max(1, int(select_every_nth))

        lower = path.lower()
        is_exr_seq = os.path.isdir(path) or lower.endswith(_EXR_EXTS)
        if is_exr_seq:
            frames = self._load_exr_or_seq(path, cap, skip, nth, exr_view_transform, float(exr_exposure))
            fps = float(force_rate) if force_rate > 0 else 24.0
        else:
            frames, native = self._load_standard(path, cap, skip, nth)
            fps = float(force_rate) if force_rate > 0 else float(native) / nth

        if not frames:
            raise RuntimeError(f"no frames decoded from {path}")
        # unify size (sequences/EXR may vary) to the first frame's, then optional resize
        h0, w0 = frames[0].shape[:2]
        uni = np.stack([f if f.shape[:2] == (h0, w0)
                        else cv2.resize(f, (w0, h0), interpolation=cv2.INTER_AREA)
                        for f in frames], axis=0).astype(np.float32)
        uni = _resize(uni, int(target_width), int(target_height))
        uni = np.clip(uni[..., :3], 0.0, 1.0)
        out = torch.from_numpy(np.ascontiguousarray(uni))
        return (out, float(fps), int(out.shape[0]), int(out.shape[2]), int(out.shape[1]))


NODE_CLASS_MAPPINGS = {"WNE_VideoImport": WNE_VideoImport}
NODE_DISPLAY_NAME_MAPPINGS = {"WNE_VideoImport": "Video Import — any format + EXR (WNE)"}
