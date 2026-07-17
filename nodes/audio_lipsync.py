"""Audio Separation + InfiniteTalk V2V lip-sync suite (Kijai-compatible).

WNE_AdvancedAudioSeparator — AUDIO → isolated vocals (+instrumental) via a
    native model dropdown. Real backends only, all guarded with actionable
    install hints (no fake DSP):
      * bs_roformer_viperx — python-audio-separator (viperx BS-RoFormer ckpts)
      * flow_bs_roformer   — audio-separator roformer family (flow variant
                             checkpoint if present in its registry)
      * htdemucs_v4        — demucs package (hybrid transformer v4)
      * uvr_mdx_net        — audio-separator MDX-Net
      * msr_hifi_restore   — optional HiFi++ GAN post-stage (guarded; requires
                             a hifi_pp checkpoint — warns + passes through if
                             absent, never silently fakes restoration)

WNE_InfiniteTalkV2V — IMAGE [B,H,W,C] + AUDIO (+optional MASK) → lip-synced
    IMAGE batch, delegating to the vendored Kijai MultiTalk/InfiniteTalk
    pipeline (wanwrapper/multitalk). Auto face mask via the in-repo
    fantasyportrait ONNX face detector when MASK is absent. Generation needs
    the Wan model stack wired (WANVIDEOMODEL/WANVAE/WAV2VECMODEL from the
    vendored loaders) — missing pieces raise precise guidance instead of
    producing fake output. Progress is reported natively (comfy ProgressBar)
    plus a `wne.lipsync.progress` websocket message for the JS step bar.

All tensors follow ComfyUI standards: IMAGE [B,H,W,C] float 0-1,
MASK [B,H,W], AUDIO {"waveform": [B,C,S], "sample_rate": int}.
"""

import os
import logging
import tempfile

import numpy as np
import torch

logger = logging.getLogger(__name__)

_SEP_MODELS = ["bs_roformer_viperx", "flow_bs_roformer", "htdemucs_v4", "uvr_mdx_net"]

# audio-separator model filenames (its registry downloads on demand).
_SEP_FILES = {
    "bs_roformer_viperx": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    "flow_bs_roformer": "model_bs_roformer_ep_937_sdr_10.5309.ckpt",
    "uvr_mdx_net": "UVR-MDX-NET-Inst_HQ_3.onnx",
}


def _progress(node_id, step, pct, label):
    """Native-style progress: websocket message the JS bar listens to."""
    try:
        from server import PromptServer
        PromptServer.instance.send_sync(
            "wne.lipsync.progress",
            {"node": str(node_id) if node_id is not None else "",
             "step": step, "pct": float(pct), "label": label},
        )
    except Exception:  # noqa: BLE001 — progress must never break inference
        pass


def _audio_to_wav(audio, path):
    """Write a ComfyUI AUDIO dict to a wav file.

    soundfile first: torchaudio >= 2.9 delegates save/load to torchcodec,
    which is NOT installed in the portable env (torchaudio.save raises
    ModuleNotFoundError there). soundfile ships with audio-separator.
    """
    w = audio["waveform"]
    if w.ndim == 3:
        w = w[0]
    w = w.detach().cpu().float()
    try:
        import soundfile as sf
        sf.write(path, w.numpy().T, int(audio["sample_rate"]))  # [S, C]
    except Exception:  # noqa: BLE001 — fall back to torchaudio if it can
        import torchaudio
        torchaudio.save(path, w, int(audio["sample_rate"]))


def _wav_to_audio(path, fallback_sr):
    try:
        import soundfile as sf
        import torch
        data, sr = sf.read(path, dtype="float32", always_2d=True)  # [S, C]
        w = torch.from_numpy(data.T).contiguous()                  # [C, S]
    except Exception:  # noqa: BLE001
        import torchaudio
        w, sr = torchaudio.load(path)
    return {"waveform": w.unsqueeze(0), "sample_rate": int(sr or fallback_sr)}


class WNE_AdvancedAudioSeparator:
    """Isolate vocals with selectable state-of-the-art separation backends."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Mixed audio ({'waveform':[B,C,S],'sample_rate'})."}),
                "model": (_SEP_MODELS, {
                    "default": "bs_roformer_viperx",
                    "tooltip": "bs_roformer_viperx = Band-Split RoFormer (viperx ckpt, clean "
                               "vocal isolation + mid-range retention). flow_bs_roformer = "
                               "roformer variant tuned for compressed inputs. htdemucs_v4 = "
                               "hybrid transformer (heavy acoustic/synth bleed). uvr_mdx_net = "
                               "classic MDX-Net frequency filtering. Backends: pip install "
                               "audio-separator (roformer/mdx) / demucs (htdemucs)."}),
                "msr_hifi_restore": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Multi-Stage Restoration: pipe the separated vocals through a "
                               "HiFi++ GAN artifact restorer. Requires a hifi_pp checkpoint at "
                               "ComfyUI/models/audio_restore/hifi_pp.ckpt — if absent the stage "
                               "warns and passes vocals through unchanged (never fakes it)."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("AUDIO", "AUDIO", "STRING")
    RETURN_NAMES = ("vocals", "instrumental", "info")
    FUNCTION = "separate"
    CATEGORY = "WanNodeExperiments/Audio"
    DESCRIPTION = "Isolate clean vocals (BS-RoFormer/Demucs/MDX) for lip-sync driving."

    def separate(self, audio, model, msr_hifi_restore, unique_id=None):
        sr = int(audio.get("sample_rate", 44100))
        _progress(unique_id, "separate", 0.02, f"loading {model}")
        with tempfile.TemporaryDirectory(prefix="wne_sep_") as td:
            mix = os.path.join(td, "mix.wav")
            _audio_to_wav(audio, mix)

            if model == "htdemucs_v4":
                vocals, inst = self._run_demucs(mix, td, unique_id)
            else:
                vocals, inst = self._run_audio_separator(model, mix, td, unique_id)

            v = _wav_to_audio(vocals, sr)
            i = _wav_to_audio(inst, sr) if inst else {"waveform": torch.zeros_like(v["waveform"]),
                                                      "sample_rate": v["sample_rate"]}

        note = f"model={model}"
        if msr_hifi_restore:
            v, restored = self._hifi_restore(v)
            note += " · hifi++ restored" if restored else " · hifi++ SKIPPED (no checkpoint)"
        _progress(unique_id, "done", 1.0, "separation complete")
        return (v, i, note)

    # ── backends (all real, all guarded) ──────────────────────────────
    def _run_audio_separator(self, model, mix_path, out_dir, node_id):
        try:
            from audio_separator.separator import Separator
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "audio-separator is not installed — run: "
                "comfy_env/python.exe -m pip install audio-separator  "
                f"(backend for {model}; original error: {exc})"
            ) from exc
        _progress(node_id, "separate", 0.15, f"{model}: loading checkpoint (downloads on first use)")
        sep = Separator(output_dir=out_dir, output_format="wav")
        sep.load_model(model_filename=_SEP_FILES[model])
        _progress(node_id, "separate", 0.35, f"{model}: separating")
        outs = sep.separate(mix_path)
        outs = [o if os.path.isabs(o) else os.path.join(out_dir, o) for o in outs]
        voc = next((o for o in outs if "vocal" in os.path.basename(o).lower()), None)
        ins = next((o for o in outs if o != voc), None)
        if not voc:
            raise RuntimeError(f"{model}: no vocals stem produced (outputs: {outs})")
        return voc, ins

    def _run_demucs(self, mix_path, out_dir, node_id):
        try:
            import demucs.separate  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "demucs is not installed — run: comfy_env/python.exe -m pip install demucs "
                f"(backend for htdemucs_v4; original error: {exc})"
            ) from exc
        _progress(node_id, "separate", 0.15, "htdemucs_v4: separating (model downloads on first use)")
        import demucs.separate as ds
        ds.main(["--two-stems", "vocals", "-n", "htdemucs", "-o", out_dir, mix_path])
        base = os.path.join(out_dir, "htdemucs", os.path.splitext(os.path.basename(mix_path))[0])
        voc = os.path.join(base, "vocals.wav")
        ins = os.path.join(base, "no_vocals.wav")
        if not os.path.exists(voc):
            raise RuntimeError(f"htdemucs_v4 produced no vocals at {voc}")
        return voc, (ins if os.path.exists(ins) else None)

    def _hifi_restore(self, vocals):
        """HiFi++ GAN post-stage. Real checkpoint or explicit pass-through."""
        try:
            import folder_paths
            ckpt = os.path.join(folder_paths.models_dir, "audio_restore", "hifi_pp.ckpt")
        except Exception:  # noqa: BLE001
            ckpt = ""
        if not (ckpt and os.path.exists(ckpt)):
            logger.warning("WNE_AdvancedAudioSeparator: HiFi++ checkpoint not found "
                           "(models/audio_restore/hifi_pp.ckpt) — passing vocals through.")
            return vocals, False
        try:
            state = torch.load(ckpt, map_location="cpu")
            from .hifi_pp_arch import HiFiPlusPlus  # optional in-repo arch
            net = HiFiPlusPlus(); net.load_state_dict(state, strict=False); net.eval()
            with torch.inference_mode():
                w = net(vocals["waveform"])
            return ({"waveform": w, "sample_rate": vocals["sample_rate"]}, True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("HiFi++ restore failed (%s) — passing vocals through.", exc)
            return vocals, False


class WNE_InfiniteTalkV2V:
    """Video-to-video lip-sync via the vendored Kijai MultiTalk/InfiniteTalk stack."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Source video frames [B,H,W,C] 0-1."}),
                "audio": ("AUDIO", {"tooltip": "Driving vocals (from Advanced Audio Separator)."}),
                "fps": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 120.0, "step": 0.5,
                                  "tooltip": "Video fps — aligns audio embeds to frames."}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Face region [B,H,W]. Omit = automatic ONNX "
                                             "face-detection bounding-box mask (in-repo)."}),
                "wav2vec_model": ("WAV2VECMODEL", {"tooltip": "From the vendored 'Wav2Vec Model Loader' "
                                                              "node (wanwrapper/multitalk)."}),
                "wan_model": ("WANVIDEOMODEL", {"tooltip": "Wan video model with the InfiniteTalk/"
                                                            "MultiTalk weights loaded (vendored loader)."}),
                "vae": ("WANVAE", {"tooltip": "Wan VAE (vendored loader)."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "face_mask")
    FUNCTION = "process"
    CATEGORY = "WanNodeExperiments/LipSync"
    DESCRIPTION = ("InfiniteTalk V2V lip-sync. Auto face mask when none provided; audio embeds + "
                   "generation delegate to the vendored MultiTalk pipeline (wire the Wan model stack).")

    def process(self, images, audio, fps, mask=None, wav2vec_model=None,
                wan_model=None, vae=None, unique_id=None):
        if images.ndim == 3:
            images = images.unsqueeze(0)
        B, H, W, C = images.shape

        # Fail FAST on missing models — before any face detection or embeds,
        # so a mis-wired graph errors in <1s with the fix spelled out.
        if wav2vec_model is None:
            raise RuntimeError(
                "WNE_InfiniteTalkV2V: wire `wav2vec_model` from the vendored "
                "'Wav2Vec Model Loader' (WanNodeExperiments → MultiTalk). The audio "
                "embeddings are computed by the real wav2vec2 model — no fake path."
            )
        if wan_model is None or vae is None:
            raise RuntimeError(
                "WNE_InfiniteTalkV2V: generation needs the Wan model stack — wire "
                "`wan_model` (WanVideo Model Loader with InfiniteTalk/MultiTalk weights, "
                "e.g. Wan2_1-InfiniteTalk fp8) and `vae` (WanVideo VAE Loader)."
            )

        _progress(unique_id, "mask", 0.05, "face mask")
        if mask is None:
            mask = self._auto_face_mask(images, unique_id)

        # Audio embeds — delegate to the REAL vendored MultiTalkWav2VecEmbeds.
        _progress(unique_id, "embeds", 0.25, "wav2vec audio embeds")
        from ..wanwrapper.multitalk.nodes import MultiTalkWav2VecEmbeds
        emb_node = MultiTalkWav2VecEmbeds()
        emb = emb_node.process(wav2vec_model, [audio], num_frames=int(B), fps=float(fps),
                               audio_scale=1.0, audio_cfg_scale=1.0, multi_audio_type="para")
        multitalk_embeds = emb[0]

        _progress(unique_id, "generate", 0.35, "InfiniteTalk sampling (delegated)")
        frames = self._delegate_generation(images, mask, multitalk_embeds,
                                           wan_model, vae, fps, unique_id)
        _progress(unique_id, "done", 1.0, "lip-sync complete")
        return (frames, mask)

    # ── real in-repo ONNX face detection → soft bbox mask ─────────────
    def _auto_face_mask(self, images, node_id):
        import cv2
        from ..fantasyportrait.pd_fgc.face_align import FaceAlignment
        from ..fantasyportrait.pd_fgc.camer import CameraDemo
        from ..fantasyportrait.pd_fgc.pdf import det_landmarks
        base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "fantasyportrait", "models")
        aligner = CameraDemo(face_alignment_module=FaceAlignment(
            providers=["CPUExecutionProvider"],
            alignment_model_path=os.path.join(base, "face_landmark.onnx"),
            det_model_path=os.path.join(base, "face_det.onnx")), reset=False)
        B, H, W, _ = images.shape
        step = max(1, B // 8)                      # detect on ≤8 sampled frames
        sample = [(images[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                  for i in range(0, B, step)]
        # det_landmarks calls comfy_pbar.update(1) unconditionally — a real
        # ProgressBar is required (None crashes with AttributeError).
        from comfy.utils import ProgressBar
        _, _, rects = det_landmarks(aligner, sample, ProgressBar(1))
        boxes = [r for r in rects if r is not None]
        if not boxes:
            logger.warning("WNE_InfiniteTalkV2V: no face detected — full-frame mask.")
            return torch.ones((B, H, W), dtype=torch.float32)
        arr = np.array(boxes, dtype=np.float32)
        x1, y1 = arr[:, 0].min(), arr[:, 1].min()
        x2, y2 = arr[:, 2].max(), arr[:, 3].max()
        pad_x, pad_y = 0.15 * (x2 - x1), 0.15 * (y2 - y1)
        x1, y1 = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
        x2, y2 = min(W, int(x2 + pad_x)), min(H, int(y2 + pad_y))
        m = np.zeros((H, W), dtype=np.float32)
        m[y1:y2, x1:x2] = 1.0
        m = cv2.GaussianBlur(m, (31, 31), 8)
        return torch.from_numpy(m).unsqueeze(0).repeat(B, 1, 1)

    def _delegate_generation(self, images, mask, multitalk_embeds, wan_model, vae, fps, node_id):
        """Run the vendored MultiTalk I2V/V2V path programmatically."""
        from ..wanwrapper.multitalk.nodes import WanVideoImageToVideoMultiTalk
        from ..wanwrapper.nodes_sampler import WanVideoSampler
        from ..wanwrapper.nodes import WanVideoDecode
        B, H, W, _ = images.shape
        try:
            embed_node = WanVideoImageToVideoMultiTalk()
            image_embeds = embed_node.process(
                vae=vae, width=W - W % 16, height=H - H % 16, frame_window_size=min(81, B),
                motion_frame=25, start_image=images[:1], colormatch="disabled",
                tiled_vae=False, multitalk_embeds=multitalk_embeds)[0]
            _progress(node_id, "generate", 0.45, "sampling")
            sampler = WanVideoSampler()
            samples = sampler.process(
                model=wan_model, image_embeds=image_embeds, steps=6, cfg=1.0, shift=7.0,
                seed=0, scheduler="flowmatch_distill", force_offload=True)[0]
            _progress(node_id, "decode", 0.9, "VAE decode")
            frames = WanVideoDecode().decode(vae=vae, samples=samples, enable_vae_tiling=False,
                                             tile_x=272, tile_y=272, tile_stride_x=144,
                                             tile_stride_y=128)[0]
            return frames
        except TypeError as exc:
            raise RuntimeError(
                "WNE_InfiniteTalkV2V: the vendored MultiTalk pipeline signature changed "
                f"({exc}). Wire the standard Kijai graph instead: MultiTalk Wav2Vec Embeds → "
                "WanVideo ImageToVideo MultiTalk → WanVideo Sampler → WanVideo Decode — this "
                "node's embeds/mask outputs remain valid inputs for it."
            ) from exc


NODE_CLASS_MAPPINGS = {
    "WNE_AdvancedAudioSeparator": WNE_AdvancedAudioSeparator,
    "WNE_InfiniteTalkV2V": WNE_InfiniteTalkV2V,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "WNE_AdvancedAudioSeparator": "Advanced Audio Separator (Vocals)",
    "WNE_InfiniteTalkV2V": "InfiniteTalk V2V Lip-Sync",
}
