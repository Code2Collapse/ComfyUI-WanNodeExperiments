"""WanT5GemmaLoader — run converted T5Gemma encoder files as a text encoder.

Loads the single-file outputs of tools/convert_t5gemma_encoder.py from
models/text_encoders (safetensors fp32/fp16/bf16/fp8/fp8-scaled, or GGUF
Q8_0) and rebuilds the encoder from the config JSON embedded in the file —
no Hugging Face access needed at load time. Attention backend selectable
(sdpa / eager / flash_attention_2).

HONESTY NOTE (do not remove): pretrained Wan 2.x was conditioned on UMT5-XXL
embeddings (4096-d). T5Gemma encoder states live in a different space and
dimension — this is an R&D encoder for adapters/finetunes/experiments, NOT a
drop-in replacement conditioning for stock Wan checkpoints. The encode node
says so in its tooltip rather than pretending otherwise.
"""

from __future__ import annotations

import json
import logging
import os

import torch

log = logging.getLogger("WanNodeExperiments")

_ATTN = ["auto", "sdpa", "eager", "flash_attention_2"]
_DTYPES = {"auto": None, "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def _encoder_files():
    try:
        import folder_paths
        files = folder_paths.get_filename_list("text_encoders")
    except Exception:  # noqa: BLE001
        files = []
    out = [f for f in files if "t5gemma" in f.lower() and "-encoder-" in f.lower()
           and f.lower().endswith((".safetensors", ".gguf"))]
    return out or ["<run tools/convert_t5gemma_encoder.py first>"]


# ── file readers → (state_dict fp/bf16-ish, config_json, source_repo) ─
def read_safetensors(path: str, compute_dtype: torch.dtype):
    from safetensors import safe_open
    sd, meta = {}, {}
    with safe_open(path, framework="pt") as f:
        meta = f.metadata() or {}
        for k in f.keys():
            sd[k] = f.get_tensor(k)
    cfg_json = meta.get("t5gemma_encoder_config")
    if not cfg_json:
        raise RuntimeError(
            f"{os.path.basename(path)} has no embedded t5gemma_encoder_config metadata — "
            f"it was not produced by tools/convert_t5gemma_encoder.py. Re-convert with "
            f"the current script (the config rides in the file so no HF access is needed).")
    if "scaled_fp8" in sd:  # ComfyUI scaled-fp8 convention → dequantize
        sd.pop("scaled_fp8")
        out = {}
        for k, v in sd.items():
            if k.endswith(".scale_weight"):
                continue
            scale = sd.get(k[:-len(".weight")] + ".scale_weight") if k.endswith(".weight") else None
            if scale is not None:
                out[k] = (v.float() * scale.float()).to(compute_dtype)
            else:
                out[k] = v.to(compute_dtype) if v.is_floating_point() else v
        sd = out
    else:
        sd = {k: (v.to(compute_dtype) if v.is_floating_point() else v) for k, v in sd.items()}
    return sd, cfg_json, meta.get("t5gemma_source_repo", "")


def read_gguf(path: str, compute_dtype: torch.dtype):
    import gguf
    import numpy as np
    r = gguf.GGUFReader(path)

    def kv_str(key):
        f = r.fields.get(key)
        if f is None:
            return None
        return bytes(f.parts[f.data[0]]).decode("utf-8", errors="replace")

    cfg_json = kv_str("t5gemma.encoder_config_json")
    if not cfg_json:
        raise RuntimeError(
            f"{os.path.basename(path)}: missing t5gemma.encoder_config_json GGUF metadata — "
            f"re-convert with tools/convert_t5gemma_encoder.py.")
    sd = {}
    for t in r.tensors:
        arr = gguf.quants.dequantize(t.data, t.tensor_type) \
            if t.tensor_type not in (gguf.GGMLQuantizationType.F32,
                                     gguf.GGMLQuantizationType.F16) else np.asarray(t.data)
        ten = torch.from_numpy(np.ascontiguousarray(arr)).reshape(
            tuple(reversed([int(d) for d in t.shape])))
        sd[t.name] = ten.to(compute_dtype)
    repo = kv_str("t5gemma.source_repo") or ""
    del r
    return sd, cfg_json, repo


def build_encoder(sd: dict, cfg_json: str, attn: str):
    """Rebuild the T5Gemma encoder module and load sd with an EXACT-match
    requirement — a partial load raises instead of silently degrading."""
    from transformers import T5GemmaConfig, T5GemmaEncoderModel
    from transformers.models.t5gemma import T5GemmaModuleConfig
    mod_cfg = T5GemmaModuleConfig(**json.loads(cfg_json))
    if attn != "auto":
        mod_cfg._attn_implementation = attn
    # is_encoder_decoder=False is REQUIRED (T5GemmaEncoderModel refuses
    # encoder-decoder configs), and the decoder config MUST be a separate
    # instance: T5GemmaConfig mutates it (is_decoder=True), and a shared
    # object would silently flip the encoder to CAUSAL attention.
    full = T5GemmaConfig(encoder=mod_cfg,
                         decoder=T5GemmaModuleConfig(**json.loads(cfg_json)),
                         is_encoder_decoder=False)
    if attn != "auto":
        full._attn_implementation = attn
    model = T5GemmaEncoderModel(full)
    # Pick the load target by EXACT key-set match first (no trial loads — a
    # failed strict=False attempt would leave half-loaded weights behind),
    # then strict-load into that live submodule.
    want = set(sd.keys())
    candidates = [model] + ([model.encoder] if hasattr(model, "encoder") else [])
    shapes = []
    for cand in candidates:
        have = set(cand.state_dict().keys())
        if have == want:
            cand.load_state_dict(sd, strict=True)
            model.eval()
            return model
        shapes.append(f"{type(cand).__name__}: missing={sorted(have - want)[:3]} "
                      f"unexpected={sorted(want - have)[:3]}")
    raise RuntimeError(
        "T5Gemma encoder state dict does not match transformers "
        f"{__import__('transformers').__version__} module layout exactly — refusing a "
        f"partial load. {' | '.join(shapes)}")


def _find_tokenizer(model_path: str, source_repo: str):
    from transformers import AutoTokenizer
    base = os.path.basename(model_path)
    short = base.split("-encoder-")[0]
    sib = os.path.join(os.path.dirname(model_path), f"{short}-tokenizer")
    if os.path.isdir(sib):
        return AutoTokenizer.from_pretrained(sib)
    if source_repo:
        try:
            return AutoTokenizer.from_pretrained(source_repo)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Tokenizer folder '{short}-tokenizer' not found next to the model and "
                f"downloading '{source_repo}' failed ({exc}). Re-run the converter (it "
                f"saves the tokenizer beside the encoder) or place the tokenizer there.") from exc
    raise RuntimeError(f"No tokenizer found for {base}: expected sibling folder '{short}-tokenizer'.")


class WanT5GemmaLoader:
    """Load a converted T5Gemma encoder file as a runnable text encoder."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model_file": (_encoder_files(), {
                "tooltip": "Converted single-file T5Gemma encoder from models/text_encoders "
                           "(made by tools/convert_t5gemma_encoder.py)."}),
            "attention_backend": (_ATTN, {
                "default": "auto",
                "tooltip": "sdpa = PyTorch fused attention (default), eager = reference math, "
                           "flash_attention_2 = requires flash-attn package (fails loudly if absent)."}),
            "dtype": (list(_DTYPES), {"default": "auto",
                                      "tooltip": "Compute dtype (auto = bf16 on CUDA, fp32 on CPU)."}),
        }}

    RETURN_TYPES = ("T5GEMMA_ENCODER",)
    RETURN_NAMES = ("t5gemma",)
    FUNCTION = "load"
    CATEGORY = "WanNodeExperiments/TextEncoders"
    DESCRIPTION = "Load a converted T5Gemma encoder (safetensors or GGUF Q8_0) with selectable attention backend."

    def load(self, model_file, attention_backend, dtype):
        if model_file.startswith("<"):
            raise RuntimeError(
                "No converted T5Gemma encoder files in models/text_encoders — run "
                "tools/convert_t5gemma_encoder.py (see its --help).")
        import folder_paths
        path = folder_paths.get_full_path("text_encoders", model_file)
        if not path:
            raise RuntimeError(f"{model_file} vanished from models/text_encoders.")
        compute = _DTYPES[dtype] or (torch.bfloat16 if torch.cuda.is_available() else torch.float32)
        reader = read_gguf if path.lower().endswith(".gguf") else read_safetensors
        sd, cfg_json, repo = reader(path, compute)
        model = build_encoder(sd, cfg_json, attention_backend)
        model.to(compute)
        tokenizer = _find_tokenizer(path, repo)
        log.info("[WanNodeExperiments] T5Gemma encoder loaded: %s (%s, %s)",
                 model_file, attention_backend, compute)
        return ({"model": model, "tokenizer": tokenizer, "dtype": compute,
                 "name": model_file},)

    @classmethod
    def IS_CHANGED(cls, model_file, attention_backend, dtype):
        try:
            import folder_paths
            p = folder_paths.get_full_path("text_encoders", model_file)
            return f"{model_file}-{attention_backend}-{dtype}-{os.path.getmtime(p) if p else 0}"
        except Exception:  # noqa: BLE001
            return f"{model_file}-{attention_backend}-{dtype}"


class WanT5GemmaTextEncode:
    """Encode text with a loaded T5Gemma encoder → CONDITIONING."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "t5gemma": ("T5GEMMA_ENCODER",),
            "text": ("STRING", {"multiline": True, "default": ""}),
            "max_tokens": ("INT", {"default": 512, "min": 8, "max": 8192}),
        }}

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "info")
    FUNCTION = "encode"
    CATEGORY = "WanNodeExperiments/TextEncoders"
    DESCRIPTION = ("T5Gemma text encoding → CONDITIONING (last hidden states + mask + mean-pooled). "
                   "NOTE: stock Wan checkpoints were trained on UMT5-XXL embeddings — T5Gemma "
                   "states are a different space/width, for adapter/finetune R&D, not a drop-in "
                   "swap on pretrained Wan.")

    def encode(self, t5gemma, text, max_tokens):
        model, tok = t5gemma["model"], t5gemma["tokenizer"]
        dev = next(model.parameters()).device
        enc = tok(text or "", return_tensors="pt", padding=True, truncation=True,
                  max_length=int(max_tokens))
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.inference_mode():
            hidden = model(**enc).last_hidden_state
        mask = enc.get("attention_mask")
        pooled = ((hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)) \
            if mask is not None else hidden.mean(1)
        cond = [[hidden.float(), {"pooled_output": pooled.float(),
                                  "attention_mask": mask}]]
        info = (f"{t5gemma['name']}: {hidden.shape[1]} tokens → hidden {tuple(hidden.shape)} "
                f"({t5gemma['dtype']})")
        return (cond, info)


NODE_CLASS_MAPPINGS = {
    "WNE_WanT5GemmaLoader": WanT5GemmaLoader,
    "WNE_WanT5GemmaTextEncode": WanT5GemmaTextEncode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "WNE_WanT5GemmaLoader": "T5Gemma Encoder Loader (WNE)",
    "WNE_WanT5GemmaTextEncode": "T5Gemma Text Encode (WNE)",
}
