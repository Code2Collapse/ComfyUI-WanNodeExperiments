#!/usr/bin/env python
"""T5Gemma encoder extractor/converter for ComfyUI text_encoders.

Downloads a Google T5Gemma checkpoint from Hugging Face, keeps ONLY the
encoder stack (decoder discarded, with a hard leak assert), and emits
single-file variants:

  safetensors: fp32 (master), fp16, bf16, fp8-e4m3fn, fp8-e4m3fn-scaled
               (scaled = ComfyUI convention: fp8 weights + per-layer
               `<layer>.scale_weight` float32 scalars + `scaled_fp8` marker,
               matching umt5_xxl_fp8_e4m3fn_scaled.safetensors)
  gguf:        Q8_0 (native gguf-py). K-quants (Q5_K_M/Q4_K_M/Q3_K_S/Q2_K)
               are accepted on the CLI but FAIL LOUDLY today: gguf-py has no
               K-quant quantizers (NotImplementedError) and llama.cpp's
               `llama-quantize` rejects this custom encoder-only arch, so
               there is no honest way to produce them yet. No silent fakes.

The fp32 master is derived once and every variant is cast/quantized from it
in memory — the checkpoint is never re-downloaded per variant. The encoder
config JSON (and source repo id) are embedded in the output metadata so the
runtime loader can rebuild the model with zero Hugging Face access.

T5Gemma is NOT covered by city96/ComfyUI-GGUF's T5_SD_MAP/LLAMA_SD_MAP
(Gemma2-style stack: RMSNorm + GQA + RoPE, no T5 relative bias) — the GGUF
here uses its own `t5gemma-encoder` arch with verbatim tensor names, read
back by nodes/wan_t5gemma_loader.py. It is intentionally not claimed to be
llama.cpp-loadable.

Usage:
  python tools/convert_t5gemma_encoder.py --model 9b-2b-ul2 \
      --precisions fp16 bf16 fp8_e4m3fn_scaled --quants Q8_0
  python tools/convert_t5gemma_encoder.py --self-test   # tiny synthetic e2e

Gated repos: accept the license on huggingface.co and export HF_TOKEN.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import torch

# model key -> (HF repo, output shortname). Spec gives exact names for
# 9b-2b-ul2 ("t5gemma-9b2b-encoder-*"); the 2-4b-4b pattern follows the same
# rule for T5Gemma 2 ("t5gemma2-4b4b-encoder-*").
REPO_MAP = {
    "9b-2b-ul2": ("google/t5gemma-9b-2b-ul2", "t5gemma-9b2b"),
    "2-4b-4b": ("google/t5gemma-2-4b-4b", "t5gemma2-4b4b"),
}
PRECISIONS = ["fp32", "fp16", "bf16", "fp8_e4m3fn", "fp8_e4m3fn_scaled"]
QUANTS = ["Q8_0", "Q5_K_M", "Q4_K_M", "Q3_K_S", "Q2_K"]
FP8_MAX = 448.0  # float8_e4m3fn finite max


def log(msg: str):
    print(f"[t5gemma-convert] {msg}", flush=True)


# ── acquisition + encoder isolation ─────────────────────────────────
def fetch_checkpoint(repo_id: str, local: str | None) -> str:
    if local:
        if not os.path.isdir(local):
            raise SystemExit(f"--local-checkpoint {local} is not a directory")
        return local
    from huggingface_hub import snapshot_download
    log(f"downloading {repo_id} (HF_TOKEN honored; gated repos need accepted license)")
    return snapshot_download(repo_id, token=os.environ.get("HF_TOKEN") or None)


def load_encoder(ckpt_dir: str):
    """Return (encoder_module, full_model_keys, encoder_config_json)."""
    from transformers import AutoModel
    model = AutoModel.from_pretrained(ckpt_dir, torch_dtype="auto",
                                      low_cpu_mem_usage=True, trust_remote_code=False)
    if not hasattr(model, "encoder"):
        raise SystemExit(f"{type(model).__name__} has no .encoder — not a T5Gemma-style "
                         f"encoder-decoder checkpoint.")
    full_keys = set(model.state_dict().keys())
    enc = model.encoder
    cfg_json = enc.config.to_json_string()
    return enc, full_keys, cfg_json


def extract_encoder_sd(encoder, full_model_keys: set) -> dict:
    """Encoder-only state dict + hard decoder-leak asserts (correctness check)."""
    sd = {k: v.detach().clone() for k, v in encoder.state_dict().items()}
    assert sd, "extracted encoder state dict is empty"
    leaked = [k for k in sd if k.startswith("decoder.") or ".decoder." in k
              or "cross_attn" in k or "cross_attention" in k]
    assert not leaked, f"decoder/cross-attn keys leaked into encoder extract: {leaked[:5]}"
    dec_keys = {k for k in full_model_keys if k.startswith("decoder.")}
    overlap = {f"encoder.{k}" for k in sd} & dec_keys
    assert not overlap, f"extracted keys collide with decoder keys: {sorted(overlap)[:5]}"
    has_attn = any("q_proj" in k or "self_attn" in k for k in sd)
    has_embed = any("embed" in k for k in sd)
    assert has_attn and has_embed, (
        f"extract does not look like a T5Gemma encoder (attn={has_attn}, "
        f"embed={has_embed}); first keys: {sorted(sd)[:5]}")
    return sd


# ── savers ──────────────────────────────────────────────────────────
def _meta(cfg_json: str, repo_id: str, variant: str) -> dict:
    return {"format": "pt", "wne_arch": "t5gemma-encoder", "wne_variant": variant,
            "t5gemma_encoder_config": cfg_json, "t5gemma_source_repo": repo_id}


def save_safetensors(sd: dict, path: str, meta: dict):
    from safetensors.torch import save_file
    save_file({k: v.contiguous() for k, v in sd.items()}, path, metadata=meta)
    log(f"wrote {path} ({os.path.getsize(path) / 1e9:.2f} GB)")


def cast_sd(master: dict, dtype: torch.dtype) -> dict:
    return {k: (v.to(dtype) if v.is_floating_point() else v) for k, v in master.items()}


def fp8_scaled_sd(master: dict) -> dict:
    """ComfyUI scaled-fp8 convention (see comfy/utils.py convert_old_quants):
    2D linear weights -> fp8 + `<layer>.scale_weight` fp32 scalar; embeddings
    and norms stay fp16; `scaled_fp8` fp8 marker tensor (1 element)."""
    out = {}
    for k, v in master.items():
        if (v.is_floating_point() and v.ndim == 2 and k.endswith(".weight")
                and "embed" not in k and "norm" not in k):
            scale = v.abs().amax().clamp(min=1e-12).float() / FP8_MAX
            out[k] = (v.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
            out[k[:-len(".weight")] + ".scale_weight"] = scale.reshape(())
        elif v.is_floating_point():
            out[k] = v.to(torch.float16)
        else:
            out[k] = v
    out["scaled_fp8"] = torch.zeros((), dtype=torch.float8_e4m3fn)
    return out


def save_gguf(master: dict, path: str, quant: str, cfg_json: str, repo_id: str):
    import numpy as np
    import gguf
    if quant != "Q8_0":
        try:
            gguf.quants.quantize(np.zeros((256, 256), dtype=np.float32),
                                 getattr(gguf.GGMLQuantizationType, quant.split("_M")[0].split("_S")[0]))
        except NotImplementedError:
            raise SystemExit(
                f"{quant}: gguf-py cannot quantize K-quants (NotImplementedError) and "
                f"llama.cpp's llama-quantize rejects this custom `t5gemma-encoder` arch, "
                f"so an honest {quant} file cannot be produced today. Use Q8_0, or rerun "
                f"once gguf-py ships K-quant quantizers.")
    cfg = json.loads(cfg_json)
    w = gguf.GGUFWriter(path, "t5gemma-encoder")
    w.add_name(os.path.basename(path))
    for kv, key in (("t5gemma-encoder.block_count", "num_hidden_layers"),
                    ("t5gemma-encoder.embedding_length", "hidden_size"),
                    ("t5gemma-encoder.feed_forward_length", "intermediate_size"),
                    ("t5gemma-encoder.attention.head_count", "num_attention_heads"),
                    ("t5gemma-encoder.attention.head_count_kv", "num_key_value_heads")):
        if key in cfg:
            w.add_uint32(kv, int(cfg[key]))
    w.add_string("t5gemma.encoder_config_json", cfg_json)
    w.add_string("t5gemma.source_repo", repo_id)
    qtype = gguf.GGMLQuantizationType.Q8_0
    for k, v in master.items():
        arr = v.float().numpy()
        if v.is_floating_point() and arr.ndim >= 2 and arr.shape[-1] % 32 == 0:
            w.add_tensor(k, gguf.quants.quantize(arr, qtype), raw_dtype=qtype)
        else:  # norms/odd shapes stay F32 (Q8_0 needs %32 blocks)
            w.add_tensor(k, arr)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    log(f"wrote {path} ({os.path.getsize(path) / 1e9:.2f} GB)")


def save_tokenizer(ckpt_dir: str, output_dir: str, short: str):
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(ckpt_dir)
        dest = os.path.join(output_dir, f"{short}-tokenizer")
        tok.save_pretrained(dest)
        log(f"tokenizer saved to {dest}")
    except Exception as exc:  # noqa: BLE001 — tokenizer is a convenience copy
        log(f"WARNING: tokenizer not saved ({exc}); the loader will need HF access")


# ── pipeline ────────────────────────────────────────────────────────
def convert(model_key: str, output_dir: str, precisions: list, quants: list,
            local_checkpoint: str | None = None):
    repo_id, short = REPO_MAP[model_key]
    os.makedirs(output_dir, exist_ok=True)
    ckpt = fetch_checkpoint(repo_id, local_checkpoint)
    enc, full_keys, cfg_json = load_encoder(ckpt)
    sd = extract_encoder_sd(enc, full_keys)
    log(f"encoder extracted: {len(sd)} tensors")
    del enc

    master = {k: (v.float() if v.is_floating_point() else v) for k, v in sd.items()}
    del sd
    save_tokenizer(ckpt, output_dir, short)

    def out(tag, ext):
        return os.path.join(output_dir, f"{short}-encoder-{tag}.{ext}")

    written = []
    if "fp32" in precisions:
        save_safetensors(master, out("fp32", "safetensors"), _meta(cfg_json, repo_id, "fp32"))
        written.append(out("fp32", "safetensors"))
    for prec, dtype in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
        if prec in precisions:
            save_safetensors(cast_sd(master, dtype), out(prec, "safetensors"),
                             _meta(cfg_json, repo_id, prec))
            written.append(out(prec, "safetensors"))
    if "fp8_e4m3fn" in precisions:
        sd8 = {k: (v.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
                   if v.is_floating_point() else v) for k, v in master.items()}
        save_safetensors(sd8, out("fp8-e4m3fn", "safetensors"),
                         _meta(cfg_json, repo_id, "fp8_e4m3fn"))
        written.append(out("fp8-e4m3fn", "safetensors"))
    if "fp8_e4m3fn_scaled" in precisions:
        save_safetensors(fp8_scaled_sd(master), out("fp8-e4m3fn-scaled", "safetensors"),
                         _meta(cfg_json, repo_id, "fp8_e4m3fn_scaled"))
        written.append(out("fp8-e4m3fn-scaled", "safetensors"))
    for q in quants:
        save_gguf(master, out(q, "gguf"), q, cfg_json, repo_id)
        written.append(out(q, "gguf"))
    log(f"done: {len(written)} files")
    return written


# ── self-test: tiny synthetic T5Gemma through the whole pipeline ────
def self_test():
    from transformers import T5GemmaConfig, T5GemmaModel
    from transformers.models.t5gemma import T5GemmaModuleConfig

    tiny = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                vocab_size=512)
    cfg = T5GemmaConfig(encoder=T5GemmaModuleConfig(**tiny),
                        decoder=T5GemmaModuleConfig(**tiny))
    model = T5GemmaModel(cfg)
    with tempfile.TemporaryDirectory(prefix="t5g_selftest_") as td:
        full_keys = set(model.state_dict().keys())
        sd = extract_encoder_sd(model.encoder, full_keys)
        print("PASS extract:", len(sd), "tensors, no decoder leak")
        try:  # the leak assert must actually fire
            bad = dict(sd); bad["decoder.layers.0.x"] = torch.zeros(1)
            class _Fake:  # minimal shim carrying the doctored sd
                @staticmethod
                def state_dict(): return bad
            extract_encoder_sd(_Fake, full_keys)
            raise SystemExit("FAIL: decoder-leak assert did not fire")
        except AssertionError:
            print("PASS leak-assert fires on doctored state dict")

        cfg_json = model.encoder.config.to_json_string()
        master = {k: v.float() for k, v in sd.items()}
        from safetensors import safe_open
        for tag, build in (("fp16", lambda: cast_sd(master, torch.float16)),
                           ("fp8-e4m3fn-scaled", lambda: fp8_scaled_sd(master))):
            p = os.path.join(td, f"tiny-encoder-{tag}.safetensors")
            save_safetensors(build(), p, _meta(cfg_json, "self-test", tag))
            with safe_open(p, framework="pt") as f:
                keys = set(f.keys())
                assert json.loads(f.metadata()["t5gemma_encoder_config"])["hidden_size"] == 64
            print(f"PASS {tag}: {len(keys)} keys, config metadata roundtrips")
        assert any(k.endswith(".scale_weight") for k in fp8_scaled_sd(master))
        assert "scaled_fp8" in fp8_scaled_sd(master)
        print("PASS scaled-fp8 convention keys present")

        import gguf
        gp = os.path.join(td, "tiny-encoder-Q8_0.gguf")
        save_gguf(master, gp, "Q8_0", cfg_json, "self-test")
        r = gguf.GGUFReader(gp)
        names = {t.name for t in r.tensors}
        assert names == set(master.keys()), "gguf tensor-name roundtrip mismatch"
        t0 = next(t for t in r.tensors if t.tensor_type == gguf.GGMLQuantizationType.Q8_0)
        deq = gguf.quants.dequantize(t0.data, t0.tensor_type)
        ref = master[t0.name].numpy().reshape(deq.shape)
        err = float(abs(deq - ref).max())
        assert err < 0.05, f"Q8_0 roundtrip error too high: {err}"
        print(f"PASS gguf Q8_0 write/read/dequant roundtrip (max err {err:.4f})")
        # GGUFReader mmaps the file — release it or Windows can't delete the tempdir
        del r, t0, deq
        import gc
        gc.collect()

        try:
            save_gguf(master, os.path.join(td, "x.gguf"), "Q4_K_M", cfg_json, "self-test")
            raise SystemExit("FAIL: K-quant did not fail loudly")
        except SystemExit as e:
            assert "cannot be produced" in str(e)
            print("PASS K-quant fails loudly with explanation")
    print("SELF-TEST PASS")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(REPO_MAP))
    ap.add_argument("--output-dir", default=os.path.join("ComfyUI", "models", "text_encoders"))
    ap.add_argument("--precisions", nargs="*", choices=PRECISIONS, default=["fp16"])
    ap.add_argument("--quants", nargs="*", choices=QUANTS, default=[])
    ap.add_argument("--local-checkpoint", default=None,
                    help="use an already-downloaded HF snapshot dir (skips download)")
    ap.add_argument("--self-test", action="store_true",
                    help="run the tiny synthetic end-to-end pipeline and exit")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.model:
        ap.error("--model is required (or use --self-test)")
    convert(args.model, args.output_dir, args.precisions, args.quants,
            args.local_checkpoint)


if __name__ == "__main__":
    main()
