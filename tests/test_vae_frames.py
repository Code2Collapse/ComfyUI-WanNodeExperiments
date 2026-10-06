"""Wan VAE frame counts: WNE_WanVAEEncode padding and the decode-side trim (L7.41, owner A8, R6).

Core's Wan VAE keeps 4n+1 frames and drops the rest without a message (comfy/ldm/wan/vae.py
"t = 1 + ((t - 1) // 4) * 4"); measured on the real Wan 2.1 VAE: 10, 11, 12 frames in -> 9 out
(docs/evidence/L7.41/vae_precision.json). The fake below follows exactly that rule, so these tests check our
padding and trimming arithmetic, not the VAE. CPU-only, weight-free.
"""

import sys
from pathlib import Path

import pytest
import torch

PACK_ROOT = Path(__file__).resolve().parents[1]
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from nodes.vae_nodes import (  # noqa: E402
    FRAME_MODES,
    SOURCE_FRAMES_KEY,
    WanVAEDecodeTiled,
    WanVAEEncode,
    _as_nhwc,
)

AS_CORE, PAD = FRAME_MODES


class FakeWanVAE:
    """Encode keeps 1 + 4k frames (core's rule); decode returns [B,T,H,W,C] like core's VAE.decode."""

    def __init__(self, temporal=4):
        self.temporal = temporal

    def temporal_compression_decode(self):
        return self.temporal

    def encode(self, pixels):
        n = pixels.shape[0]
        keep = 1 + ((n - 1) // 4) * 4
        frames = pixels[:keep]
        t_lat = 1 + (keep - 1) // 4
        # latent [1, C, T, h, w]; channel 0 remembers each latent frame's first pixel value so decode can expand it
        idx = [0] + [1 + 4 * i for i in range(t_lat - 1)]
        lat = frames[idx, :1, :1, 0].reshape(1, 1, t_lat, 1, 1).expand(1, 16, t_lat, 2, 2).clone()
        return lat

    def decode(self, z):
        t_lat = z.shape[2]
        t_out = 1 + 4 * (t_lat - 1)
        return torch.zeros(1, t_out, 16, 16, 3) + torch.arange(t_out, dtype=torch.float32).view(1, t_out, 1, 1, 1)


class FakeImageVAE:
    def temporal_compression_decode(self):
        raise TypeError("image VAE")

    def encode(self, pixels):
        return torch.zeros(pixels.shape[0], 4, 2, 2)


def clip(n):
    return torch.arange(n, dtype=torch.float32).view(n, 1, 1, 1).expand(n, 16, 16, 3).contiguous() / 100.0


@pytest.mark.parametrize("n", [9, 10, 11, 12, 13])
def test_as_core_keeps_todays_latent(n):
    (out,) = WanVAEEncode().encode(clip(n), FakeWanVAE(), 0, 64, AS_CORE)
    assert SOURCE_FRAMES_KEY not in out
    assert out["samples"].shape[2] == 1 + (n - 1) // 4


@pytest.mark.parametrize("n", [10, 11, 12])
def test_as_core_warns_about_dropped_frames(n, caplog):
    with caplog.at_level("WARNING"):
        WanVAEEncode().encode(clip(n), FakeWanVAE(), 0, 64, AS_CORE)
    assert f"drop the last {n - 9} of {n} frames" in caplog.text


@pytest.mark.parametrize("n", [10, 11, 12, 14, 17])
def test_pad_then_decode_returns_the_source_length(n):
    (lat,) = WanVAEEncode().encode(clip(n), FakeWanVAE(), 0, 64, PAD)
    keep = 1 + ((n - 1) // 4) * 4
    if keep == n:
        assert SOURCE_FRAMES_KEY not in lat
    else:
        assert lat[SOURCE_FRAMES_KEY] == n
        assert lat["samples"].shape[2] == 1 + (keep + 4 - 1) // 4      # padded up to the next 4n+1
    sampled = lat.copy()                                                # what KSampler does with a latent dict
    sampled["samples"] = lat["samples"] + 0.0
    (img,) = WanVAEDecodeTiled().decode(sampled, FakeWanVAE(), 0, 64)
    assert img.ndim == 4 and img.shape[0] == n


def test_pad_is_a_no_op_on_4n_plus_1():
    (lat,) = WanVAEEncode().encode(clip(13), FakeWanVAE(), 0, 64, PAD)
    assert SOURCE_FRAMES_KEY not in lat and lat["samples"].shape[2] == 4


def test_image_vae_is_left_alone():
    (lat,) = WanVAEEncode().encode(clip(3), FakeImageVAE(), 0, 64, PAD)
    assert SOURCE_FRAMES_KEY not in lat and lat["samples"].shape[0] == 3


def test_decode_without_the_key_is_unchanged():
    (img,) = WanVAEDecodeTiled().decode({"samples": torch.zeros(1, 16, 3, 2, 2)}, FakeWanVAE(), 0, 64)
    assert img.shape[0] == 9


def test_video_decode_is_reshaped_to_frames_as_batch():
    assert _as_nhwc(torch.zeros(2, 5, 8, 8, 3)).shape == (10, 8, 8, 3)
    assert _as_nhwc(torch.zeros(4, 8, 8, 3)).shape == (4, 8, 8, 3)


def test_is_changed_accepts_the_new_input():
    a = WanVAEEncode.IS_CHANGED(clip(10), None, 0, 64, frames=AS_CORE)
    b = WanVAEEncode.IS_CHANGED(clip(10), None, 0, 64, frames=PAD)
    assert a != b
    assert WanVAEEncode.IS_CHANGED(clip(10), None, 0, 64) == a      # old prompts without the input
