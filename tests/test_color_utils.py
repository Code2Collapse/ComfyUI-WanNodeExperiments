"""color_utils: colour-space round trip and colour matching. CPU-only, weight-free.

These back WNE_WanAnchorColorMatchC2C / WNE_YUVColorLockDecode / WNE_WaveletColorLock.
A colour transform that is subtly wrong still produces a plausible picture — that
is exactly the class of bug that shipped reversed EOTF/OETF in a competitor pack
(CoCoTools issue #14) — so the round trip is asserted numerically, not eyeballed.
"""

import sys
from pathlib import Path

import pytest
import torch

PACK_ROOT = Path(__file__).resolve().parents[1]
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from nodes.color_utils import color_match, rgb_to_ycbcr, ycbcr_to_rgb  # noqa: E402

B, H, W = 2, 8, 8


def _img(seed=0):
    torch.manual_seed(seed)
    return torch.rand(B, H, W, 3)


def test_ycbcr_round_trip_is_lossless():
    # INVARIANT: rgb -> ycbcr -> rgb returns the original. The matrices are exact
    # inverses, so this is tight; a loose tolerance here would hide a transposed
    # matrix, which still looks like a picture.
    x = _img()
    rt = ycbcr_to_rgb(rgb_to_ycbcr(x))
    assert torch.allclose(rt, x, atol=1e-5), (
        f"YCbCr round trip drifted by {(rt - x).abs().max().item():.3e}"
    )


def test_ycbcr_is_not_a_no_op():
    # INVARIANT: the transform must actually transform. Guards the degenerate
    # "identity matrix" fix that would make the round-trip test pass trivially.
    x = _img()
    assert not torch.allclose(rgb_to_ycbcr(x), x, atol=1e-3)


def test_grey_maps_to_zero_chroma():
    # INVARIANT: a neutral grey has no chroma — Cb and Cr sit at the 0.5 offset.
    # Pins the offset, which a round-trip test alone cannot see (it cancels).
    grey = torch.full((1, 4, 4, 3), 0.5)
    ycc = rgb_to_ycbcr(grey)
    assert ycc[..., 1].std().item() == pytest.approx(0.0, abs=1e-6)
    assert ycc[..., 2].mean().item() == pytest.approx(0.5, abs=1e-4)


@pytest.mark.parametrize("method", ["reinhard", "mkl", "hm-mvgd"])
def test_color_match_output_is_a_valid_image(method):
    # INVARIANT: every advertised method returns a finite IMAGE in [0,1] with the
    # target's shape and dtype. Parametrised so a method that only works on one
    # code path cannot hide behind the default.
    target, ref = _img(0), _img(1)
    out = color_match(target, ref, method=method)
    assert out.shape == target.shape
    assert out.dtype == target.dtype
    assert torch.isfinite(out).all(), f"{method} produced non-finite values"
    assert out.min().item() >= 0.0 and out.max().item() <= 1.0


@pytest.mark.parametrize("method", ["reinhard", "mkl", "hm-mvgd"])
def test_color_match_moves_target_toward_reference(method):
    # INVARIANT: matching a DARK target to a BRIGHT reference must raise its mean.
    # This is the actual job; the shape/range test above would pass on a no-op.
    dark = torch.full((1, H, W, 3), 0.2)
    bright = torch.full((1, H, W, 3), 0.8)
    out = color_match(dark, bright, method=method)
    assert out.mean().item() > dark.mean().item() + 0.05, (
        f"{method} did not move the target toward the reference "
        f"({dark.mean().item():.3f} -> {out.mean().item():.3f})"
    )


def test_color_match_with_no_reference_is_a_passthrough():
    # INVARIANT: documented behaviour — a missing reference returns the target
    # untouched rather than raising.
    target = _img()
    assert torch.equal(color_match(target, None), target)


def test_color_match_accepts_unbatched_reference():
    # INVARIANT: a [H,W,C] reference is promoted to a batch (color_utils.py:123-124).
    target = _img()
    out = color_match(target, torch.rand(H, W, 3), method="reinhard")
    assert out.shape == target.shape


def test_unknown_method_falls_back_instead_of_raising():
    # INVARIANT: an unrecognised method degrades to reinhard rather than crashing
    # a render. Pins CURRENT behaviour (color_utils.py:132-133) so a future change
    # to strict validation is a deliberate decision, not a silent one.
    target, ref = _img(0), _img(1)
    assert torch.allclose(
        color_match(target, ref, method="not-a-real-method"),
        color_match(target, ref, method="reinhard"),
    )
