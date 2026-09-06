"""flow_core: warp / occlusion / visualisation invariants. CPU-only, weight-free.

First tests this repo has had. flow_core is the module CustomNodePacks #82
(offset-energy QC) will consume behaviourally via a data contract, so its
guarantees need pinning before anything is built on them.

RAFT itself is deliberately NOT exercised: load_raft() downloads torchvision
weights. Everything below is the pure geometry around it, which is where the
errors that matter actually live — a warp that is off by a sign or half a pixel
still produces a plausible picture.
"""

import sys
from pathlib import Path

import pytest
import torch

PACK_ROOT = Path(__file__).resolve().parents[1]
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from nodes.flow.flow_core import (  # noqa: E402
    _base_grid,
    flow_to_image,
    flow_warp,
    occlusion_mask,
)

N, C, H, W = 1, 3, 16, 24


def _img():
    torch.manual_seed(0)
    return torch.rand(N, C, H, W)


def _flow(dx=0.0, dy=0.0):
    f = torch.zeros(N, 2, H, W)
    f[:, 0] = dx
    f[:, 1] = dy
    return f


def test_zero_flow_is_identity():
    # INVARIANT: warping by zero displacement must return the image unchanged.
    # align_corners=True with the pixel-coord grid makes this EXACT, so any
    # half-pixel error in the normalisation shows up here immediately.
    x = _img()
    out = flow_warp(x, _flow(0.0, 0.0))
    assert torch.allclose(out, x, atol=1e-5), (
        f"identity warp drifted by {(out - x).abs().max().item():.3e} — "
        "grid normalisation is off"
    )


def test_integer_shift_moves_content_the_right_way():
    # INVARIANT: flow_warp SAMPLES x at p+flow, so a +dx flow pulls content from
    # the RIGHT into position p. Pins direction, which a sign flip would invert
    # while leaving every shape assertion happy.
    x = torch.zeros(N, 1, H, W)
    x[:, :, 8, 10] = 1.0
    out = flow_warp(x, _flow(dx=2.0, dy=0.0), mode="nearest")
    assert out[0, 0, 8, 8].item() == pytest.approx(1.0), (
        "content did not arrive at p where flow points to p+2; direction inverted"
    )


def test_base_grid_is_pixel_coordinates():
    # INVARIANT: the grid is (x,y) in PIXELS, not normalised — flow_warp adds a
    # pixel displacement to it, so a normalised grid would silently rescale flow.
    g = _base_grid(1, H, W, torch.device("cpu"), torch.float32)
    assert g.shape == (1, H, W, 2)
    assert g[0, 0, 0].tolist() == [0.0, 0.0]
    assert g[0, H - 1, W - 1].tolist() == [float(W - 1), float(H - 1)]


def test_occlusion_mask_is_all_reliable_for_consistent_flow():
    # INVARIANT: a forward/backward pair that round-trips exactly is fully
    # reliable. fwd=+2, bwd=-2 cancels, so every pixel passes.
    m = occlusion_mask(_flow(dx=2.0), _flow(dx=-2.0))
    assert m.shape == (N, 1, H, W)
    assert m.min().item() == 1.0, "consistent flow was marked occluded"


def test_occlusion_mask_rejects_inconsistent_flow():
    # INVARIANT: the mask must actually reject something — a round trip that does
    # NOT cancel is exactly what FlowVid/FRESCO refuse to propagate. Without this
    # the mask could be a constant 1 and every other test would still pass.
    m = occlusion_mask(_flow(dx=20.0), _flow(dx=20.0))
    assert m.max().item() == 0.0, (
        "grossly inconsistent forward/backward flow was accepted as reliable"
    )


def test_occlusion_mask_is_binary():
    # INVARIANT: documented as {0,1} float; downstream code multiplies by it.
    m = occlusion_mask(_flow(dx=3.0), _flow(dx=-1.0))
    assert set(m.unique().tolist()) <= {0.0, 1.0}


def test_flow_to_image_is_a_valid_image():
    # INVARIANT: visualisation output is a displayable IMAGE — finite and in [0,1].
    torch.manual_seed(0)
    vis = flow_to_image(torch.randn(N, 2, H, W) * 5.0)
    assert torch.isfinite(vis).all()
    assert vis.min().item() >= 0.0 and vis.max().item() <= 1.0


def test_flow_warp_accepts_mismatched_flow_resolution():
    # INVARIANT: the docstring promises flow is resized to match x, so latents can
    # be warped by image-resolution flow. Pins the documented behaviour.
    x = _img()
    small = torch.zeros(N, 2, H // 2, W // 2)
    out = flow_warp(x, small)
    assert out.shape == x.shape


def test_flow_warp_handles_hostile_values():
    # INVARIANT: NaN/Inf in the flow must not crash the warp; padding_mode='border'
    # keeps sampling in range.
    x = _img()
    f = _flow(1.0, 1.0)
    f[0, 0, 0, 0] = float("inf")
    out = flow_warp(x, f)
    assert out.shape == x.shape
    assert torch.isfinite(out[:, :, 1:, 1:]).all(), "hostile flow contaminated the whole frame"
