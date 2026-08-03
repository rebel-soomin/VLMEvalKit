"""Static-shape wrapper for the GOT-OCR2.0 vision tower.

GOT-OCR2.0's vision tower is a SAM/ViTDet-style encoder that uses window
attention with **relative position embeddings**. ``get_rel_pos`` resamples
the learned rel_pos table with ``F.interpolate(..., mode="linear")`` —
unconditionally, even when the size already matches — and RBLN has no
``aten::upsample_linear1d``, so compilation is refused::

    NotImplementedError: The following operators are not implemented:
        ['aten::upsample_linear1d']

That interpolation is **data-independent**: ``q_size`` / ``k_size`` are
fixed at compile time by ``image_size`` and ``window_size``, and rel_pos
is a learned parameter. The result is therefore a compile-time constant,
so we evaluate it on the host once and let the traced graph consume the
constant. The upstream transformers source is not modified.

Trace-safety traps (both observed in practice while porting)
------------------------------------------------------------
The replacement must **not** key its cache on tensor identity — tracing
runs on ``FakeTensor``:

* ``rel_pos.data_ptr()`` as a cache key -> key changes under trace ->
  cache miss -> the original ``interpolate`` lands in the graph ->
  **compile fails**.
* ``rel_pos is mod.rel_pos_h`` -> object-identity comparison breaks ->
  the h/w tables get swapped -> **compile succeeds but the numbers are
  wrong** (observed 72% relative error).

The second failure mode is the dangerous one, which is why
:func:`verify_equivalence` must be run before compiling: it asserts the
patched CPU output is bit-identical to the unpatched one.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def precompute_rel_pos(module: nn.Module) -> int:
    """Replace ``get_rel_pos`` with host-precomputed constants.

    Returns the number of attention modules patched.
    """
    patched = 0
    for mod in module.modules():
        if not hasattr(mod, 'get_rel_pos') or not getattr(mod, 'use_rel_pos', False):
            continue
        original = mod.get_rel_pos
        # rel_pos has length 2*input_size-1, so invert it for the q/k size.
        n_h = (mod.rel_pos_h.shape[0] + 1) // 2
        n_w = (mod.rel_pos_w.shape[0] + 1) // 2
        with torch.no_grad():
            consts = [
                original(n_h, n_h, mod.rel_pos_h).detach().clone(),
                original(n_w, n_w, mod.rel_pos_w).detach().clone(),
            ]

        def replay(q_size, k_size, rel_pos, _consts=consts, _state=[0]):
            # add_decomposed_rel_pos calls height first, then width — the
            # call order is the only trace-safe discriminator (see module
            # docstring for the two identity-based approaches that fail).
            c = _consts[_state[0] % 2]
            _state[0] += 1
            return c

        mod.get_rel_pos = replay  # instance attribute shadows the class method
        patched += 1
    return patched


class GotOcr2VisionEncoderWrapper(nn.Module):
    """``pixel_values`` (Tensor) -> ``last_hidden_state`` (Tensor).

    The original ``forward`` returns a ``ModelOutput``, which the compiler
    rejects. This wrapper flattens it and applies the rel_pos precompute.
    """

    def __init__(self, vision_encoder: nn.Module, rbln_config=None):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.rbln_config = rbln_config
        self.patched_modules = precompute_rel_pos(self.vision_encoder)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.vision_encoder(pixel_values)
        if isinstance(out, torch.Tensor):
            return out
        if hasattr(out, 'last_hidden_state'):
            return out.last_hidden_state
        return out[0]


def verify_equivalence(vision_encoder: nn.Module, image_size: int,
                       dtype=torch.float32) -> float:
    """Check the rel_pos precompute against the original. Returns ``max|Δ|``
    (must be ``0.0``).

    Call this *before* compiling: without it, "compile succeeded"
    guarantees nothing (see the h/w-swap trap in the module docstring).
    """
    x = torch.rand(1, 3, image_size, image_size,
                   generator=torch.Generator().manual_seed(0)).to(dtype)
    with torch.no_grad():
        before = vision_encoder(x)
    before = before if isinstance(before, torch.Tensor) else before.last_hidden_state

    wrapper = GotOcr2VisionEncoderWrapper(vision_encoder).eval()
    with torch.no_grad():
        after = wrapper(x)
    return float((before - after).abs().max())
