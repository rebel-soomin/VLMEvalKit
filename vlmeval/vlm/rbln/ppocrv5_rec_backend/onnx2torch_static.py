"""Patches that pin an ``onnx2torch``-converted graph to a **static** shape.

Why this is needed
------------------
The recognition model cannot go through ``compile_from_onnx`` (see
:mod:`.compile_backend`), so it is routed ONNX -> torch ->
``compile_from_torch``. But several ``onnx2torch`` converters treat
**constant** parameters as runtime tensors, which puts data-dependent
operations into the trace and makes the RBLN compile fail — with an error
message that points at an unrelated op.

``OnnxReshape``
    Branches on ``torch.any(shape == 0)`` at run time. ``torch.export``
    fails, falls back to ``jit.trace``, and the compile dies with
    ``TVMError``.
``OnnxSlice``
    Builds ``slice(start, end, step)`` out of **tensor elements**, giving
    ``TypeError: int() argument ... not 'TupleGetItem'``.
``OnnxSqueezeDynamicAxes``
    Iterates ``torch.sort(axes).values``, which introduces ``unbind`` +
    ``item()`` and hence an unbacked symint (``u0``).

Those three parameters really are constants (ONNX initializers), so one CPU
warm-up caches them as Python ints and every later call uses the cached
value — which makes the graph static.

⚠️ The cache key must **not** be tensor identity (``is``) or ``data_ptr()``:
tracing runs on FakeTensors and both break. The cache lives on the **module
instance** instead — ``onnx2torch`` creates one module per ONNX node, so an
instance-level cache is exactly per-node.

Separately, one **compiler bug workaround** rides along here (it is not
staticisation):

``OnnxGlobalAveragePool*``
    ``mean(dim=(2,3))`` right after an ``Add`` is divided by ``W`` only, not
    ``H*W``. ⚠️ **The compile succeeds** and the output is exactly ``H``
    times too large — silently wrong.

Usage
-----
::

    net = convert(onnx.load(path)).eval()
    with torch.no_grad():
        ref = net(*xs)                 # reference, before patching
    counts = make_static(net)
    assert_equivalent(net, xs, ref)    # raises unless bit-identical
"""

from __future__ import annotations

import torch

__all__ = ['make_static', 'assert_equivalent', 'VERIFIED_ONNX2TORCH']


def _patch_reshape(mod):
    def fwd(input_tensor, shape, _s=mod):
        if not hasattr(_s, '_frozen_shape'):
            v = [int(a) for a in shape.tolist()]
            _s._frozen_shape = [input_tensor.shape[i] if a == 0 else a
                                for i, a in enumerate(v)]
        return torch.reshape(input_tensor, _s._frozen_shape)

    mod.forward = fwd


def _patch_slice(mod):
    def fwd(input_tensor, starts, ends, axes=None, steps=None, _s=mod):
        if not hasattr(_s, '_frozen_slice'):
            st = [int(v) for v in starts.tolist()]
            en = [int(v) for v in ends.tolist()]
            ax = [int(v) for v in axes.tolist()] if axes is not None else list(range(len(st)))
            sp = [int(v) for v in steps.tolist()] if steps is not None else [1] * len(st)
            _s._frozen_slice = (st, en, ax, sp)
        st, en, ax, sp = _s._frozen_slice
        ndim = input_tensor.dim()
        idx = [slice(None)] * ndim
        flip = []
        for s0, e0, a0, p0 in zip(st, en, ax, sp):
            a0 = a0 if a0 >= 0 else ndim + a0
            if p0 < 0:                      # negative step -> flip, then positive step
                flip.append(a0)
                s0, e0, p0 = -s0 - 1, -e0 - 1, -p0
            idx[a0] = slice(s0, e0, p0)
        x = torch.flip(input_tensor, dims=flip) if flip else input_tensor
        return x[tuple(idx)]

    mod.forward = fwd


def _patch_squeeze(mod):
    """``OnnxSqueezeDynamicAxes`` iterates ``torch.sort(axes).values`` and passes
    ``dim`` as a **tensor**, which creates ``unbind`` + ``item()`` and hence an
    unbacked symint (``u0``). ``axes`` is constant, so pin it to Python ints.
    """
    def fwd(input_tensor, axes=None, _s=mod):
        if not hasattr(_s, '_frozen_axes'):
            if axes is None or axes.nelement() == 0:
                _s._frozen_axes = None
            else:
                _s._frozen_axes = sorted((int(v) for v in axes.tolist()), reverse=True)
        ax = _s._frozen_axes
        if ax is None:
            return torch.squeeze(input_tensor)
        result = input_tensor
        for a in ax:
            result = torch.squeeze(result, dim=a)
        return result

    mod.forward = fwd


def _patch_global_avg_pool(mod):
    """Replace ``GlobalAveragePool`` with ``adaptive_avg_pool2d(x, 1)``.

    ⚠️ This is a **compiler bug workaround**, not staticisation.

    When ``torch.mean(dim=(2,3), keepdim=True)`` directly follows an ``Add``,
    RBLN divides by ``W`` instead of ``H*W`` — the output comes out exactly
    ``H`` times too large (measured: H=6 -> 6.01x, H=2 -> 2.00x, H=1 -> correct,
    since there W == H*W). **The compile succeeds**, so the failure is silent:
    every timestep predicts blank and the model returns empty strings.

    Mathematically equivalent spellings are all fine (``sum/(H*W)``,
    ``avg_pool2d``, ``adaptive_avg_pool2d``, ``flatten(2).mean(-1)``);
    ``adaptive_avg_pool2d(x, 1)`` is used because it is shape-independent.
    """
    def fwd(input_tensor):
        return torch.nn.functional.adaptive_avg_pool2d(input_tensor, 1)

    mod.forward = fwd


# These patches depend on ``onnx2torch``'s **internal class names**. If a
# version bump renames one, the patch silently stops applying — and a missing
# GlobalAveragePool workaround means **the compile still succeeds and only the
# results are wrong**. So the targets are listed explicitly and any miss raises.
# Together with the pin in requirements this is two layers of defence.
_TARGETS = (
    ('onnx2torch.node_converters.reshape', 'OnnxReshape', _patch_reshape),
    ('onnx2torch.node_converters.slice', 'OnnxSlice', _patch_slice),
    ('onnx2torch.node_converters.squeeze', 'OnnxSqueezeDynamicAxes', _patch_squeeze),
    ('onnx2torch.node_converters.global_average_pool',
     'OnnxGlobalAveragePool', _patch_global_avg_pool),
    ('onnx2torch.node_converters.global_average_pool',
     'OnnxGlobalAveragePoolWithKnownInputShape', _patch_global_avg_pool),
)

VERIFIED_ONNX2TORCH = '1.5.15'      # the version this was validated against

_GAP_NAMES = ('OnnxGlobalAveragePool', 'OnnxGlobalAveragePoolWithKnownInputShape')


def _resolve_handlers():
    """Resolve every patch target, raising if any is missing (no silent no-op)."""
    import importlib

    handlers, missing = [], []
    for module_path, cls_name, patch in _TARGETS:
        try:
            cls = getattr(importlib.import_module(module_path), cls_name)
        except (ImportError, AttributeError):
            missing.append(f'{module_path}.{cls_name}')
            continue
        handlers.append((cls, patch))
    if missing:
        try:
            import importlib.metadata as md
            installed = md.version('onnx2torch')
        except Exception:
            installed = 'unknown'
        raise RuntimeError(
            'onnx2torch patch targets not found: ' + ', '.join(missing) + '\n'
            f'  installed {installed} / validated {VERIFIED_ONNX2TORCH}\n'
            '  Without these patches the compile still SUCCEEDS and only the '
            'results are wrong (missing GlobalAveragePool workaround). Pin '
            f'onnx2torch=={VERIFIED_ONNX2TORCH} or update the patches.'
        )
    return handlers


def make_static(model) -> dict:
    """Swap the problematic converters for static versions.

    Returns ``{converter_name: count}``.
    """
    handlers = _resolve_handlers()
    counts: dict[str, int] = {}
    for mod in model.modules():
        for cls, patch in handlers:
            if isinstance(mod, cls):
                patch(mod)
                counts[cls.__name__] = counts.get(cls.__name__, 0) + 1
    if not any(counts.get(name) for name in _GAP_NAMES):
        # PP-OCRv5 rec always has GlobalAveragePool in its SE blocks. Zero
        # matches means the graph is not what we think it is — and passing
        # silently would ship wrong numbers.
        raise RuntimeError(
            'No GlobalAveragePool converter found. The graph differs from '
            'what is expected, or onnx2torch uses a different class. '
            'Compiling without the workaround produces wrong results.'
        )
    return counts


def assert_equivalent(model, xs, ref, label='staticisation'):
    """Assert the patched model still matches ``ref`` bit-for-bit.

    Runs the model twice — the first call fills the constant caches.
    """
    with torch.no_grad():
        model(*xs)               # warm-up = populate caches
        out = model(*xs)
    out = out if isinstance(out, torch.Tensor) else out[0]
    d = float((ref - out).abs().max())
    if d != 0.0:
        raise AssertionError(f'{label} patches are not equivalent: max|delta|={d:.6g}')
    return d
