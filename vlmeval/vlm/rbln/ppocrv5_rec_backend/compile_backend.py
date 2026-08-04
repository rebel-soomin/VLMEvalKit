"""paddle -> ONNX(opset 17) -> static -> onnx2torch -> RBLN compile.

Unlike the *detection* model (plain CNN, ``compile_from_onnx`` works
directly), recognition needs **two workarounds**. Both were pinned down by
measurement; see the package docstring and
:mod:`.onnx2torch_static`.

Workaround (1) lives here: export at **opset 17** so LayerNorm stays a
single fused ``LayerNormalization`` node, and enter through the **torch**
frontend. A *decomposed* LayerNorm (mean/sub/pow/sqrt/div) applied to a
``Conv`` output is rejected by both frontends, and ``compile_from_onnx``
re-decomposes opset-17 LayerNorm internally — so the torch frontend is not
a preference, it is the only path that compiles.

Workaround (2) lives in :func:`~.onnx2torch_static.make_static`.
"""

from __future__ import annotations

from pathlib import Path

# The ONNX graph's single input. PaddleOCR names it 'x' for both det and rec.
INPUT_NAME = 'x'


def export_onnx_op17(model_dir: str, out_path: str) -> str:
    """Convert a paddle 3.x inference model to ONNX at **opset 17**.

    The opset is not incidental: at opset 13 LayerNorm is emitted decomposed
    (mean/sub/pow/sqrt/div), and in that form — applied to a ``Conv`` output —
    the RBLN compiler rejects it. Opset 17 has a fused
    ``LayerNormalization`` node that ``onnx2torch`` maps to ``nn.LayerNorm``.

    ``paddle2onnx`` 2.x needs a paddle runtime (the CPU build is enough).
    """
    import paddle2onnx

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    paddle2onnx.export(
        model_filename=str(Path(model_dir) / 'inference.json'),
        params_filename=str(Path(model_dir) / 'inference.pdiparams'),
        save_file=out_path,
        opset_version=17,
        enable_onnx_checker=True,
    )
    return out_path


def simplify_static(onnx_path: str, out_path: str, shape: list,
                    input_name: str = INPUT_NAME) -> str:
    """Pin the input shape **and fold the computed shapes**.

    ``make_input_shape_fixed`` alone (which is all the detection model needs)
    is insufficient here: ``Reshape`` target shapes stay symbolic
    (``Reshape_*_o0__d2``) and get rejected during broadcast resolution.
    ``onnxsim`` constant-folds them.
    """
    import onnx
    import onnxsim

    model, ok = onnxsim.simplify(onnx.load(onnx_path),
                                 overwrite_input_shapes={input_name: list(shape)})
    if not ok:
        raise RuntimeError('onnxsim simplification failed')
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, out_path)
    return out_path


def _prepare_torch_module(onnx_static_path: str, shape, verbose: bool = True):
    """ONNX(static) -> patched ``torch.nn.Module``, equivalence checked.

    Returns ``(module, example_input)``. Raises unless the patched module is
    **bit-identical** to the unpatched one on a random input — without that
    check "the compile succeeded" guarantees nothing, since the
    GlobalAveragePool bug is silent.
    """
    import numpy as np
    import onnx
    import torch
    from onnx2torch import convert

    from .onnx2torch_static import assert_equivalent, make_static

    net = convert(onnx.load(onnx_static_path)).eval()
    x = torch.from_numpy(np.random.default_rng(0).random(tuple(shape), dtype=np.float32))
    with torch.no_grad():
        ref = net(x)
    ref = ref if isinstance(ref, torch.Tensor) else ref[0]

    counts = make_static(net)
    d = assert_equivalent(net, [x], ref)
    if verbose:
        print(f'  patches applied {counts} · equivalence max|delta|={d}')
    return net, x


def compile_rec(onnx_static_path: str, save_path: str, shape,
                input_name: str = INPUT_NAME, verbose: bool = True) -> str:
    """Compile a static recognition graph into an RBLN artifact (no NPU needed)."""
    import rebel
    import torch

    net, _ = _prepare_torch_module(onnx_static_path, shape, verbose=verbose)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    compiled = rebel.compile_from_torch(net, [(input_name, list(shape), torch.float32)])
    compiled.save(save_path)
    return save_path


def export_torch_onnx(onnx_static_path: str, save_path: str, shape,
                      input_name: str = INPUT_NAME, verbose: bool = True) -> str:
    """Re-export the **patched torch module** back to ONNX, for the GPU twin.

    This is :func:`compile_rec` stopped one step before
    ``rebel.compile_from_torch``, so the GPU runs the same graph the NPU
    compiles — including both workarounds. Re-exporting from torch rather
    than reusing ``onnx_static_path`` directly is what makes it the same
    graph: the ``adaptive_avg_pool2d`` substitution and the frozen
    Reshape/Slice/Squeeze constants are applied at the torch level, so an
    ONNX file taken from before that point would be a *different* graph.

    (On the NPU the substitution is a bug workaround; on the GPU it is
    mathematically equivalent to the original and therefore harmless — which
    is exactly why the equivalence assertion in
    :func:`_prepare_torch_module` matters.)
    """
    import torch

    net, x = _prepare_torch_module(onnx_static_path, shape, verbose=verbose)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            net, (x,), save_path,
            input_names=[input_name], output_names=['out'],
            opset_version=17, dynamo=False,
        )
    return save_path
