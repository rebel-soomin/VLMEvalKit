"""paddle -> ONNX -> static shape -> RBLN compile.

DBNet-family detection models use only standard CNN ops
(Conv/BN/ReLU/Resize/ConvTranspose), so nothing is unsupported. The only
work needed is **fixing the dynamic shapes** — the exported ONNX has a
fully dynamic ``[N,3,H,W]`` input, which RBLN cannot compile (static
shapes only).

Unlike the PP-OCRv5 *recognition* model, no numerical workarounds are
required here.
"""

from __future__ import annotations

from pathlib import Path


def export_onnx(model_dir: str, out_path: str, opset: int = 13) -> str:
    """Convert a paddle 3.x inference model (json + pdiparams) to ONNX.

    ``paddle2onnx`` 2.x requires a paddle runtime (the CPU build suffices
    for conversion).
    """
    import paddle2onnx

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    paddle2onnx.export(
        model_filename=str(Path(model_dir) / 'inference.json'),
        params_filename=str(Path(model_dir) / 'inference.pdiparams'),
        save_file=out_path,
        opset_version=opset,
        enable_onnx_checker=True,
    )
    return out_path


def fix_static_shape(onnx_path: str, out_path: str, input_name: str,
                     shape: list) -> str:
    """Pin the input shape statically.

    ``make_input_shape_fixed`` alone was enough for the detection model —
    no computed shape reaches a broadcast decision. (The recognition model
    needs ``onnxsim`` on top because symbolic dims survive.)
    """
    import onnx
    from onnxruntime.tools.onnx_model_utils import fix_output_shapes, make_input_shape_fixed

    model = onnx.load(onnx_path)
    make_input_shape_fixed(model.graph, input_name, shape)
    fix_output_shapes(model)
    onnx.save(model, out_path)
    return out_path


def compile_det(onnx_static_path: str, save_path: str, input_name: str,
                shape: tuple) -> str:
    """Compile a static ONNX graph into an RBLN artifact (no NPU needed)."""
    import onnx
    import rebel

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    compiled = rebel.compile_from_onnx(onnx.load(onnx_static_path),
                                       shape={input_name: tuple(shape)})
    compiled.save(save_path)
    return save_path
